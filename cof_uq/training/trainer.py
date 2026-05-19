"""
Training pipeline for deepfake detection models.

Features:
  - Balanced real/fake sampling
  - Label smoothing
  - Cosine / step / plateau LR scheduling
  - Mixed precision training
  - Warmup epochs with frozen backbone
  - Comprehensive logging
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List, Callable
from pathlib import Path
from tqdm import tqdm

from ..models.architectures import DeepfakeDetector
from ..models.factory import ModelFactory
from ..config import Config
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


class Trainer:
    """
    Complete training loop for deepfake detection models.

    Parameters
    ----------
    model : DeepfakeDetector
    config : Config
    device : str
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        config: Config,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        # Loss with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.train.label_smoothing
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        # LR Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = device == "cuda"

        # History
        self.history: List[Dict] = []

    def _build_scheduler(self):
        cfg = self.config.train
        total_steps = cfg.epochs
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-7
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        return None

    def _freeze_backbone(self):
        """Freeze backbone parameters during warmup."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters after warmup."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auc = 0.5

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "auc": auc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List] = None,
    ) -> Dict:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        callbacks : list of callback objects

        Returns
        -------
        history : dict with training and validation metrics per epoch
        """
        cfg = self.config.train
        callbacks = callbacks or [
            EarlyStopping(patience=cfg.early_stopping_patience, metric="val_loss"),
            ModelCheckpoint(
                save_dir=self.config.checkpoint_dir,
                arch_name=self.model.arch_name,
                metric="val_auc",
                mode="max",
            ),
        ]

        # Backbone freeze warmup
        if cfg.freeze_backbone_epochs > 0:
            self._freeze_backbone()

        start_time = time.time()

        for epoch in range(1, cfg.epochs + 1):
            # Unfreeze backbone after warmup
            if epoch == cfg.freeze_backbone_epochs + 1:
                self._unfreeze_backbone()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # LR scheduling
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record
            epoch_record = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
            }
            self.history.append(epoch_record)

            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"LR: {lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

            # Run callbacks
            stop = False
            for cb in callbacks:
                result = cb.on_epoch_end(epoch, epoch_record, self.model)
                if result == "stop":
                    stop = True

            if stop:
                print(f"Early stopping at epoch {epoch}.")
                break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s")

        return {
            "history": self.history,
            "total_time": total_time,
            "best_val_auc": max(h["val_auc"] for h in self.history),
        }
