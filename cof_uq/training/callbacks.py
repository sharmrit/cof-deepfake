"""
Training callbacks: early stopping, checkpointing, LR scheduling.
"""

import numpy as np
from pathlib import Path
from ..models.factory import ModelFactory


class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.best = np.inf if mode == "min" else -np.inf
        self.counter = 0

    def on_epoch_end(self, epoch, metrics, model):
        val = metrics.get(self.metric, None)
        if val is None:
            return

        improved = (
            (self.mode == "min" and val < self.best - self.min_delta)
            or (self.mode == "max" and val > self.best + self.min_delta)
        )

        if improved:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return "stop"


class ModelCheckpoint:
    """Save model when a metric improves."""

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        arch_name: str = "model",
        metric: str = "val_auc",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.arch_name = arch_name
        self.metric = metric
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, epoch, metrics, model):
        val = metrics.get(self.metric, None)
        if val is None:
            return

        improved = (
            (self.mode == "min" and val < self.best)
            or (self.mode == "max" and val > self.best)
        )

        if improved:
            self.best = val
            path = self.save_dir / f"{self.arch_name}_best.pth"
            ModelFactory.save_checkpoint(
                model, str(path), epoch, metrics=metrics
            )


class LRSchedulerCallback:
    """Callback wrapper for LR schedulers."""

    def __init__(self, scheduler, metric: str = "val_loss"):
        self.scheduler = scheduler
        self.metric = metric

    def on_epoch_end(self, epoch, metrics, model):
        import torch
        val = metrics.get(self.metric)
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            if val is not None:
                self.scheduler.step(val)
        else:
            self.scheduler.step()
