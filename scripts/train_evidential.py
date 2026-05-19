#!/usr/bin/env python3
"""
Evidential Deep Learning — Dirichlet-based uncertainty for deepfake detection.

Replaces standard softmax with a Dirichlet output layer following
Sensoy et al. (2018). The model outputs concentration parameters
(evidence) for a Dirichlet distribution over class probabilities.

Usage:
    python -m scripts.train_evidential --arch xception --seed 42 --config configs/tifs.yaml
"""

import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from cof_uq.config import Config, ARCHITECTURES
from cof_uq.data.datasets import FaceForensicsDataset
from cof_uq.data.transforms import get_train_transforms, get_eval_transforms
from cof_uq.data.sampling import create_data_loaders

import timm


class EvidentialDetector(nn.Module):
    """
    Deepfake detector with Evidential Deep Learning output layer.

    Instead of softmax probabilities, outputs Dirichlet concentration
    parameters (evidence) from which uncertainty can be derived.
    """

    def __init__(self, arch_name, num_classes=2, pretrained=True):
        super().__init__()
        self.arch_name = arch_name
        self.num_classes = num_classes

        # Backbone
        if arch_name == "xception":
            self.backbone = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
            feature_dim = 2048
        elif arch_name == "resnet50":
            self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            feature_dim = 2048
        elif arch_name == "resnet101":
            self.backbone = timm.create_model("resnet101", pretrained=pretrained, num_classes=0)
            feature_dim = 2048
        elif arch_name == "efficientnet_b0":
            self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
            feature_dim = 1280
        elif arch_name == "efficientnet_b4":
            self.backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)
            feature_dim = 1792
        elif arch_name == "efficientnet_v2_s":
            self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=0)
            feature_dim = 1280
        elif arch_name == "vit_base_patch16_224":
            self.backbone = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
            feature_dim = 768
        elif arch_name == "deit_base_patch16_224":
            self.backbone = timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=0)
            feature_dim = 768
        elif arch_name == "swin_base_patch4_window7_224":
            self.backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0)
            feature_dim = 1024
        elif arch_name == "convnext_base":
            self.backbone = timm.create_model("convnext_base", pretrained=pretrained, num_classes=0)
            feature_dim = 1024
        elif arch_name == "maxvit_base_tf_224":
            self.backbone = timm.create_model("maxvit_base_tf_224.in1k", pretrained=pretrained, num_classes=0)
            feature_dim = 768
        else:
            raise ValueError("Unsupported architecture: {}".format(arch_name))

        self.feature_dim = feature_dim

        # Evidential output: produces evidence (non-negative) for each class
        self.evidence_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softplus(),  # Ensures non-negative evidence
        )

    def forward(self, x):
        """Returns evidence (Dirichlet concentration - 1)."""
        features = self.backbone(x)
        evidence = self.evidence_layer(features)
        return evidence

    def get_uncertainty(self, evidence):
        """
        Compute uncertainty from Dirichlet parameters.

        alpha = evidence + 1 (Dirichlet concentration)
        S = sum(alpha) (Dirichlet strength)
        uncertainty = num_classes / S (vacuity/total uncertainty)
        """
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S.squeeze(1)
        return prob, uncertainty


def evidential_loss(evidence, targets, epoch, n_epochs, annealing_step=10):
    """
    Evidential Deep Learning loss (Sensoy et al. 2018).

    Combines:
    1. Type II Maximum Likelihood (Bayes risk for cross-entropy)
    2. KL divergence regularizer (annealed)
    """
    num_classes = evidence.shape[1]
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)

    # One-hot encode targets
    one_hot = F.one_hot(targets, num_classes).float()

    # Type II ML loss: E[CE] under Dirichlet
    loss_ce = torch.sum(
        one_hot * (torch.digamma(S) - torch.digamma(alpha)),
        dim=1
    )

    # KL divergence regularizer
    # Remove evidence for correct class to only penalize incorrect evidence
    alpha_tilde = one_hot + (1 - one_hot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

    # KL(Dir(alpha_tilde) || Dir(1))
    kl = (
        torch.lgamma(S_tilde.squeeze(1))
        - torch.sum(torch.lgamma(alpha_tilde), dim=1)
        - torch.lgamma(torch.tensor(float(num_classes), device=evidence.device))
        + torch.sum(
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1,
        )
    )

    # Annealing coefficient
    annealing = min(1.0, epoch / annealing_step)

    loss = torch.mean(loss_ce + annealing * kl)
    return loss


def train_evidential(arch_name, seed, config):
    """Train an Evidential DL model."""
    print("\n" + "=" * 70)
    print("Evidential DL: {} | Seed {}".format(arch_name, seed))
    print("=" * 70)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config.ensure_dirs()

    # Data
    train_ds = FaceForensicsDataset(
        root=config.data.ff_root,
        split="train",
        transform=get_train_transforms(config.data.image_size),
        max_samples_per_class=config.data.max_samples_per_class,
    )
    val_ds = FaceForensicsDataset(
        root=config.data.ff_root,
        split="val",
        transform=get_eval_transforms(config.data.image_size),
        max_samples_per_class=config.data.max_samples_per_class,
    )

    train_loader, val_loader = create_data_loaders(
        train_ds, val_ds,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
        balanced=config.train.balanced_sampling,
    )

    print("  Train: {} | Val: {}".format(len(train_ds), len(val_ds)))

    # Model
    device = config.device
    model = EvidentialDetector(arch_name, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  Parameters: {:,}".format(n_params))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr,
                                 weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=1e-7
    )
    scaler = GradScaler()

    best_auc = 0.0
    patience_counter = 0
    ckpt_dir = Path(config.checkpoint_dir) / "evidential"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.train.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Epoch {}".format(epoch))
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                evidence = model(images)
                loss = evidential_loss(evidence, labels, epoch, config.train.epochs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            alpha = evidence + 1.0
            preds = alpha.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        scheduler.step()

        # Validate
        model.eval()
        all_probs = []
        all_labels = []
        all_uncertainties = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                evidence = model(images)
                prob, uncertainty = model.get_uncertainty(evidence)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(labels.numpy())
                all_uncertainties.append(uncertainty.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_uncertainties = np.concatenate(all_uncertainties)

        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            val_auc = 0.5

        val_acc = (all_probs.argmax(axis=1) == all_labels).mean()

        # Check if uncertainty is constant (known failure mode)
        unc_std = np.std(all_uncertainties)

        lr = optimizer.param_groups[0]["lr"]
        print("Epoch {}/{} | LR: {:.2e} | Loss: {:.4f} | Val Acc: {:.4f} | "
              "Val AUC: {:.4f} | Unc Std: {:.6f}".format(
                  epoch, config.train.epochs, lr,
                  total_loss / total, val_acc, val_auc, unc_std))

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            ckpt_path = ckpt_dir / "{}_seed{}_best.pth".format(arch_name, seed)
            torch.save({
                "arch_name": arch_name,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "unc_std": unc_std,
            }, str(ckpt_path))
        else:
            patience_counter += 1
            if patience_counter >= config.train.early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    # Save final
    final_path = ckpt_dir / "{}_seed{}_final.pth".format(arch_name, seed)
    torch.save({
        "arch_name": arch_name,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "val_auc": val_auc,
        "unc_std": unc_std,
    }, str(final_path))

    print("  Best Val AUC: {:.4f}".format(best_auc))
    print("  Final Unc Std: {:.6f}".format(unc_std))
    if unc_std < 1e-6:
        print("  WARNING: Constant uncertainty output detected!")
    print("  Saved: {}".format(final_path))


def main():
    parser = argparse.ArgumentParser(description="Train Evidential DL")
    parser.add_argument("--config", type=str, default="configs/tifs.yaml")
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.device = args.device

    train_evidential(args.arch, args.seed, config)


if __name__ == "__main__":
    main()
