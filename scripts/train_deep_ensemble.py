#!/usr/bin/env python3
"""
Deep Ensemble Training — Trains M independent models per architecture.

Each model uses a different random seed for weight initialization,
data shuffling, and dropout masks, following Lakshminarayanan et al. (2017).

Usage:
    python -m scripts.train_deep_ensemble --arch xception --member 0 --config configs/tifs.yaml
    python -m scripts.train_deep_ensemble --arch xception --member 0 --n-members 5
"""

import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from cof_uq.config import Config, ARCHITECTURES
from cof_uq.models.factory import ModelFactory
from cof_uq.data.datasets import FaceForensicsDataset
from cof_uq.data.transforms import get_train_transforms, get_eval_transforms
from cof_uq.data.sampling import create_data_loaders
from cof_uq.training.trainer import Trainer


# Deep Ensemble seeds: each member gets a unique seed
ENSEMBLE_SEEDS = [100, 200, 300, 400, 500]


def train_ensemble_member(arch_name, member_id, config, n_members=5):
    """Train one member of a Deep Ensemble."""
    seed = ENSEMBLE_SEEDS[member_id]

    print("\n" + "=" * 70)
    print("Deep Ensemble: {} | Member {}/{} | Seed {}".format(
        arch_name, member_id + 1, n_members, seed))
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

    print("  Train: {} samples | Val: {} samples".format(len(train_ds), len(val_ds)))

    # Model — fresh initialization (no MC Dropout needed for ensembles)
    model = ModelFactory.create(
        arch_name,
        mc_dropout_rate=0.0,  # No dropout for ensemble members
        pretrained=True,
        device=config.device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  Parameters: {:,}".format(n_params))

    # Train
    trainer = Trainer(model, config, device=config.device)
    result = trainer.fit(train_loader, val_loader)

    # Save checkpoint
    ckpt_dir = Path(config.checkpoint_dir) / "deep_ensemble" / arch_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "member{}_seed{}.pth".format(member_id, seed)

    ModelFactory.save_checkpoint(
        model, str(ckpt_path), epoch=len(result["history"]),
        metrics={"best_val_auc": result["best_val_auc"]},
    )
    print("  Saved: {}".format(ckpt_path))
    print("  Best Val AUC: {:.4f}".format(result["best_val_auc"]))

    return result


def main():
    parser = argparse.ArgumentParser(description="Train Deep Ensemble member")
    parser.add_argument("--config", type=str, default="configs/tifs.yaml")
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--member", type=int, required=True, help="Member index (0-4)")
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.device = args.device

    train_ensemble_member(args.arch, args.member, config, args.n_members)


if __name__ == "__main__":
    main()
