#!/usr/bin/env python3
"""
Train deepfake detection models — TIFS version.
Uses 60/20/20 train/val/test split matching paper protocol.
"""

import argparse
import time
import numpy as np
from pathlib import Path

import torch

from cof_uq.config import Config, ARCHITECTURES, SEEDS
from cof_uq.models.factory import ModelFactory
from cof_uq.data.datasets import FaceForensicsDataset
from cof_uq.data.transforms import get_train_transforms, get_eval_transforms
from cof_uq.data.sampling import create_data_loaders
from cof_uq.training.trainer import Trainer


def train_single(arch_name: str, seed: int, config: Config):
    """Train a single architecture with a given seed."""
    print(f"\n{'='*70}")
    print(f"Training {arch_name} | Seed {seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config.ensure_dirs()

    # Data — 60/20/20 split matching paper Section IV.F
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

    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    model = ModelFactory.create(
        arch_name,
        mc_dropout_rate=config.train.mc_dropout_rate,
        pretrained=True,
        device=config.device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"  MC Dropout rate: {config.train.mc_dropout_rate}")

    # Train
    trainer = Trainer(model, config, device=config.device)
    result = trainer.fit(train_loader, val_loader)

    # Save final checkpoint
    ckpt_path = (
        Path(config.checkpoint_dir) / "{}_seed{}_final.pth".format(arch_name, seed)
    )
    ModelFactory.save_checkpoint(
        model, str(ckpt_path), epoch=len(result["history"]),
        metrics={"best_val_auc": result["best_val_auc"]},
    )
    print("  Saved: {}".format(ckpt_path))
    print("  Best Val AUC: {:.4f}".format(result["best_val_auc"]))

    return result


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detectors")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-archs", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config.device = args.device

    archs = ARCHITECTURES if args.all_archs else [args.arch or "xception"]
    seeds = SEEDS if args.all_seeds else [args.seed]

    total = len(archs) * len(seeds)
    print("Training plan: {} architectures x {} seeds = {} runs".format(
        len(archs), len(seeds), total))

    start = time.time()
    for arch in archs:
        for seed in seeds:
            train_single(arch, seed, config)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("All training complete in {:.1f} hours".format(elapsed / 3600))


if __name__ == "__main__":
    main()
