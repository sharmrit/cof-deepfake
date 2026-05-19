#!/usr/bin/env python3
"""
Extract all five uncertainty sources from trained models — TIFS version.
Always fits reference distribution from FF++ training data.
"""

import argparse
import numpy as np
from pathlib import Path

import torch

from cof_uq.config import Config, ARCHITECTURES, DATASETS, SEEDS
from cof_uq.models.factory import ModelFactory
from cof_uq.data.datasets import get_dataset
from cof_uq.data.transforms import get_eval_transforms
from cof_uq.uncertainty.extraction import UncertaintyExtractor
from cof_uq.uncertainty.normalization import MinMaxNormalizer


def extract_single(arch_name, dataset_name, seed, config):
    """Extract uncertainties for one arch+dataset+seed."""
    print("\nExtracting: {} | {} | seed={}".format(arch_name, dataset_name, seed))

    # Load model
    ckpt_path = Path(config.checkpoint_dir) / "{}_seed{}_final.pth".format(arch_name, seed)
    if not ckpt_path.exists():
        ckpt_path = Path(config.checkpoint_dir) / "{}_best.pth".format(arch_name)
    if not ckpt_path.exists():
        print("  WARNING: No checkpoint found for {} seed={}. Skipping.".format(arch_name, seed))
        return

    model = ModelFactory.load_checkpoint(
        str(ckpt_path), device=config.device,
        mc_dropout_rate=config.train.mc_dropout_rate,
    )

    # Dataset root
    dataset_roots = {
        "faceforensics": config.data.ff_root,
        "celebdf": config.data.celebdf_root,
        "dfdc": config.data.dfdc_root,
    }
    root = dataset_roots[dataset_name]
    transform = get_eval_transforms(config.data.image_size)

    # Extractor
    extractor = UncertaintyExtractor(
        model,
        n_mc_passes=config.train.mc_forward_passes,
        device=config.device,
    )

    # ALWAYS fit reference distribution from FF++ training data
    ff_root = config.data.ff_root
    train_ds = get_dataset("faceforensics", ff_root, split="train", transform=transform,
                           max_samples_per_class=config.data.max_samples_per_class)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
    )
    print("  Fitting reference distribution ({} samples)...".format(len(train_ds)))
    extractor.fit_reference_distribution(train_loader)

    # Test dataset
    test_ds = get_dataset(dataset_name, root, split="test", transform=transform,
                          max_samples_per_class=config.data.max_samples_per_class)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
    )

    # Output directory
    save_dir = Path(config.output_dir) / "uncertainties" / arch_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Normalize using training stats
    normalizer = MinMaxNormalizer()
    is_source = dataset_name == "faceforensics"

    if not is_source:
        norm_path = save_dir / "normalizer_seed{}.json".format(seed)
        if norm_path.exists():
            normalizer = MinMaxNormalizer.load(str(norm_path))
        else:
            normalizer = None

    # Extract
    save_path = save_dir / "{}_seed{}.npz".format(dataset_name, seed)
    results = extractor.extract_and_save(
        test_loader, str(save_path),
        normalizer=normalizer,
        fit_normalizer=is_source,
    )

    # Save normalizer stats if source dataset
    if is_source and normalizer is not None:
        norm_path = save_dir / "normalizer_seed{}.json".format(seed)
        normalizer.save(str(norm_path))

    n_samples = len(results["errors"])
    error_rate = results["errors"].mean()
    print("  Saved: {}".format(save_path))
    print("  Samples: {} | Error rate: {:.3f}".format(n_samples, error_rate))

    # Individual source correlations
    from scipy.stats import pearsonr
    for src in ["epistemic", "aleatoric", "calibration", "conformal", "distributional"]:
        if src in results and len(results[src]) == len(results["errors"]):
            vals = results[src]
            if np.std(vals) > 1e-12:
                corr, _ = pearsonr(vals, results["errors"])
                print("  {:15s}: rho = {:.4f}".format(src, corr))


def main():
    parser = argparse.ArgumentParser(description="Extract uncertainties")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="faceforensics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-archs", action="store_true")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config.device = args.device
    config.ensure_dirs()

    archs = ARCHITECTURES if args.all_archs else [args.arch or "xception"]
    datasets = DATASETS if args.all_datasets else [args.dataset]
    seeds = [args.seed]

    for arch in archs:
        for ds in datasets:
            for seed in seeds:
                extract_single(arch, ds, seed, config)


if __name__ == "__main__":
    main()
