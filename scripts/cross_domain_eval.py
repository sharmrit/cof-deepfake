#!/usr/bin/env python3
"""
Cross-domain evaluation across all architectures and datasets.

Usage:
    python -m scripts.cross_domain_eval --seed 42
    python -m scripts.cross_domain_eval --all-archs
"""

import argparse
import json
import numpy as np
from pathlib import Path

from cof_uq.config import Config, ARCHITECTURES, DATASETS
from cof_uq.evaluation.cross_domain import CrossDomainEvaluator
from cof_uq.visualization.plots import plot_cross_domain_heatmap


def main():
    parser = argparse.ArgumentParser(description="Cross-domain evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-archs", action="store_true")
    parser.add_argument("--arch", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config) if args.config else Config()
    config.ensure_dirs()

    archs = ARCHITECTURES if args.all_archs else [args.arch or "xception"]
    evaluator = CrossDomainEvaluator(config=config)

    all_source_data = {}
    all_target_data = {}

    for arch in archs:
        unc_dir = Path(config.output_dir) / "uncertainties" / arch

        # Load source (FF++) data
        src_path = unc_dir / f"faceforensics_seed{args.seed}.npz"
        if not src_path.exists():
            print(f"Skipping {arch}: no FF++ data at {src_path}")
            continue

        src = np.load(src_path)
        all_source_data[arch] = {
            "uncertainties": src["uncertainties"],
            "errors": src["errors"],
            "probs": src["probs"],
            "labels": src["labels"],
        }

        # Load target datasets
        targets = {}
        for ds in ["celebdf", "dfdc"]:
            tgt_path = unc_dir / f"{ds}_seed{args.seed}.npz"
            if tgt_path.exists():
                tgt = np.load(tgt_path)
                targets[ds] = {
                    "uncertainties": tgt["uncertainties"],
                    "errors": tgt["errors"],
                    "probs": tgt["probs"],
                    "labels": tgt["labels"],
                }
            else:
                print(f"  Warning: {ds} data not found for {arch}")

        all_target_data[arch] = targets

    if not all_source_data:
        print("No data found. Run extract_uncertainty first.")
        return

    # Run evaluation
    results = evaluator.evaluate_all_architectures(
        all_source_data, all_target_data
    )

    # Print summary
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Architecture':<15s} {'FF++ (ρ)':>10s} {'CelebDF (ρ)':>12s} {'DFDC (ρ)':>10s} {'Degradation':>12s}")
    print("-" * 60)

    for arch, res in results["per_architecture"].items():
        ds = res["datasets"]
        ff_corr = ds.get("faceforensics", {}).get("correlation", 0)
        cel_corr = ds.get("celebdf", {}).get("correlation", 0)
        dfdc_corr = ds.get("dfdc", {}).get("correlation", 0)
        deg = ds.get("celebdf", {}).get("degradation_pct", 0)
        from cof_uq.config import ARCH_SHORT_NAMES
        name = ARCH_SHORT_NAMES.get(arch, arch)
        print(f"{name:<15s} {ff_corr:>10.3f} {cel_corr:>12.3f} {dfdc_corr:>10.3f} {deg:>11.1f}%")

    print(f"\nAggregate:")
    agg = results
    print(f"  In-domain mean ρ: {agg['in_domain']['mean_correlation']:.3f} ± {agg['in_domain']['std_correlation']:.3f}")
    for ds in ["celebdf", "dfdc"]:
        ood = agg["out_of_domain"][ds]
        print(f"  {ds.upper()} mean ρ: {ood['mean_correlation']:.3f} | "
              f"Degradation: {ood['mean_degradation_pct']:.1f}% | "
              f"Inversions: {ood['inversion_count']}")

    # Save
    save_path = Path(config.output_dir) / "cross_domain_results.json"
    evaluator.save_results(results, str(save_path))
    print(f"\nResults saved: {save_path}")

    # Plot heatmap
    heatmap_data = {}
    for arch, res in results["per_architecture"].items():
        heatmap_data[arch] = {
            ds: res["datasets"].get(ds, {}).get("correlation", 0)
            for ds in DATASETS
        }
    plot_cross_domain_heatmap(
        heatmap_data,
        save_path=str(Path(config.figure_dir) / "cross_domain_heatmap.pdf"),
    )
    print("Saved cross-domain heatmap.")


if __name__ == "__main__":
    main()
