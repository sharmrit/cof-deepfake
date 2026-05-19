#!/usr/bin/env python3
"""
Aggregate all results across seeds and generate TIFS paper tables.

Run after all multi-seed jobs complete:
    python scripts/aggregate_results.py --config configs/tifs.yaml
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict

import sys
sys.path.insert(0, ".")

from cof_uq.config import Config, ARCHITECTURES, ARCH_SHORT_NAMES, UNCERTAINTY_SOURCES, SEEDS
from cof_uq.fusion.cof import CorrelationOptimizedFusion
from scripts.run_all_fusion import run_all_12_methods


def load_data(arch, seed, config):
    """Load extracted uncertainty data."""
    base = Path(config.output_dir) / "uncertainties" / arch
    result = {}
    for ds in ["faceforensics", "celebdf", "dfdc"]:
        fp = base / "{}_seed{}.npz".format(ds, seed)
        if fp.exists():
            result[ds] = dict(np.load(fp))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tifs.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    seeds = config.seeds

    print("=" * 80)
    print("TIFS RESULTS AGGREGATION")
    print("Seeds: {}".format(seeds))
    print("=" * 80)

    # Find available seeds per architecture
    available = {}
    for arch in ARCHITECTURES:
        arch_seeds = []
        for seed in seeds:
            fp = Path(config.output_dir) / "uncertainties" / arch / "faceforensics_seed{}.npz".format(seed)
            if fp.exists():
                arch_seeds.append(seed)
        if arch_seeds:
            available[arch] = arch_seeds
    
    print("\nAvailable data:")
    for arch, s in available.items():
        name = ARCH_SHORT_NAMES.get(arch, arch)
        print("  {:15s}: seeds {}".format(name, s))

    # =====================================================================
    # TABLE IV: 12-Method Ranking (mean across architectures and seeds)
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("TABLE IV: Fusion Methods Ranked by Mean rho")
    print("=" * 80)

    method_scores = defaultdict(list)

    for arch in available:
        for seed in available[arch]:
            data = load_data(arch, seed, config)
            if "faceforensics" not in data:
                continue
            U = data["faceforensics"]["uncertainties"]
            errors = data["faceforensics"]["errors"]
            n = len(errors)
            np.random.seed(seed)
            idx = np.random.permutation(n)
            split = int(0.75 * n)

            results = run_all_12_methods(
                U[idx[:split]], errors[idx[:split]],
                U[idx[split:]], errors[idx[split:]],
                n_restarts=20,
            )
            for r in results:
                method_scores[r["name"]].append(r["correlation"])

    # Sort by mean
    method_stats = {}
    for name, scores in method_scores.items():
        method_stats[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "cv": np.std(scores) / abs(np.mean(scores)) * 100 if abs(np.mean(scores)) > 1e-12 else 0,
        }

    ranked = sorted(method_stats.items(), key=lambda x: -x[1]["mean"])
    print("\n{:4s} {:20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Rank", "Method", "Mean", "Std", "Min", "Max", "CV%"))
    print("-" * 70)
    for i, (name, s) in enumerate(ranked):
        proposed = "*" if name in ["COF", "L1-COF", "Meta-Ens.", "2M-Ens.", "Hier-Fus.", "SC-Weight"] else " "
        print("{:3d}{} {:20s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>7.1f}".format(
            i+1, proposed, name, s["mean"], s["std"], s["min"], s["max"], s["cv"]))

    # =====================================================================
    # TABLE: Individual Source Correlations
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("TABLE: Individual Source Correlations (mean across seeds)")
    print("=" * 80)

    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Architecture", "Epist", "Aleat", "Calib", "Conf", "Distr", "COF-5"))
    print("-" * 75)

    for arch in available:
        name = ARCH_SHORT_NAMES.get(arch, arch)
        seed_corrs = []
        seed_cof = []

        for seed in available[arch]:
            data = load_data(arch, seed, config)
            if "faceforensics" not in data:
                continue
            U = data["faceforensics"]["uncertainties"]
            errors = data["faceforensics"]["errors"]

            corrs = []
            for i in range(5):
                if np.std(U[:, i]) > 1e-12:
                    c, _ = pearsonr(U[:, i], errors)
                else:
                    c = 0.0
                corrs.append(c)
            seed_corrs.append(corrs)

            cof = CorrelationOptimizedFusion(k_sources=5, n_restarts=20)
            cof.fit(U, errors)
            seed_cof.append(cof.result_.correlation)

        mean_corrs = np.mean(seed_corrs, axis=0)
        mean_cof = np.mean(seed_cof)
        print("{:15s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            name, mean_corrs[0], mean_corrs[1], mean_corrs[2],
            mean_corrs[3], mean_corrs[4], mean_cof))

    # =====================================================================
    # TABLE: Cross-Domain Generalization
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("TABLE: Cross-Domain Generalization (mean across seeds)")
    print("=" * 80)

    print("\n{:15s} {:>8s} {:>10s} {:>8s} {:>10s}".format(
        "Architecture", "FF++", "CelebDF", "DFDC", "Avg Drop%"))
    print("-" * 55)

    for arch in available:
        name = ARCH_SHORT_NAMES.get(arch, arch)
        ff_corrs, cel_corrs, dfdc_corrs = [], [], []

        for seed in available[arch]:
            data = load_data(arch, seed, config)
            if "faceforensics" not in data:
                continue
            U_ff = data["faceforensics"]["uncertainties"]
            e_ff = data["faceforensics"]["errors"]

            cof = CorrelationOptimizedFusion(k_sources=5, n_restarts=20)
            cof.fit(U_ff, e_ff)
            ff_corrs.append(cof.result_.correlation)

            for ds, store in [("celebdf", cel_corrs), ("dfdc", dfdc_corrs)]:
                if ds in data:
                    fused = cof.predict(data[ds]["uncertainties"])
                    if np.std(fused) > 1e-12:
                        c, _ = pearsonr(fused, data[ds]["errors"])
                    else:
                        c = 0.0
                    store.append(c)

        ff_mean = np.mean(ff_corrs) if ff_corrs else 0
        cel_mean = np.mean(cel_corrs) if cel_corrs else 0
        dfdc_mean = np.mean(dfdc_corrs) if dfdc_corrs else 0
        avg_ext = np.mean([cel_mean, dfdc_mean])
        avg_drop = (ff_mean - avg_ext) / abs(ff_mean) * 100 if abs(ff_mean) > 1e-12 else 0

        print("{:15s} {:>8.4f} {:>10.4f} {:>8.4f} {:>9.1f}%".format(
            name, ff_mean, cel_mean, dfdc_mean, avg_drop))

    # =====================================================================
    # TABLE: Learned Weights
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("TABLE: COF-5 Learned Weights (mean across seeds)")
    print("=" * 80)

    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Architecture", "Epist", "Aleat", "Calib", "Conf", "Distr"))
    print("-" * 60)

    for arch in available:
        name = ARCH_SHORT_NAMES.get(arch, arch)
        all_weights = []

        for seed in available[arch]:
            data = load_data(arch, seed, config)
            if "faceforensics" not in data:
                continue
            U = data["faceforensics"]["uncertainties"]
            errors = data["faceforensics"]["errors"]
            cof = CorrelationOptimizedFusion(k_sources=5, n_restarts=20)
            cof.fit(U, errors)
            all_weights.append(cof.weights_)

        mean_w = np.mean(all_weights, axis=0)
        std_w = np.std(all_weights, axis=0)
        print("{:15s} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}".format(
            name, mean_w[0], mean_w[1], mean_w[2], mean_w[3], mean_w[4]))

    # =====================================================================
    # TABLE: Multi-Seed Stability
    # =====================================================================
    print("\n\n" + "=" * 80)
    print("TABLE: Multi-Seed Stability (COF-5)")
    print("=" * 80)

    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>10s}".format(
        "Architecture", "Mean", "Std", "CV%", "Seeds"))
    print("-" * 55)

    for arch in available:
        name = ARCH_SHORT_NAMES.get(arch, arch)
        cof_corrs = []

        for seed in available[arch]:
            data = load_data(arch, seed, config)
            if "faceforensics" not in data:
                continue
            U = data["faceforensics"]["uncertainties"]
            errors = data["faceforensics"]["errors"]
            cof = CorrelationOptimizedFusion(k_sources=5, n_restarts=20)
            cof.fit(U, errors)
            cof_corrs.append(cof.result_.correlation)

        mean = np.mean(cof_corrs)
        std = np.std(cof_corrs)
        cv = std / abs(mean) * 100 if abs(mean) > 1e-12 else 0
        print("{:15s} {:>8.4f} {:>8.4f} {:>7.1f}% {:>10d}".format(
            name, mean, std, cv, len(cof_corrs)))

    print("\n\nAggregation complete.")


if __name__ == "__main__":
    main()
