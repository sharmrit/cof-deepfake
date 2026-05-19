#!/usr/bin/env python3
"""
Extract uncertainty from Deep Ensembles and Evidential DL models.

Computes:
  - Deep Ensemble: predictive entropy from M=5 independent models
  - Evidential DL: vacuity (K/S) from Dirichlet parameters

Evaluates Pearson correlation with prediction errors for Table V.

Usage:
    python -m scripts.eval_baselines --config configs/tifs.yaml
    python -m scripts.eval_baselines --arch xception
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cof_uq.config import Config, ARCHITECTURES, ARCH_SHORT_NAMES
from cof_uq.models.factory import ModelFactory
from cof_uq.data.datasets import FaceForensicsDataset
from cof_uq.data.transforms import get_eval_transforms
from scripts.train_evidential import EvidentialDetector


def safe_corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    return pearsonr(x, y)


def eval_deep_ensemble(arch_name, config, n_members=5, seed=42):
    """Evaluate Deep Ensemble uncertainty for one architecture."""
    device = config.device
    ckpt_dir = Path(config.checkpoint_dir) / "deep_ensemble" / arch_name
    transform = get_eval_transforms(config.data.image_size)

    # Load test data
    test_ds = FaceForensicsDataset(
        root=config.data.ff_root,
        split="test",
        transform=transform,
        max_samples_per_class=config.data.max_samples_per_class,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
    )

    # Collect predictions from each member
    all_member_probs = []
    members_found = 0

    from cof_uq.models.factory import ModelFactory
    ENSEMBLE_SEEDS = [100, 200, 300, 400, 500]

    for m in range(n_members):
        seed_m = ENSEMBLE_SEEDS[m]
        ckpt_path = ckpt_dir / "member{}_seed{}.pth".format(m, seed_m)
        if not ckpt_path.exists():
            # Try best checkpoint
            ckpt_path = ckpt_dir / "member{}_seed{}_best.pth".format(m, seed_m)
        if not ckpt_path.exists():
            print("  Missing member {} for {}".format(m, arch_name))
            continue

        model = ModelFactory.load_checkpoint(
            str(ckpt_path), device=device, mc_dropout_rate=0.0
        )
        model.eval()
        members_found += 1

        member_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                member_probs.append(probs.cpu().numpy())

        all_member_probs.append(np.concatenate(member_probs, axis=0))
        del model
        torch.cuda.empty_cache()

    if members_found < 2:
        print("  Not enough members for {} ({} found)".format(arch_name, members_found))
        return None

    # Stack: (M, N, C)
    ensemble_probs = np.stack(all_member_probs, axis=0)

    # Mean prediction
    mean_probs = np.mean(ensemble_probs, axis=0)  # (N, C)
    predictions = np.argmax(mean_probs, axis=1)

    # Ground truth
    labels = np.array(test_ds.labels)[:len(predictions)]
    errors = (predictions != labels).astype(float)

    # Deep Ensemble uncertainty: predictive entropy
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

    # Also compute mutual information (epistemic) = H[E[p]] - E[H[p]]
    member_entropies = -np.sum(
        ensemble_probs * np.log(ensemble_probs + 1e-12), axis=2
    )  # (M, N)
    mean_member_entropy = np.mean(member_entropies, axis=0)  # (N,)
    mutual_info = entropy - mean_member_entropy  # epistemic

    corr_entropy, p_entropy = safe_corr(entropy, errors)
    corr_mi, p_mi = safe_corr(mutual_info, errors)

    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(labels, mean_probs[:, 1])
    acc = accuracy_score(labels, predictions)

    print("  Deep Ensemble ({} members): rho={:.4f} (entropy), "
          "rho={:.4f} (MI), AUC={:.4f}, Acc={:.4f}".format(
              members_found, corr_entropy, corr_mi, auc, acc))

    return {
        "method": "deep_ensemble",
        "architecture": arch_name,
        "n_members": members_found,
        "correlation_entropy": float(corr_entropy),
        "correlation_mutual_info": float(corr_mi),
        "auc": float(auc),
        "accuracy": float(acc),
        "n_samples": len(errors),
        "error_rate": float(errors.mean()),
    }


def eval_evidential(arch_name, config, seed=42):
    """Evaluate Evidential DL uncertainty for one architecture."""
    device = config.device
    ckpt_dir = Path(config.checkpoint_dir) / "evidential"
    transform = get_eval_transforms(config.data.image_size)

    # Find checkpoint
    ckpt_path = ckpt_dir / "{}_seed{}_best.pth".format(arch_name, seed)
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "{}_seed{}_final.pth".format(arch_name, seed)
    if not ckpt_path.exists():
        print("  No evidential checkpoint for {}".format(arch_name))
        return None

    # Load model
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = EvidentialDetector(arch_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load test data
    test_ds = FaceForensicsDataset(
        root=config.data.ff_root,
        split="test",
        transform=transform,
        max_samples_per_class=config.data.max_samples_per_class,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.train.batch_size,
        shuffle=False, num_workers=config.data.num_workers,
    )

    all_probs = []
    all_uncertainties = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evidential eval"):
            images = images.to(device)
            evidence = model(images)
            prob, uncertainty = model.get_uncertainty(evidence)
            all_probs.append(prob.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    uncertainties = np.concatenate(all_uncertainties)
    labels = np.concatenate(all_labels)
    predictions = np.argmax(probs, axis=1)
    errors = (predictions != labels).astype(float)

    # Check for constant output
    unc_std = np.std(uncertainties)
    if unc_std < 1e-6:
        print("  Evidential {}: CONSTANT OUTPUT (nan)".format(arch_name))
        return {
            "method": "evidential",
            "architecture": arch_name,
            "correlation": float("nan"),
            "constant_output": True,
            "unc_std": float(unc_std),
        }

    corr, pval = safe_corr(uncertainties, errors)

    from sklearn.metrics import roc_auc_score, accuracy_score
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.5
    acc = accuracy_score(labels, predictions)

    print("  Evidential {}: rho={:.4f}, AUC={:.4f}, Unc Std={:.6f}".format(
        arch_name, corr, auc, unc_std))

    return {
        "method": "evidential",
        "architecture": arch_name,
        "correlation": float(corr),
        "auc": float(auc),
        "accuracy": float(acc),
        "constant_output": False,
        "unc_std": float(unc_std),
        "n_samples": len(errors),
        "error_rate": float(errors.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate UQ baselines")
    parser.add_argument("--config", type=str, default="configs/tifs.yaml")
    parser.add_argument("--arch", type=str, default=None, help="Single arch or all")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    config.device = args.device

    archs = [args.arch] if args.arch else ARCHITECTURES

    all_results = {"deep_ensemble": {}, "evidential": {}}

    print("=" * 70)
    print("UQ BASELINE EVALUATION")
    print("=" * 70)

    # Also compute MC Dropout and COF for comparison
    from cof_uq.fusion.cof import CorrelationOptimizedFusion

    print("\n{:15s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Architecture", "MC Drop.", "DeepEns.", "Evid.DL", "COF-5"))
    print("-" * 60)

    for arch in archs:
        name = ARCH_SHORT_NAMES.get(arch, arch)

        # MC Dropout (from existing extraction)
        ff_path = Path(config.output_dir) / "uncertainties" / arch / "faceforensics_seed42.npz"
        mc_corr = 0.0
        cof_corr = 0.0
        if ff_path.exists():
            d = np.load(ff_path)
            U = d["uncertainties"]
            errors = d["errors"]
            mc_probs = d.get("mc_probs", None)
            if mc_probs is not None:
                mean_p = np.mean(mc_probs, axis=0)
                mc_ent = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=1)
                mc_corr = safe_corr(mc_ent, errors)[0]
            cof = CorrelationOptimizedFusion(k_sources=5, n_restarts=20)
            cof.fit(U, errors)
            cof_corr = cof.result_.correlation

        # Deep Ensemble
        de_result = eval_deep_ensemble(arch, config)
        de_corr = de_result["correlation_entropy"] if de_result else 0.0
        all_results["deep_ensemble"][arch] = de_result

        # Evidential
        ev_result = eval_evidential(arch, config)
        ev_corr = ev_result["correlation"] if ev_result and not ev_result.get("constant_output") else float("nan")
        all_results["evidential"][arch] = ev_result

        ev_str = "nan" if np.isnan(ev_corr) else "{:.4f}".format(ev_corr)
        de_str = "{:.4f}".format(de_corr) if de_result else "---"

        print("{:15s} {:>10.4f} {:>10s} {:>10s} {:>10.4f}".format(
            name, mc_corr, de_str, ev_str, cof_corr))

    # Save results
    save_dir = Path(config.output_dir) / "baselines"
    save_dir.mkdir(parents=True, exist_ok=True)

    def conv(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return "nan"
        return obj

    with open(save_dir / "baseline_results_{}.json".format(args.arch) if args.arch else "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=conv)

    print("\nSaved: {}".format(save_dir / "baseline_results_{}.json".format(args.arch) if args.arch else "baseline_results.json"))


if __name__ == "__main__":
    main()
