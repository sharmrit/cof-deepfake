"""
Evaluation metrics for uncertainty quantification in deepfake detection.

Primary metric: Pearson correlation ρ(U_fused, errors)
Secondary metrics: ECE, Brier score, AUC, accuracy
Statistical: bootstrap confidence intervals, multi-seed aggregation
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Dict, Tuple, Optional


# =============================================================================
# Core Metrics
# =============================================================================

def compute_correlation(
    fused_uncertainty: np.ndarray,
    errors: np.ndarray,
    method: str = "pearson",
) -> Tuple[float, float]:
    """
    Compute correlation between fused uncertainty and errors.

    Parameters
    ----------
    fused_uncertainty : ndarray (N,)
    errors : ndarray (N,)
    method : 'pearson' or 'spearman'

    Returns
    -------
    correlation : float
    p_value : float
    """
    if np.std(fused_uncertainty) < 1e-12 or np.std(errors) < 1e-12:
        return 0.0, 1.0
    if method == "pearson":
        return pearsonr(fused_uncertainty, errors)
    elif method == "spearman":
        return spearmanr(fused_uncertainty, errors)
    raise ValueError(f"Unknown method: {method}")


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    probs : ndarray (N, C) — softmax probabilities
    labels : ndarray (N,) — ground truth
    n_bins : int

    Returns
    -------
    ece : float
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_weight = mask.sum() / n_total
        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Brier score for binary classification."""
    return brier_score_loss(labels, probs[:, 1])


def compute_auc(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """ROC-AUC score."""
    try:
        return roc_auc_score(labels, probs[:, 1])
    except ValueError:
        return 0.5


def compute_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Classification accuracy."""
    preds = np.argmax(probs, axis=1)
    return accuracy_score(labels, preds)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_confidence_interval(
    fused_uncertainty: np.ndarray,
    errors: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    metric_fn=None,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence interval for correlation.

    Returns
    -------
    dict with 'mean', 'std', 'ci_low', 'ci_high', 'samples'
    """
    rng = np.random.RandomState(seed)
    n = len(errors)

    if metric_fn is None:
        def metric_fn(u, e):
            return compute_correlation(u, e)[0]

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        val = metric_fn(fused_uncertainty[idx], errors[idx])
        boot_values.append(val)

    boot_values = np.array(boot_values)
    alpha = 1.0 - confidence_level
    ci_low = np.percentile(boot_values, 100 * alpha / 2)
    ci_high = np.percentile(boot_values, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(boot_values)),
        "std": float(np.std(boot_values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "samples": boot_values,
    }


# =============================================================================
# Full Evaluation
# =============================================================================

def full_evaluation(
    fused_uncertainty: np.ndarray,
    errors: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    n_bins: int = 15,
) -> Dict:
    """
    Comprehensive evaluation of a fusion method.

    Returns dict with all metrics + confidence intervals.
    """
    corr_p, pval = compute_correlation(fused_uncertainty, errors)
    corr_s, _ = compute_correlation(fused_uncertainty, errors, method="spearman")
    ece = compute_ece(probs, labels, n_bins=n_bins)
    brier = compute_brier_score(probs, labels)
    auc = compute_auc(probs, labels)
    acc = compute_accuracy(probs, labels)

    preds = np.argmax(probs, axis=1)
    f1 = f1_score(labels, preds, average="binary", zero_division=0.0)
    prec = precision_score(labels, preds, average="binary", zero_division=0.0)
    rec = recall_score(labels, preds, average="binary", zero_division=0.0)

    boot = bootstrap_confidence_interval(
        fused_uncertainty, errors, n_bootstrap=n_bootstrap
    )

    return {
        "pearson_correlation": corr_p,
        "spearman_correlation": corr_s,
        "p_value": pval,
        "ece": ece,
        "brier_score": brier,
        "auc": auc,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "bootstrap_mean": boot["mean"],
        "bootstrap_std": boot["std"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
    }


def aggregate_seeds(
    seed_results: Dict[int, Dict],
) -> Dict:
    """
    Aggregate metrics across multiple seeds.

    Parameters
    ----------
    seed_results : dict mapping seed -> metrics dict

    Returns
    -------
    dict with mean, std, cv for each metric
    """
    metric_names = list(next(iter(seed_results.values())).keys())
    agg = {}
    for metric in metric_names:
        if metric in ("bootstrap_samples",):
            continue
        values = [
            seed_results[s][metric]
            for s in seed_results
            if isinstance(seed_results[s].get(metric), (int, float))
        ]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            agg[metric] = {
                "mean": float(mean),
                "std": float(std),
                "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else 0.0,
                "values": values,
            }
    return agg
