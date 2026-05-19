"""
Baseline fusion methods for comparison against COF.

Implements 7 baseline strategies:
  1. Uniform averaging
  2. Best single source
  3. PCA-based fusion
  4. Entropy-weighted fusion
  5. Rank-based fusion
  6. MC Dropout ensemble
  7. Deep Ensemble
"""

import numpy as np
from scipy.stats import pearsonr, rankdata
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional, List


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Pearson correlation with constant-array guard."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    return pearsonr(x, y)


# =============================================================================
# 1. Uniform Average
# =============================================================================

def uniform_average(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    **kwargs,
) -> Dict:
    """
    Equal-weight averaging of all K sources.
    U_fused = (1/K) * Σ ũᵢ
    """
    k = uncertainties.shape[1]
    weights = np.ones(k) / k
    fused = uncertainties @ weights
    corr, pval = _safe_pearsonr(fused, errors)
    return {
        "name": "uniform_average",
        "fused": fused,
        "weights": weights,
        "correlation": corr,
        "p_value": pval,
    }


# =============================================================================
# 2. Best Single Source
# =============================================================================

def best_single_source(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    source_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict:
    """
    Select the single source with highest correlation to errors.
    Paper finding: conformal consistently dominates (7/8 architectures).
    """
    k = uncertainties.shape[1]
    if source_names is None:
        source_names = [f"source_{i}" for i in range(k)]

    best_corr = -np.inf
    best_idx = 0
    individual = {}

    for i in range(k):
        corr, pval = _safe_pearsonr(uncertainties[:, i], errors)
        individual[source_names[i]] = corr
        if corr > best_corr:
            best_corr = corr
            best_idx = i

    weights = np.zeros(k)
    weights[best_idx] = 1.0

    return {
        "name": "best_single",
        "fused": uncertainties[:, best_idx],
        "weights": weights,
        "correlation": best_corr,
        "p_value": _safe_pearsonr(uncertainties[:, best_idx], errors)[1],
        "best_source": source_names[best_idx],
        "individual_correlations": individual,
    }


# =============================================================================
# 3. PCA Fusion
# =============================================================================

def pca_fusion(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_components: int = 1,
    **kwargs,
) -> Dict:
    """
    PCA-based fusion: project K sources onto first principal component.
    """
    pca = PCA(n_components=n_components)
    fused = pca.fit_transform(uncertainties).ravel()
    # Flip sign if negatively correlated with errors
    corr, pval = _safe_pearsonr(fused, errors)
    if corr < 0:
        fused = -fused
        corr = -corr

    return {
        "name": "pca_fusion",
        "fused": fused,
        "weights": pca.components_[0],
        "correlation": corr,
        "p_value": pval,
        "explained_variance_ratio": pca.explained_variance_ratio_[0],
    }


# =============================================================================
# 4. Entropy-Weighted Fusion
# =============================================================================

def entropy_weighted_fusion(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    **kwargs,
) -> Dict:
    """
    Weight each source by inverse normalized entropy.
    Sources with more peaked distributions get higher weight.
    """
    k = uncertainties.shape[1]
    weights = np.zeros(k)

    for i in range(k):
        vals = uncertainties[:, i]
        # Discretize and compute entropy
        hist, _ = np.histogram(vals, bins=50, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-12))
        max_entropy = np.log(50)
        # Inverse normalized entropy
        weights[i] = 1.0 - (entropy / max_entropy)

    # Normalize to sum to 1
    weights = np.maximum(weights, 0.0)
    if weights.sum() > 1e-12:
        weights /= weights.sum()
    else:
        weights = np.ones(k) / k

    fused = uncertainties @ weights
    corr, pval = _safe_pearsonr(fused, errors)

    return {
        "name": "entropy_weighted",
        "fused": fused,
        "weights": weights,
        "correlation": corr,
        "p_value": pval,
    }


# =============================================================================
# 5. Rank-Based Fusion
# =============================================================================

def rank_fusion(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    **kwargs,
) -> Dict:
    """
    Rank-based fusion: average rank across sources.
    Robust to outliers and non-linear relationships.
    """
    k = uncertainties.shape[1]
    n = uncertainties.shape[0]

    ranked = np.zeros_like(uncertainties)
    for i in range(k):
        ranked[:, i] = rankdata(uncertainties[:, i]) / n

    fused = ranked.mean(axis=1)
    corr, pval = _safe_pearsonr(fused, errors)

    return {
        "name": "rank_fusion",
        "fused": fused,
        "weights": np.ones(k) / k,
        "correlation": corr,
        "p_value": pval,
    }


# =============================================================================
# 6. MC Dropout Ensemble (baseline using epistemic only)
# =============================================================================

def mc_dropout_ensemble(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    mc_probs: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    """
    MC Dropout ensemble baseline.
    Uses predictive entropy from MC forward passes as uncertainty.
    Falls back to epistemic column if mc_probs not available.
    """
    if mc_probs is not None:
        mean_probs = np.mean(mc_probs, axis=0)
        # Predictive entropy
        fused = -np.sum(
            mean_probs * np.log(mean_probs + 1e-12), axis=1
        )
    else:
        # Fallback: use epistemic column (index 0)
        fused = uncertainties[:, 0]

    corr, pval = _safe_pearsonr(fused, errors)

    return {
        "name": "mc_dropout_ensemble",
        "fused": fused,
        "weights": None,
        "correlation": corr,
        "p_value": pval,
    }


# =============================================================================
# 7. Deep Ensemble (stub — requires multiple trained models)
# =============================================================================

def deep_ensemble(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    ensemble_probs: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict:
    """
    Deep Ensemble baseline.

    If ensemble_probs (M, N, C) are provided (from M independently
    trained models), computes predictive entropy.
    Otherwise falls back to uniform average of available sources.
    """
    if ensemble_probs is not None and ensemble_probs.ndim == 3:
        mean_probs = np.mean(ensemble_probs, axis=0)
        fused = -np.sum(
            mean_probs * np.log(mean_probs + 1e-12), axis=1
        )
    else:
        # Fallback: uniform average
        fused = uncertainties.mean(axis=1)

    corr, pval = _safe_pearsonr(fused, errors)

    return {
        "name": "deep_ensemble",
        "fused": fused,
        "weights": None,
        "correlation": corr,
        "p_value": pval,
    }
