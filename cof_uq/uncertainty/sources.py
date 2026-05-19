"""
Five uncertainty source computations for deepfake detection.

Sources
-------
1. Epistemic  : MC Dropout variance (model uncertainty)
2. Aleatoric  : Predictive entropy / variance (data uncertainty)
3. Calibration: 1 − max(softmax) (confidence-based)
4. Conformal  : Nonconformity score (distribution-free)
5. Distributional: Mahalanobis distance (feature-space OOD)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


# =============================================================================
# 1. Epistemic Uncertainty — MC Dropout Variance
# =============================================================================

def compute_epistemic(
    mc_probs: np.ndarray,
) -> np.ndarray:
    """
    Epistemic uncertainty via MC Dropout variance.

    Parameters
    ----------
    mc_probs : ndarray of shape (T, N, C)
        Softmax probabilities from T MC forward passes.

    Returns
    -------
    epistemic : ndarray of shape (N,)
        Variance of the positive-class probability across MC passes.
    """
    # Variance of P(fake) across T stochastic passes
    return np.var(mc_probs[:, :, 1], axis=0)


# =============================================================================
# 2. Aleatoric Uncertainty — Predictive Variance
# =============================================================================

def compute_aleatoric(
    mean_probs: np.ndarray,
) -> np.ndarray:
    """
    Aleatoric uncertainty via Bernoulli variance.

    Parameters
    ----------
    mean_probs : ndarray of shape (N, C)
        Mean softmax probabilities (averaged across MC passes).

    Returns
    -------
    aleatoric : ndarray of shape (N,)
        p*(1-p) for the positive-class probability.
    """
    p = mean_probs[:, 1]
    return p * (1.0 - p)


# =============================================================================
# 3. Calibration Uncertainty — Confidence Gap
# =============================================================================

def compute_calibration(
    mean_probs: np.ndarray,
) -> np.ndarray:
    """
    Calibration uncertainty: 1 − max(softmax).

    Higher values indicate the model is less confident in its prediction.

    Parameters
    ----------
    mean_probs : ndarray of shape (N, C)

    Returns
    -------
    calibration : ndarray of shape (N,)
    """
    return 1.0 - np.max(mean_probs, axis=1)


# =============================================================================
# 4. Conformal Uncertainty — Nonconformity Score
# =============================================================================

def compute_conformal(
    mc_probs: np.ndarray,
    cal_scores: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Conformal prediction-based uncertainty.

    Uses the standard deviation across MC passes as a nonconformity
    measure. If calibration scores from a held-out set are provided,
    computes the conformal p-value for each sample.

    Parameters
    ----------
    mc_probs : ndarray of shape (T, N, C)
    cal_scores : ndarray of shape (M,), optional
        Calibration nonconformity scores from a held-out set.
    alpha : float
        Significance level for conformal sets.

    Returns
    -------
    conformal : ndarray of shape (N,)
    """
    # Nonconformity score = std across MC forward passes
    nonconformity = np.std(mc_probs[:, :, 1], axis=0)

    if cal_scores is not None and len(cal_scores) > 0:
        # Conformal p-value: fraction of calibration scores >= this score
        conformal = np.array([
            np.mean(cal_scores >= s) for s in nonconformity
        ])
        # Invert: smaller p-value = more uncertain
        conformal = 1.0 - conformal
    else:
        conformal = nonconformity

    return conformal


def compute_conformal_calibration_scores(
    mc_probs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute nonconformity scores on calibration set for conformal prediction.

    Parameters
    ----------
    mc_probs : ndarray of shape (T, N, C)
    labels : ndarray of shape (N,)

    Returns
    -------
    cal_scores : ndarray of shape (N,)
    """
    mean_probs = np.mean(mc_probs, axis=0)
    # Nonconformity = 1 - P(true class)
    cal_scores = np.array([
        1.0 - mean_probs[i, labels[i]] for i in range(len(labels))
    ])
    return cal_scores


# =============================================================================
# 5. Distributional Uncertainty — Mahalanobis Distance
# =============================================================================

def compute_distributional(
    features: np.ndarray,
    train_features: Optional[np.ndarray] = None,
    train_mean: Optional[np.ndarray] = None,
    train_cov_inv: Optional[np.ndarray] = None,
    use_ledoit_wolf: bool = True,
) -> np.ndarray:
    """
    Distributional uncertainty via Mahalanobis distance.

    Measures how far each sample's feature representation is from the
    training distribution centroid in the learned feature space.

    Parameters
    ----------
    features : ndarray of shape (N, D)
        Penultimate-layer features for test samples.
    train_features : ndarray of shape (M, D), optional
        Training set features (used to fit covariance if not precomputed).
    train_mean : ndarray of shape (D,), optional
        Precomputed training feature centroid.
    train_cov_inv : ndarray of shape (D, D), optional
        Precomputed inverse covariance matrix.
    use_ledoit_wolf : bool
        Use Ledoit-Wolf shrinkage for covariance estimation.

    Returns
    -------
    distributional : ndarray of shape (N,)
        Mahalanobis distances.
    """
    if train_mean is None or train_cov_inv is None:
        if train_features is None:
            raise ValueError(
                "Either provide (train_mean, train_cov_inv) or train_features."
            )
        train_mean = np.mean(train_features, axis=0)
        try:
            if use_ledoit_wolf:
                cov_est = LedoitWolf().fit(train_features)
            else:
                cov_est = EmpiricalCovariance().fit(train_features)
            train_cov_inv = np.linalg.pinv(cov_est.covariance_)
        except Exception:
            # Fallback: Euclidean distance
            diff = features - train_mean
            return np.linalg.norm(diff, axis=1)

    # Vectorized Mahalanobis: sqrt( (x-mu)^T Σ^-1 (x-mu) )
    diff = features - train_mean
    left = diff @ train_cov_inv
    mahal_sq = np.sum(left * diff, axis=1)
    mahal_sq = np.maximum(mahal_sq, 0.0)  # numerical safety
    return np.sqrt(mahal_sq)


def fit_distributional_params(
    train_features: np.ndarray,
    use_ledoit_wolf: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit training distribution parameters for Mahalanobis distance.

    Returns
    -------
    train_mean : ndarray of shape (D,)
    train_cov_inv : ndarray of shape (D, D)
    """
    train_mean = np.mean(train_features, axis=0)
    if use_ledoit_wolf:
        cov_est = LedoitWolf().fit(train_features)
    else:
        cov_est = EmpiricalCovariance().fit(train_features)
    train_cov_inv = np.linalg.pinv(cov_est.covariance_)
    return train_mean, train_cov_inv
