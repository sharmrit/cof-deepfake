"""
Correlation-Optimized Fusion (COF) — Core Algorithm.

Directly maximizes Pearson correlation between fused uncertainty
and prediction errors via constrained optimization (Eq. 4-7 in paper).
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict


@dataclass
class COFResult:
    """Result from COF optimization."""
    weights: np.ndarray
    correlation: float
    p_value: float
    optimization_time: float
    n_sources: int
    source_names: List[str]
    converged: bool
    n_restarts_used: int


class CorrelationOptimizedFusion:
    """
    Correlation-Optimized Fusion (COF).

    Maximizes ρ(U_fused, errors) = ρ(Σ wᵢ·ũᵢ, e) subject to
    constraints on w (simplex or non-negative).

    Parameters
    ----------
    k_sources : int
        Number of uncertainty sources to fuse (2-5).
    constraint : str
        'simplex' : weights sum to 1, all >= 0  (default)
        'non_negative' : weights >= 0, no sum constraint
    optimizer : str
        Scipy optimizer method (default 'SLSQP').
    n_restarts : int
        Number of random restarts for multi-start optimization.
    max_iter : int
        Maximum iterations per restart.
    tol : float
        Convergence tolerance.
    regularization : float
        L2 regularization penalty on weights (encourages diversity).
    """

    def __init__(
        self,
        k_sources: int = 5,
        constraint: str = "simplex",
        optimizer: str = "SLSQP",
        n_restarts: int = 10,
        max_iter: int = 1000,
        tol: float = 1e-10,
        regularization: float = 0.0,
    ):
        self.k = k_sources
        self.constraint = constraint
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

        self.weights_: Optional[np.ndarray] = None
        self.result_: Optional[COFResult] = None

    def _objective(
        self, w: np.ndarray, U: np.ndarray, errors: np.ndarray
    ) -> float:
        """
        Negative Pearson correlation + optional L2 regularization.
        (Eq. 5 in paper: minimize -ρ to maximize ρ)
        """
        fused = U @ w
        # Guard against constant arrays
        if np.std(fused) < 1e-12 or np.std(errors) < 1e-12:
            return 1.0
        corr, _ = pearsonr(fused, errors)
        penalty = self.regularization * np.sum(w ** 2)
        return -corr + penalty

    def _get_constraints(self):
        """Build scipy constraint dicts."""
        if self.constraint == "simplex":
            return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        return []

    def _get_bounds(self):
        """Bounds for each weight: [0, 1] always."""
        return [(0.0, 1.0)] * self.k

    def fit(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
    ) -> "CorrelationOptimizedFusion":
        """
        Optimize fusion weights on training/validation data.

        Parameters
        ----------
        uncertainties : ndarray of shape (N, K)
            Normalized uncertainty matrix (K sources).
        errors : ndarray of shape (N,)
            Binary prediction errors.
        source_names : list of str, optional
            Names of the K sources.

        Returns
        -------
        self
        """
        assert uncertainties.shape[1] == self.k, (
            f"Expected {self.k} sources, got {uncertainties.shape[1]}"
        )
        if source_names is None:
            source_names = [f"source_{i}" for i in range(self.k)]

        start_time = time.time()
        constraints = self._get_constraints()
        bounds = self._get_bounds()

        best_result = None
        best_corr = -np.inf

        for restart in range(self.n_restarts):
            # Random starting point on simplex
            if restart == 0:
                w0 = np.ones(self.k) / self.k  # uniform start
            else:
                w0 = np.random.dirichlet(np.ones(self.k))

            try:
                res = minimize(
                    self._objective,
                    w0,
                    args=(uncertainties, errors),
                    method=self.optimizer,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self.max_iter, "ftol": self.tol},
                )
                if res.success or res.fun < -best_corr:
                    fused = uncertainties @ res.x
                    if np.std(fused) > 1e-12:
                        corr, pval = pearsonr(fused, errors)
                        if corr > best_corr:
                            best_corr = corr
                            best_result = (res, corr, pval, restart + 1)
            except Exception:
                continue

        opt_time = time.time() - start_time

        if best_result is not None:
            res, corr, pval, n_used = best_result
            self.weights_ = res.x
            self.result_ = COFResult(
                weights=res.x,
                correlation=corr,
                p_value=pval,
                optimization_time=opt_time,
                n_sources=self.k,
                source_names=source_names,
                converged=res.success,
                n_restarts_used=n_used,
            )
        else:
            # Fallback to uniform
            self.weights_ = np.ones(self.k) / self.k
            fused = uncertainties @ self.weights_
            corr, pval = pearsonr(fused, errors)
            self.result_ = COFResult(
                weights=self.weights_,
                correlation=corr,
                p_value=pval,
                optimization_time=opt_time,
                n_sources=self.k,
                source_names=source_names,
                converged=False,
                n_restarts_used=self.n_restarts,
            )

        return self

    def predict(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Compute fused uncertainty for new data.

        Parameters
        ----------
        uncertainties : ndarray of shape (N, K)

        Returns
        -------
        fused : ndarray of shape (N,)
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        return uncertainties @ self.weights_

    def fit_predict(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit on data and return fused uncertainty."""
        self.fit(uncertainties, errors, source_names)
        return self.predict(uncertainties)

    def get_weight_analysis(self) -> Dict:
        """Analyze learned weights: sparsity, entropy, dominance."""
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")
        w = self.weights_
        names = self.result_.source_names

        # Sparsity: fraction of near-zero weights
        sparsity = np.mean(w < 1e-4)

        # Normalized entropy
        w_pos = w[w > 1e-10]
        if len(w_pos) > 1:
            entropy = -np.sum(w_pos * np.log(w_pos)) / np.log(len(w_pos))
        else:
            entropy = 0.0

        # Dominant source
        dominant_idx = np.argmax(w)

        return {
            "weights": dict(zip(names, w.tolist())),
            "sparsity": float(sparsity),
            "entropy": float(entropy),
            "dominant_source": names[dominant_idx],
            "dominant_weight": float(w[dominant_idx]),
        }


def cof_with_source_selection(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    source_names: List[str],
    k_values: List[int] = [2, 3, 4, 5],
    **kwargs,
) -> Dict[int, COFResult]:
    """
    Run COF for multiple K values (source selection ablation).

    For each K, selects the top-K sources by individual correlation
    and optimizes fusion weights.

    Returns
    -------
    results : dict mapping K -> COFResult
    """
    # Rank sources by individual correlation
    individual_corrs = {}
    for i, name in enumerate(source_names):
        src = uncertainties[:, i]
        if np.std(src) > 1e-12:
            corr, _ = pearsonr(src, errors)
            individual_corrs[i] = corr
        else:
            individual_corrs[i] = 0.0

    ranked = sorted(individual_corrs.items(), key=lambda x: -x[1])

    results = {}
    for k in k_values:
        if k > len(source_names):
            continue
        top_k_indices = [idx for idx, _ in ranked[:k]]
        top_k_names = [source_names[i] for i in top_k_indices]
        U_k = uncertainties[:, top_k_indices]

        cof = CorrelationOptimizedFusion(k_sources=k, **kwargs)
        cof.fit(U_k, errors, source_names=top_k_names)
        results[k] = cof.result_

    return results
