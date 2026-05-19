"""
Stability analysis across seeds and perturbations.

Evaluates:
  - Multi-seed weight stability (CV < 0.5% from paper)
  - Perturbation robustness
  - Per-source stability (CV analysis from Figure 5)
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Optional

from ..fusion.cof import CorrelationOptimizedFusion
from ..config import UNCERTAINTY_SOURCES, SEEDS


class StabilityAnalyzer:
    """
    Multi-seed and perturbation stability analysis.

    Parameters
    ----------
    uncertainties : ndarray (N, K)
    errors : ndarray (N,)
    source_names : list of str
    """

    def __init__(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
    ):
        self.U = uncertainties
        self.errors = errors
        self.K = uncertainties.shape[1]
        self.source_names = source_names or list(UNCERTAINTY_SOURCES[: self.K])

    def multi_seed_stability(
        self,
        seeds: Optional[List[int]] = None,
        n_restarts: int = 10,
    ) -> Dict:
        """
        Run COF with different random seeds and analyze weight stability.
        """
        seeds = seeds or list(SEEDS)
        all_weights = []
        all_correlations = []

        for seed in seeds:
            np.random.seed(seed)
            cof = CorrelationOptimizedFusion(
                k_sources=self.K, n_restarts=n_restarts
            )
            cof.fit(self.U, self.errors, source_names=self.source_names)
            all_weights.append(cof.weights_)
            all_correlations.append(cof.result_.correlation)

        weights_array = np.array(all_weights)  # (n_seeds, K)
        corr_array = np.array(all_correlations)

        # Per-source weight statistics
        per_source = {}
        for i, name in enumerate(self.source_names):
            w_vals = weights_array[:, i]
            per_source[name] = {
                "mean": float(np.mean(w_vals)),
                "std": float(np.std(w_vals)),
                "cv": float(np.std(w_vals) / abs(np.mean(w_vals)))
                if abs(np.mean(w_vals)) > 1e-12
                else 0.0,
                "min": float(np.min(w_vals)),
                "max": float(np.max(w_vals)),
                "values": w_vals.tolist(),
            }

        return {
            "seeds": seeds,
            "correlation": {
                "mean": float(np.mean(corr_array)),
                "std": float(np.std(corr_array)),
                "cv": float(np.std(corr_array) / abs(np.mean(corr_array)))
                if abs(np.mean(corr_array)) > 1e-12
                else 0.0,
                "values": corr_array.tolist(),
            },
            "per_source_weights": per_source,
            "overall_weight_cv": float(
                np.mean(weights_array.std(axis=0) / np.maximum(weights_array.mean(axis=0), 1e-12))
            ),
        }

    def perturbation_robustness(
        self,
        noise_levels: Optional[List[float]] = None,
        n_trials: int = 50,
        seed: int = 42,
    ) -> Dict:
        """
        Evaluate robustness to input perturbations.

        Adds Gaussian noise to uncertainties and measures correlation
        degradation.
        """
        noise_levels = noise_levels or [0.01, 0.05, 0.10, 0.20, 0.50]
        rng = np.random.RandomState(seed)

        # Baseline (no noise)
        cof = CorrelationOptimizedFusion(k_sources=self.K, n_restarts=10)
        cof.fit(self.U, self.errors, source_names=self.source_names)
        baseline_corr = cof.result_.correlation
        baseline_weights = cof.weights_

        results = {"baseline_correlation": baseline_corr}

        for noise in noise_levels:
            trial_corrs = []
            for _ in range(n_trials):
                U_noisy = self.U + rng.normal(0, noise, self.U.shape)
                U_noisy = np.clip(U_noisy, 0, 1)
                fused = U_noisy @ baseline_weights
                if np.std(fused) > 1e-12:
                    corr, _ = pearsonr(fused, self.errors)
                else:
                    corr = 0.0
                trial_corrs.append(corr)

            trial_corrs = np.array(trial_corrs)
            results[f"noise_{noise}"] = {
                "mean_correlation": float(np.mean(trial_corrs)),
                "std_correlation": float(np.std(trial_corrs)),
                "degradation_pct": float(
                    (baseline_corr - np.mean(trial_corrs))
                    / abs(baseline_corr)
                    * 100
                )
                if abs(baseline_corr) > 1e-12
                else 0.0,
            }

        return results

    def per_source_stability(
        self,
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> Dict:
        """
        Evaluate stability of individual source correlations.
        Reproduces Figure 5 analysis from CVPR paper.
        """
        rng = np.random.RandomState(seed)
        n = len(self.errors)

        source_boot = {name: [] for name in self.source_names}

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            for i, name in enumerate(self.source_names):
                src = self.U[idx, i]
                err = self.errors[idx]
                if np.std(src) > 1e-12 and np.std(err) > 1e-12:
                    corr, _ = pearsonr(src, err)
                else:
                    corr = 0.0
                source_boot[name].append(corr)

        results = {}
        for name in self.source_names:
            vals = np.array(source_boot[name])
            results[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "cv": float(np.std(vals) / abs(np.mean(vals)))
                if abs(np.mean(vals)) > 1e-12
                else 0.0,
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            }

        # Rank by stability (lowest CV = most stable)
        ranked = sorted(results.items(), key=lambda x: x[1]["cv"])
        results["_ranking"] = [name for name, _ in ranked]

        return results
