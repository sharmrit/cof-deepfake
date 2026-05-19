"""
Registry for running all 13 fusion methods systematically.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr

from .cof import CorrelationOptimizedFusion
from .baselines import (
    uniform_average,
    best_single_source,
    pca_fusion,
    entropy_weighted_fusion,
    rank_fusion,
    mc_dropout_ensemble,
    deep_ensemble,
)
from ..config import UNCERTAINTY_SOURCES, FUSION_METHODS


class FusionRegistry:
    """
    Central registry for all 13 fusion methods.

    Provides a unified interface to fit/evaluate every method
    on the same data for fair comparison.
    """

    def __init__(
        self,
        source_names: Optional[List[str]] = None,
        n_restarts: int = 10,
        max_iter: int = 1000,
    ):
        self.source_names = source_names or list(UNCERTAINTY_SOURCES)
        self.n_restarts = n_restarts
        self.max_iter = max_iter

    def _select_top_k(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Select top-k sources by individual correlation."""
        corrs = []
        for i in range(uncertainties.shape[1]):
            if np.std(uncertainties[:, i]) > 1e-12:
                c, _ = pearsonr(uncertainties[:, i], errors)
            else:
                c = 0.0
            corrs.append(c)
        ranked_idx = np.argsort(corrs)[::-1][:k]
        return uncertainties[:, sorted(ranked_idx)]

    def run_all(
        self,
        train_uncertainties: np.ndarray,
        train_errors: np.ndarray,
        test_uncertainties: np.ndarray,
        test_errors: np.ndarray,
        mc_probs: Optional[np.ndarray] = None,
        ensemble_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict]:
        """
        Run all 13 fusion methods.

        Parameters
        ----------
        train_uncertainties : ndarray (N_train, 5)
        train_errors : ndarray (N_train,)
        test_uncertainties : ndarray (N_test, 5)
        test_errors : ndarray (N_test,)
        mc_probs : ndarray (T, N_test, C), optional
        ensemble_probs : ndarray (M, N_test, C), optional

        Returns
        -------
        results : dict mapping method_name -> result_dict
        """
        results = {}

        # --- COF variants ---
        for k, label in [(5, "cof_k5"), (4, "cof_k4"), (3, "cof_k3"), (2, "cof_k2")]:
            if k > train_uncertainties.shape[1]:
                continue
            U_train = self._select_top_k(train_uncertainties, train_errors, k)
            U_test = self._select_top_k(test_uncertainties, test_errors, k)
            cof = CorrelationOptimizedFusion(
                k_sources=k,
                constraint="simplex",
                n_restarts=self.n_restarts,
                max_iter=self.max_iter,
            )
            cof.fit(U_train, train_errors)
            fused_test = cof.predict(U_test)
            corr, pval = pearsonr(fused_test, test_errors) if np.std(fused_test) > 1e-12 else (0.0, 1.0)
            results[label] = {
                "name": label,
                "fused": fused_test,
                "weights": cof.weights_,
                "correlation": corr,
                "p_value": pval,
                "train_correlation": cof.result_.correlation,
                "optimization_time": cof.result_.optimization_time,
                "weight_analysis": cof.get_weight_analysis(),
            }

        # --- COF constrained (non-negative, no simplex) ---
        for k, label in [(5, "cof_k5_constrained"), (4, "cof_k4_constrained")]:
            if k > train_uncertainties.shape[1]:
                continue
            U_train = self._select_top_k(train_uncertainties, train_errors, k)
            U_test = self._select_top_k(test_uncertainties, test_errors, k)
            cof = CorrelationOptimizedFusion(
                k_sources=k,
                constraint="non_negative",
                n_restarts=self.n_restarts,
                max_iter=self.max_iter,
            )
            cof.fit(U_train, train_errors)
            fused_test = cof.predict(U_test)
            corr, pval = pearsonr(fused_test, test_errors) if np.std(fused_test) > 1e-12 else (0.0, 1.0)
            results[label] = {
                "name": label,
                "fused": fused_test,
                "weights": cof.weights_,
                "correlation": corr,
                "p_value": pval,
                "train_correlation": cof.result_.correlation,
            }

        # --- Baselines (no train/test split needed) ---
        baseline_methods = {
            "uniform_average": uniform_average,
            "best_single": best_single_source,
            "pca_fusion": pca_fusion,
            "entropy_weighted": entropy_weighted_fusion,
            "rank_fusion": rank_fusion,
        }
        for name, func in baseline_methods.items():
            res = func(
                test_uncertainties,
                test_errors,
                source_names=self.source_names,
            )
            results[name] = res

        # --- MC Dropout Ensemble ---
        results["mc_dropout_ensemble"] = mc_dropout_ensemble(
            test_uncertainties, test_errors, mc_probs=mc_probs
        )

        # --- Deep Ensemble ---
        results["deep_ensemble"] = deep_ensemble(
            test_uncertainties, test_errors, ensemble_probs=ensemble_probs
        )

        return results


def run_all_methods(
    train_uncertainties: np.ndarray,
    train_errors: np.ndarray,
    test_uncertainties: np.ndarray,
    test_errors: np.ndarray,
    **kwargs,
) -> Dict[str, Dict]:
    """Convenience function wrapping FusionRegistry."""
    registry = FusionRegistry()
    return registry.run_all(
        train_uncertainties, train_errors,
        test_uncertainties, test_errors,
        **kwargs,
    )
