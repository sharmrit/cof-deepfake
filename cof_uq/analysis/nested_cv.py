"""
Nested K-Fold Cross-Validation for unbiased COF evaluation.

Addresses validation bias: COF weights optimized on val set should NOT
be evaluated on the same val set. Nested CV provides unbiased estimates.

Outer loop: evaluation (5-fold)
Inner loop: hyperparameter selection / weight optimization (3-fold)
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple
import time

from ..fusion.cof import CorrelationOptimizedFusion
from ..config import UNCERTAINTY_SOURCES


class NestedCrossValidator:
    """
    Nested K-Fold CV for rigorous COF evaluation.

    Parameters
    ----------
    outer_folds : int
        Number of outer evaluation folds.
    inner_folds : int
        Number of inner optimization folds.
    stratified : bool
        Use stratified splits based on error labels.
    n_restarts : int
        Number of restarts for COF optimization.
    """

    def __init__(
        self,
        outer_folds: int = 5,
        inner_folds: int = 3,
        stratified: bool = True,
        n_restarts: int = 10,
    ):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.stratified = stratified
        self.n_restarts = n_restarts

    def evaluate(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
        seed: int = 42,
    ) -> Dict:
        """
        Run nested cross-validation.

        Parameters
        ----------
        uncertainties : ndarray (N, K)
        errors : ndarray (N,)
        source_names : list of str
        k_values : list of int — K values to compare in inner loop

        Returns
        -------
        results : dict with per-fold and aggregate metrics
        """
        K_max = uncertainties.shape[1]
        source_names = source_names or list(UNCERTAINTY_SOURCES[:K_max])
        k_values = k_values or list(range(2, K_max + 1))

        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds,
            shuffle=True,
            random_state=seed,
        )

        fold_results = []
        all_test_correlations = []
        start_time = time.time()

        # Binarize errors for stratification
        error_labels = (errors > 0).astype(int)

        for fold_idx, (train_val_idx, test_idx) in enumerate(
            outer_cv.split(uncertainties, error_labels)
        ):
            U_train_val = uncertainties[train_val_idx]
            e_train_val = errors[train_val_idx]
            U_test = uncertainties[test_idx]
            e_test = errors[test_idx]

            # --- Inner loop: select best K ---
            inner_cv = StratifiedKFold(
                n_splits=self.inner_folds,
                shuffle=True,
                random_state=seed + fold_idx,
            )
            e_train_val_labels = (e_train_val > 0).astype(int)

            best_inner_corr = -np.inf
            best_k = k_values[0]

            k_inner_scores = {k: [] for k in k_values}

            for inner_train_idx, inner_val_idx in inner_cv.split(
                U_train_val, e_train_val_labels
            ):
                U_inner_train = U_train_val[inner_train_idx]
                e_inner_train = e_train_val[inner_train_idx]
                U_inner_val = U_train_val[inner_val_idx]
                e_inner_val = e_train_val[inner_val_idx]

                for k in k_values:
                    if k > K_max:
                        continue
                    cof = CorrelationOptimizedFusion(
                        k_sources=k, n_restarts=self.n_restarts
                    )
                    cof.fit(U_inner_train[:, :k], e_inner_train)
                    fused_val = cof.predict(U_inner_val[:, :k])
                    if np.std(fused_val) > 1e-12:
                        val_corr, _ = pearsonr(fused_val, e_inner_val)
                    else:
                        val_corr = 0.0
                    k_inner_scores[k].append(val_corr)

            # Select best K by inner loop performance
            for k in k_values:
                if k_inner_scores[k]:
                    mean_inner = np.mean(k_inner_scores[k])
                    if mean_inner > best_inner_corr:
                        best_inner_corr = mean_inner
                        best_k = k

            # --- Outer evaluation with selected K ---
            cof_final = CorrelationOptimizedFusion(
                k_sources=best_k, n_restarts=self.n_restarts
            )
            cof_final.fit(U_train_val[:, :best_k], e_train_val)
            fused_test = cof_final.predict(U_test[:, :best_k])

            if np.std(fused_test) > 1e-12:
                test_corr, test_pval = pearsonr(fused_test, e_test)
            else:
                test_corr, test_pval = 0.0, 1.0

            all_test_correlations.append(test_corr)

            fold_results.append({
                "fold": fold_idx,
                "best_k": best_k,
                "inner_k_scores": {
                    k: {
                        "mean": float(np.mean(k_inner_scores[k])),
                        "std": float(np.std(k_inner_scores[k])),
                    }
                    for k in k_values
                    if k_inner_scores[k]
                },
                "test_correlation": float(test_corr),
                "test_p_value": float(test_pval),
                "weights": cof_final.weights_.tolist(),
                "n_train_val": len(train_val_idx),
                "n_test": len(test_idx),
            })

        total_time = time.time() - start_time

        # Aggregate
        test_corrs = np.array(all_test_correlations)

        return {
            "folds": fold_results,
            "aggregate": {
                "mean_correlation": float(np.mean(test_corrs)),
                "std_correlation": float(np.std(test_corrs)),
                "min_correlation": float(np.min(test_corrs)),
                "max_correlation": float(np.max(test_corrs)),
                "cv": float(np.std(test_corrs) / abs(np.mean(test_corrs)))
                if abs(np.mean(test_corrs)) > 1e-12
                else 0.0,
            },
            "configuration": {
                "outer_folds": self.outer_folds,
                "inner_folds": self.inner_folds,
                "k_values": k_values,
                "n_restarts": self.n_restarts,
                "seed": seed,
            },
            "total_time_seconds": total_time,
        }

    def compare_with_standard_cv(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        seed: int = 42,
    ) -> Dict:
        """
        Compare nested CV vs standard (biased) train/val split.
        Quantifies validation bias from the CVPR paper.
        """
        # Standard split (potential bias)
        n = len(errors)
        idx = np.random.RandomState(seed).permutation(n)
        split = int(0.7 * n)
        train_idx, val_idx = idx[:split], idx[split:]

        K = uncertainties.shape[1]
        cof_standard = CorrelationOptimizedFusion(
            k_sources=K, n_restarts=self.n_restarts
        )
        cof_standard.fit(uncertainties[train_idx], errors[train_idx])
        fused_val = cof_standard.predict(uncertainties[val_idx])
        standard_corr, _ = pearsonr(fused_val, errors[val_idx]) if np.std(fused_val) > 1e-12 else (0.0, 1.0)

        # Nested CV (unbiased)
        nested_results = self.evaluate(uncertainties, errors, seed=seed)

        bias = standard_corr - nested_results["aggregate"]["mean_correlation"]

        return {
            "standard_cv_correlation": float(standard_corr),
            "nested_cv_correlation": nested_results["aggregate"]["mean_correlation"],
            "estimated_bias": float(bias),
            "bias_pct": float(bias / abs(standard_corr) * 100)
            if abs(standard_corr) > 1e-12
            else 0.0,
            "nested_cv_details": nested_results,
        }
