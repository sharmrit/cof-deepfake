"""
Systematic ablation studies for COF.

Ablation dimensions:
  1. Source selection (K=2,3,4,5)
  2. Individual source contribution (leave-one-out)
  3. Constraint type (simplex vs non-negative)
  4. Normalization method
  5. Distributional paradox verification
"""

import numpy as np
import itertools
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple

from ..config import UNCERTAINTY_SOURCES
from ..fusion.cof import CorrelationOptimizedFusion


class AblationStudy:
    """
    Comprehensive ablation framework for COF.

    Parameters
    ----------
    uncertainties : ndarray (N, 5)
    errors : ndarray (N,)
    source_names : list of str
    """

    def __init__(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        source_names: Optional[List[str]] = None,
        n_restarts: int = 10,
    ):
        self.U = uncertainties
        self.errors = errors
        self.source_names = source_names or list(UNCERTAINTY_SOURCES)
        self.n_restarts = n_restarts
        self.n_sources = uncertainties.shape[1]

    # -----------------------------------------------------------------
    # 1. Source Selection Ablation (K = 2..5)
    # -----------------------------------------------------------------
    def source_selection_ablation(self) -> Dict[int, Dict]:
        """
        Run COF with varying K (number of sources).

        Returns dict mapping K -> {correlation, weights, source_names}.
        """
        # Rank by individual correlation
        individual_corrs = []
        for i in range(self.n_sources):
            c, _ = pearsonr(self.U[:, i], self.errors) if np.std(self.U[:, i]) > 1e-12 else (0.0, 1.0)
            individual_corrs.append((i, c))
        ranked = sorted(individual_corrs, key=lambda x: -x[1])

        results = {}
        for k in range(2, self.n_sources + 1):
            top_k_idx = [idx for idx, _ in ranked[:k]]
            top_k_names = [self.source_names[i] for i in top_k_idx]
            U_k = self.U[:, top_k_idx]

            cof = CorrelationOptimizedFusion(
                k_sources=k, n_restarts=self.n_restarts
            )
            cof.fit(U_k, self.errors, source_names=top_k_names)

            results[k] = {
                "correlation": cof.result_.correlation,
                "weights": dict(zip(top_k_names, cof.weights_.tolist())),
                "source_names": top_k_names,
                "optimization_time": cof.result_.optimization_time,
            }

        return results

    # -----------------------------------------------------------------
    # 2. Leave-One-Out Ablation
    # -----------------------------------------------------------------
    def leave_one_out(self) -> Dict[str, Dict]:
        """
        Remove each source individually and measure impact.

        Returns dict mapping removed_source -> {correlation, delta}.
        """
        # Full model baseline
        cof_full = CorrelationOptimizedFusion(
            k_sources=self.n_sources, n_restarts=self.n_restarts
        )
        cof_full.fit(self.U, self.errors, source_names=self.source_names)
        full_corr = cof_full.result_.correlation

        results = {}
        for i, name in enumerate(self.source_names):
            remaining_idx = [j for j in range(self.n_sources) if j != i]
            remaining_names = [self.source_names[j] for j in remaining_idx]
            U_loo = self.U[:, remaining_idx]

            cof = CorrelationOptimizedFusion(
                k_sources=len(remaining_idx), n_restarts=self.n_restarts
            )
            cof.fit(U_loo, self.errors, source_names=remaining_names)

            delta = cof.result_.correlation - full_corr
            results[name] = {
                "correlation_without": cof.result_.correlation,
                "delta": delta,
                "delta_pct": delta / abs(full_corr) * 100 if abs(full_corr) > 1e-12 else 0.0,
                "weights": dict(zip(remaining_names, cof.weights_.tolist())),
            }

        results["_baseline"] = {"full_correlation": full_corr}
        return results

    # -----------------------------------------------------------------
    # 3. Constraint Type Ablation
    # -----------------------------------------------------------------
    def constraint_ablation(self) -> Dict[str, Dict]:
        """Compare simplex vs non-negative constraint types."""
        results = {}
        for constraint in ["simplex", "non_negative"]:
            cof = CorrelationOptimizedFusion(
                k_sources=self.n_sources,
                constraint=constraint,
                n_restarts=self.n_restarts,
            )
            cof.fit(self.U, self.errors, source_names=self.source_names)
            results[constraint] = {
                "correlation": cof.result_.correlation,
                "weights": dict(zip(self.source_names, cof.weights_.tolist())),
                "sparsity": float(np.mean(cof.weights_ < 1e-4)),
                "weight_entropy": cof.get_weight_analysis()["entropy"],
            }
        return results

    # -----------------------------------------------------------------
    # 4. Distributional Paradox Analysis
    # -----------------------------------------------------------------
    def distributional_paradox(self) -> Dict:
        """
        Analyze the distributional paradox: distributional source receives
        ~zero weight yet systematically affects performance.
        """
        dist_idx = self.source_names.index("distributional") if "distributional" in self.source_names else -1
        if dist_idx < 0:
            return {"error": "distributional source not found"}

        # K=5 (with distributional)
        cof_5 = CorrelationOptimizedFusion(
            k_sources=self.n_sources, n_restarts=self.n_restarts
        )
        cof_5.fit(self.U, self.errors, source_names=self.source_names)
        w5 = cof_5.weights_

        # K=4 (without distributional)
        idx_4 = [i for i in range(self.n_sources) if i != dist_idx]
        names_4 = [self.source_names[i] for i in idx_4]
        cof_4 = CorrelationOptimizedFusion(
            k_sources=len(idx_4), n_restarts=self.n_restarts
        )
        cof_4.fit(self.U[:, idx_4], self.errors, source_names=names_4)

        delta = cof_5.result_.correlation - cof_4.result_.correlation
        w_dist = w5[dist_idx]

        return {
            "distributional_weight": float(w_dist),
            "weight_is_zero": bool(w_dist < 1e-4),
            "correlation_k5": cof_5.result_.correlation,
            "correlation_k4": cof_4.result_.correlation,
            "delta": float(delta),
            "delta_pct": float(delta / abs(cof_4.result_.correlation) * 100)
            if abs(cof_4.result_.correlation) > 1e-12
            else 0.0,
            "paradox_confirmed": bool(w_dist < 1e-4 and abs(delta) > 0.001),
            "weights_k5": dict(zip(self.source_names, w5.tolist())),
            "weights_k4": dict(zip(names_4, cof_4.weights_.tolist())),
        }

    # -----------------------------------------------------------------
    # 5. All Combinations Ablation
    # -----------------------------------------------------------------
    def all_combinations(self) -> Dict[str, Dict]:
        """
        Exhaustive evaluation of all possible source combinations.
        For 5 sources: 2^5 - 5 - 1 = 26 combinations (excluding singletons
        and empty set).
        """
        results = {}
        for r in range(2, self.n_sources + 1):
            for combo in itertools.combinations(range(self.n_sources), r):
                names = [self.source_names[i] for i in combo]
                key = "+".join(names)
                U_combo = self.U[:, list(combo)]

                cof = CorrelationOptimizedFusion(
                    k_sources=len(combo), n_restarts=self.n_restarts
                )
                cof.fit(U_combo, self.errors, source_names=names)

                results[key] = {
                    "sources": names,
                    "k": len(combo),
                    "correlation": cof.result_.correlation,
                    "weights": dict(zip(names, cof.weights_.tolist())),
                }

        # Sort by correlation
        results = dict(
            sorted(results.items(), key=lambda x: -x[1]["correlation"])
        )
        return results

    # -----------------------------------------------------------------
    # Run All Ablations
    # -----------------------------------------------------------------
    def run_all(self) -> Dict:
        """Run all ablation studies and return consolidated results."""
        return {
            "source_selection": self.source_selection_ablation(),
            "leave_one_out": self.leave_one_out(),
            "constraint_type": self.constraint_ablation(),
            "distributional_paradox": self.distributional_paradox(),
            "all_combinations": self.all_combinations(),
        }
