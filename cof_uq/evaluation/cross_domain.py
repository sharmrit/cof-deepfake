"""
Cross-domain evaluation: measures generalization of uncertainty estimates
across datasets (FF++ -> CelebDF, FF++ -> DFDC).

Key finding: correlations collapse from 0.42-0.56 in-domain to near-zero
or negative out-of-domain, representing catastrophic failure.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import pearsonr

from ..config import Config, ARCHITECTURES, DATASETS, ARCH_SHORT_NAMES
from ..fusion.cof import CorrelationOptimizedFusion
from .metrics import compute_correlation, full_evaluation


class CrossDomainEvaluator:
    """
    Evaluates cross-dataset generalization of COF and baseline methods.

    Workflow:
      1. Train COF weights on source dataset (FF++)
      2. Apply trained weights on target datasets (CelebDF, DFDC)
      3. Measure correlation degradation
    """

    def __init__(
        self,
        source_dataset: str = "faceforensics",
        target_datasets: Optional[List[str]] = None,
        config: Optional[Config] = None,
    ):
        self.source_dataset = source_dataset
        self.target_datasets = target_datasets or ["celebdf", "dfdc"]
        self.config = config or Config()

    def evaluate_architecture(
        self,
        arch_name: str,
        source_data: Dict[str, np.ndarray],
        target_data: Dict[str, Dict[str, np.ndarray]],
        k_sources: int = 5,
    ) -> Dict:
        """
        Evaluate one architecture across all datasets.

        Parameters
        ----------
        arch_name : str
        source_data : dict with 'uncertainties', 'errors', 'probs', 'labels'
        target_data : dict mapping dataset_name -> data_dict

        Returns
        -------
        results : dict mapping dataset_name -> evaluation metrics
        """
        # Fit COF on source domain
        cof = CorrelationOptimizedFusion(
            k_sources=k_sources,
            n_restarts=self.config.fusion.n_restarts,
        )
        cof.fit(
            source_data["uncertainties"][:, :k_sources],
            source_data["errors"],
        )

        results = {}

        # In-domain evaluation
        fused_source = cof.predict(source_data["uncertainties"][:, :k_sources])
        corr_source, _ = compute_correlation(fused_source, source_data["errors"])
        results[self.source_dataset] = {
            "correlation": corr_source,
            "n_samples": len(source_data["errors"]),
            "error_rate": float(source_data["errors"].mean()),
            "domain": "in-domain",
        }

        # Cross-domain evaluation
        for target_name, tgt_data in target_data.items():
            fused_target = cof.predict(tgt_data["uncertainties"][:, :k_sources])
            corr_target, pval = compute_correlation(fused_target, tgt_data["errors"])

            # Performance degradation
            degradation = (
                (corr_source - corr_target) / abs(corr_source) * 100
                if abs(corr_source) > 1e-12
                else 0.0
            )

            # Uncertainty inversion detection
            inversion = corr_target < 0

            results[target_name] = {
                "correlation": corr_target,
                "p_value": pval,
                "degradation_pct": degradation,
                "uncertainty_inversion": inversion,
                "n_samples": len(tgt_data["errors"]),
                "error_rate": float(tgt_data["errors"].mean()),
                "domain": "out-of-domain",
                "weights_used": cof.weights_.tolist(),
            }

        return {
            "architecture": arch_name,
            "k_sources": k_sources,
            "source_weights": cof.weights_.tolist(),
            "datasets": results,
        }

    def evaluate_all_architectures(
        self,
        all_source_data: Dict[str, Dict],
        all_target_data: Dict[str, Dict[str, Dict]],
        k_sources: int = 5,
    ) -> Dict:
        """
        Evaluate all architectures.

        Parameters
        ----------
        all_source_data : dict mapping arch_name -> source_data_dict
        all_target_data : dict mapping arch_name -> {dataset: data_dict}

        Returns
        -------
        summary : dict with per-architecture and aggregate results
        """
        arch_results = {}

        for arch_name in all_source_data:
            arch_results[arch_name] = self.evaluate_architecture(
                arch_name,
                all_source_data[arch_name],
                all_target_data.get(arch_name, {}),
                k_sources=k_sources,
            )

        # Aggregate statistics
        summary = self._compute_aggregate(arch_results)
        summary["per_architecture"] = arch_results
        return summary

    def _compute_aggregate(self, arch_results: Dict) -> Dict:
        """Compute aggregate cross-domain statistics."""
        in_domain_corrs = []
        out_domain_corrs = {ds: [] for ds in self.target_datasets}
        degradations = {ds: [] for ds in self.target_datasets}
        inversions = {ds: 0 for ds in self.target_datasets}

        for arch, res in arch_results.items():
            datasets = res["datasets"]
            if self.source_dataset in datasets:
                in_domain_corrs.append(datasets[self.source_dataset]["correlation"])
            for ds in self.target_datasets:
                if ds in datasets:
                    out_domain_corrs[ds].append(datasets[ds]["correlation"])
                    degradations[ds].append(datasets[ds]["degradation_pct"])
                    if datasets[ds]["uncertainty_inversion"]:
                        inversions[ds] += 1

        n_arch = len(arch_results)
        return {
            "n_architectures": n_arch,
            "in_domain": {
                "mean_correlation": float(np.mean(in_domain_corrs)) if in_domain_corrs else 0.0,
                "std_correlation": float(np.std(in_domain_corrs)) if in_domain_corrs else 0.0,
                "range": (
                    float(np.min(in_domain_corrs)),
                    float(np.max(in_domain_corrs)),
                ) if in_domain_corrs else (0.0, 0.0),
            },
            "out_of_domain": {
                ds: {
                    "mean_correlation": float(np.mean(out_domain_corrs[ds])) if out_domain_corrs[ds] else 0.0,
                    "mean_degradation_pct": float(np.mean(degradations[ds])) if degradations[ds] else 0.0,
                    "inversion_count": inversions[ds],
                    "inversion_rate": inversions[ds] / n_arch if n_arch > 0 else 0.0,
                }
                for ds in self.target_datasets
            },
        }

    def save_results(self, results: Dict, path: str) -> None:
        """Save results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=_convert)
