"""
Min-max normalization for uncertainty sources (Eq. 3 in paper).
"""

import numpy as np
from typing import Dict, Optional
from ..config import UNCERTAINTY_SOURCES


class MinMaxNormalizer:
    """
    Fits min/max statistics on training data, then normalizes to [0, 1].

    Stores per-source statistics so that validation/test data can be
    normalized using training set parameters (avoiding information leakage).
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, sources: Dict[str, np.ndarray]) -> "MinMaxNormalizer":
        """
        Fit normalization statistics from training sources.

        Parameters
        ----------
        sources : dict mapping source name -> ndarray of shape (N,)
        """
        for name in UNCERTAINTY_SOURCES:
            if name not in sources:
                continue
            vals = sources[name]
            self.stats[name] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        self._fitted = True
        return self

    def transform(
        self, sources: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Normalize sources using fitted statistics.

        Clips values to [0, 1] to handle out-of-distribution samples
        that exceed training range.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        normalized = {}
        for name in UNCERTAINTY_SOURCES:
            if name not in sources:
                continue
            vals = sources[name]
            s = self.stats.get(name, {"min": 0.0, "max": 1.0})
            denom = s["max"] - s["min"]
            if denom < 1e-12:
                normalized[name] = np.zeros_like(vals)
            else:
                normalized[name] = np.clip(
                    (vals - s["min"]) / denom, 0.0, 1.0
                )
        return normalized

    def fit_transform(
        self, sources: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return self.fit(sources).transform(sources)

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MinMaxNormalizer":
        import json
        norm = cls()
        with open(path) as f:
            norm.stats = json.load(f)
        norm._fitted = True
        return norm
