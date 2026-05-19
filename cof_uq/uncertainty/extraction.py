"""
Batch uncertainty extraction pipeline.

Runs a trained model over a dataset and computes all five uncertainty
sources in a single pass (with MC Dropout forward passes).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from ..models.architectures import DeepfakeDetector
from .sources import (
    compute_epistemic,
    compute_aleatoric,
    compute_calibration,
    compute_conformal,
    compute_distributional,
    fit_distributional_params,
    compute_conformal_calibration_scores,
)
from .normalization import MinMaxNormalizer
from ..config import UNCERTAINTY_SOURCES


class UncertaintyExtractor:
    """
    Extracts all five uncertainty sources from a trained model on a dataset.

    Parameters
    ----------
    model : DeepfakeDetector
        Trained model with MC Dropout support.
    n_mc_passes : int
        Number of MC Dropout forward passes.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        n_mc_passes: int = 30,
        device: str = "cuda",
    ):
        self.model = model
        self.n_mc_passes = n_mc_passes
        self.device = device

        # Distributional params (fitted from training set)
        self._train_mean: Optional[np.ndarray] = None
        self._train_cov_inv: Optional[np.ndarray] = None
        # Conformal calibration scores
        self._conformal_cal_scores: Optional[np.ndarray] = None

    def fit_reference_distribution(self, train_loader: DataLoader) -> None:
        """
        Fit distributional parameters from training data features.
        Also computes conformal calibration scores.
        """
        all_features = []
        all_mc_probs = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Fitting reference distribution"):
                images = images.to(self.device)
                mc_logits, features = self.model.mc_forward(
                    images, n_passes=self.n_mc_passes
                )
                mc_probs = F.softmax(mc_logits, dim=-1).cpu().numpy()
                all_mc_probs.append(mc_probs)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_mc_probs = np.concatenate(all_mc_probs, axis=1)  # (T, N_total, C)
        all_labels = np.concatenate(all_labels)

        # Fit Mahalanobis parameters
        self._train_mean, self._train_cov_inv = fit_distributional_params(
            all_features, use_ledoit_wolf=True
        )

        # Compute conformal calibration scores
        self._conformal_cal_scores = compute_conformal_calibration_scores(
            all_mc_probs, all_labels
        )

    @torch.no_grad()
    def extract(
        self,
        data_loader: DataLoader,
        normalizer: Optional[MinMaxNormalizer] = None,
        fit_normalizer: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extract all uncertainty sources from a data loader.

        Parameters
        ----------
        data_loader : DataLoader
        normalizer : MinMaxNormalizer, optional
            If provided, normalize each source to [0, 1].
        fit_normalizer : bool
            If True, fit the normalizer on this data (use for training set).

        Returns
        -------
        results : dict with keys:
            'uncertainties' : ndarray (N, 5) — all five sources stacked
            'epistemic', 'aleatoric', 'calibration', 'conformal',
            'distributional' : ndarray (N,) — individual raw sources
            'predictions' : ndarray (N,) — predicted classes
            'labels' : ndarray (N,) — ground truth
            'errors' : ndarray (N,) — binary error indicator
            'probs' : ndarray (N, 2) — mean softmax probabilities
            'features' : ndarray (N, D) — penultimate features
            'mc_probs' : ndarray (T, N, 2) — all MC pass probabilities
        """
        all_mc_probs = []
        all_features = []
        all_labels = []

        self.model.eval()
        for images, labels in tqdm(data_loader, desc="Extracting uncertainties"):
            images = images.to(self.device)
            mc_logits, features = self.model.mc_forward(
                images, n_passes=self.n_mc_passes
            )
            mc_probs = F.softmax(mc_logits, dim=-1).cpu().numpy()
            all_mc_probs.append(mc_probs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        # Concatenate
        mc_probs = np.concatenate(all_mc_probs, axis=1)   # (T, N, C)
        features = np.concatenate(all_features, axis=0)    # (N, D)
        labels = np.concatenate(all_labels)                # (N,)
        mean_probs = np.mean(mc_probs, axis=0)             # (N, C)
        predictions = np.argmax(mean_probs, axis=1)        # (N,)
        errors = (predictions != labels).astype(np.float64)

        # Compute each source
        epistemic = compute_epistemic(mc_probs)
        aleatoric = compute_aleatoric(mean_probs)
        calibration = compute_calibration(mean_probs)
        conformal = compute_conformal(
            mc_probs, cal_scores=self._conformal_cal_scores
        )
        distributional = compute_distributional(
            features,
            train_mean=self._train_mean,
            train_cov_inv=self._train_cov_inv,
        )

        # Stack into (N, 5) matrix
        raw_sources = {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "calibration": calibration,
            "conformal": conformal,
            "distributional": distributional,
        }

        # Normalize if requested
        if normalizer is not None:
            if fit_normalizer:
                normalizer.fit(raw_sources)
            norm_sources = normalizer.transform(raw_sources)
        else:
            norm_sources = raw_sources

        uncertainties = np.column_stack([
            norm_sources[s] for s in UNCERTAINTY_SOURCES
        ])

        return {
            "uncertainties": uncertainties,
            **{f"{s}_raw": raw_sources[s] for s in UNCERTAINTY_SOURCES},
            **{s: norm_sources[s] for s in UNCERTAINTY_SOURCES},
            "predictions": predictions,
            "labels": labels,
            "errors": errors,
            "probs": mean_probs,
            "features": features,
            "mc_probs": mc_probs,
        }

    def extract_and_save(
        self,
        data_loader: DataLoader,
        save_path: str,
        normalizer: Optional[MinMaxNormalizer] = None,
        fit_normalizer: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Extract uncertainties and save to .npz file."""
        results = self.extract(data_loader, normalizer, fit_normalizer)
        np.savez_compressed(save_path, **results)
        return results
