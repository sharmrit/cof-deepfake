"""
Model factory for creating, saving, and loading deepfake detectors.
"""

import torch
from pathlib import Path
from typing import Optional

from ..config import ARCHITECTURES
from .architectures import DeepfakeDetector


class ModelFactory:
    """Factory for creating and managing detector models."""

    @staticmethod
    def create(
        arch_name: str,
        num_classes: int = 2,
        mc_dropout_rate: float = 0.3,
        pretrained: bool = True,
        device: str = "cuda",
    ) -> DeepfakeDetector:
        if arch_name not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{arch_name}'. "
                f"Available: {ARCHITECTURES}"
            )
        model = DeepfakeDetector(
            arch_name=arch_name,
            num_classes=num_classes,
            mc_dropout_rate=mc_dropout_rate,
            pretrained=pretrained,
        )
        return model.to(device)

    @staticmethod
    def save_checkpoint(
        model: DeepfakeDetector,
        path: str,
        epoch: int,
        optimizer=None,
        metrics: Optional[dict] = None,
    ) -> None:
        checkpoint = {
            "arch_name": model.arch_name,
            "feature_dim": model.feature_dim,
            "num_classes": model.num_classes,
            "state_dict": model.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if metrics is not None:
            checkpoint["metrics"] = metrics
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(
        path: str,
        device: str = "cuda",
        mc_dropout_rate: float = 0.3,
    ) -> DeepfakeDetector:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = DeepfakeDetector(
            arch_name=checkpoint["arch_name"],
            num_classes=checkpoint["num_classes"],
            mc_dropout_rate=mc_dropout_rate,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def list_architectures():
        return list(ARCHITECTURES)
