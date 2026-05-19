"""
Unified deepfake detection architectures with feature extraction hooks.

Supports 11 architectures spanning CNNs, EfficientNets, Vision Transformers,
and hybrid models. Each model exposes:
  - Forward pass with binary classification output
  - Feature extraction from penultimate layer
  - MC Dropout support for epistemic uncertainty
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional


class MCDropout(nn.Module):
    """Dropout that stays active during eval (for MC Dropout inference)."""

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, p=self.p, training=True)


class DeepfakeDetector(nn.Module):
    """
    Unified wrapper for all deepfake detection architectures.

    Parameters
    ----------
    arch_name : str
        Architecture identifier (see config.ARCHITECTURES).
    num_classes : int
        Number of output classes (default 2 for real/fake).
    mc_dropout_rate : float
        Dropout rate for MC Dropout inference.
    pretrained : bool
        Load ImageNet pretrained weights.
    """

    def __init__(
        self,
        arch_name: str,
        num_classes: int = 2,
        mc_dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.arch_name = arch_name
        self.num_classes = num_classes
        self._features: Optional[torch.Tensor] = None

        # ---- Build backbone via timm ----
        if arch_name == "xception":
            self.backbone = timm.create_model(
                "xception", pretrained=pretrained, num_classes=0
            )
            feature_dim = 2048
        elif arch_name == "resnet50":
            self.backbone = timm.create_model(
                "resnet50", pretrained=pretrained, num_classes=0
            )
            feature_dim = 2048
        elif arch_name == "resnet101":
            self.backbone = timm.create_model(
                "resnet101", pretrained=pretrained, num_classes=0
            )
            feature_dim = 2048
        elif arch_name == "efficientnet_b0":
            self.backbone = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=0
            )
            feature_dim = 1280
        elif arch_name == "efficientnet_b4":
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=pretrained, num_classes=0
            )
            feature_dim = 1792
        elif arch_name == "efficientnet_v2_s":
            self.backbone = timm.create_model(
                "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
            )
            feature_dim = 1280
        elif arch_name == "vit_base_patch16_224":
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=0
            )
            feature_dim = 768
        elif arch_name == "deit_base_patch16_224":
            self.backbone = timm.create_model(
                "deit_base_patch16_224", pretrained=pretrained, num_classes=0
            )
            feature_dim = 768
        elif arch_name == "swin_base_patch4_window7_224":
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,
            )
            feature_dim = 1024
        elif arch_name == "convnext_base":
            self.backbone = timm.create_model(
                "convnext_base", pretrained=pretrained, num_classes=0
            )
            feature_dim = 1024
        elif arch_name == "maxvit_base_tf_224":
            self.backbone = timm.create_model(
                "maxvit_base_tf_224.in1k",
                pretrained=pretrained,
                num_classes=0,
            )
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")

        self.feature_dim = feature_dim

        # ---- Classification head with MC Dropout ----
        self.mc_dropout = MCDropout(p=mc_dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer features (before classifier)."""
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Returns
        -------
        logits : Tensor of shape (B, num_classes)
        features : Tensor of shape (B, feature_dim)  [if return_features]
        """
        features = self.extract_features(x)
        self._features = features.detach()
        dropped = self.mc_dropout(features)
        logits = self.classifier(dropped)
        if return_features:
            return logits, features
        return logits

    def mc_forward(
        self, x: torch.Tensor, n_passes: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout forward passes.

        Returns
        -------
        mc_logits : Tensor of shape (n_passes, B, num_classes)
        features  : Tensor of shape (B, feature_dim)
        """
        mc_logits = []
        features = self.extract_features(x)
        for _ in range(n_passes):
            dropped = self.mc_dropout(features)
            logits = self.classifier(dropped)
            mc_logits.append(logits)
        mc_logits = torch.stack(mc_logits, dim=0)
        return mc_logits, features

    def get_last_features(self) -> Optional[torch.Tensor]:
        """Retrieve features from the last forward() call."""
        return self._features

    @property
    def short_name(self) -> str:
        from ..config import ARCH_SHORT_NAMES
        return ARCH_SHORT_NAMES.get(self.arch_name, self.arch_name)
