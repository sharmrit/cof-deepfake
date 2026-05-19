"""
Configuration and constants for the COF-UQ framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import yaml


# =============================================================================
# Constants
# =============================================================================

# TIFS extended: 10-11 architectures (original 8 + new additions)
ARCHITECTURES = [
    # --- Original CVPR 8 ---
    "xception",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b4",
    "vit_base_patch16_224",
    "deit_base_patch16_224",
    "swin_base_patch4_window7_224",
    "convnext_base",
    # --- TIFS additions ---
    "resnet101",          # Deeper CNN baseline
    "efficientnet_v2_s",  # Modern efficient CNN
    "maxvit_base_tf_224", # MaxViT hybrid architecture
]

ARCHITECTURE_FAMILIES = {
    "cnn": ["xception", "resnet50", "resnet101"],
    "efficientnet": ["efficientnet_b0", "efficientnet_b4", "efficientnet_v2_s"],
    "transformer": ["vit_base_patch16_224", "deit_base_patch16_224"],
    "hybrid": ["swin_base_patch4_window7_224", "convnext_base", "maxvit_base_tf_224"],
}

# Short display names for figures / tables
ARCH_SHORT_NAMES = {
    "xception": "Xception",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "efficientnet_b0": "EffNet-B0",
    "efficientnet_b4": "EffNet-B4",
    "efficientnet_v2_s": "EffNetV2-S",
    "vit_base_patch16_224": "ViT-B",
    "deit_base_patch16_224": "DeiT-B",
    "swin_base_patch4_window7_224": "Swin-B",
    "convnext_base": "ConvNeXt-B",
    "maxvit_base_tf_224": "MaxViT-B",
}

DATASETS = ["faceforensics", "celebdf", "dfdc"]

UNCERTAINTY_SOURCES = [
    "epistemic",       # MC Dropout variance
    "aleatoric",       # Predictive entropy / variance
    "calibration",     # 1 - max(softmax)
    "conformal",       # Nonconformity score
    "distributional",  # Mahalanobis distance
]

SEEDS = [42, 43, 44, 45, 46]

# All 13 fusion methods evaluated in the paper
FUSION_METHODS = [
    # --- COF variants (ours) ---
    "cof_k5",            # COF with all 5 sources
    "cof_k4",            # COF with 4 sources (no distributional)
    "cof_k3",            # COF with top-3 sources
    "cof_k2",            # COF with top-2 sources
    "cof_k5_constrained", # COF-5 with non-negativity only
    "cof_k4_constrained", # COF-4 with non-negativity only
    # --- Baselines ---
    "uniform_average",    # Equal-weight averaging
    "best_single",        # Best individual source (conformal)
    "pca_fusion",         # PCA-based dimensionality reduction
    "entropy_weighted",   # Entropy-based weighting
    "rank_fusion",        # Rank-based fusion
    "mc_dropout_ensemble",# MC Dropout ensemble baseline
    "deep_ensemble",      # Deep ensemble baseline
]


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class DataConfig:
    """Dataset paths and loading parameters."""
    ff_root: str = "datasets/FaceForensics"
    celebdf_root: str = "datasets/CelebDF"
    dfdc_root: str = "datasets/DFDC"
    image_size: int = 224
    num_workers: int = 4
    max_samples_per_class: Optional[int] = None  # None = use all


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # cosine | step | plateau
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    label_smoothing: float = 0.0
    balanced_sampling: bool = True
    mc_dropout_rate: float = 0.3
    mc_forward_passes: int = 30
    freeze_backbone_epochs: int = 0  # 0 = no freezing


@dataclass
class FusionConfig:
    """COF and fusion method parameters."""
    optimizer: str = "SLSQP"
    max_iter: int = 1000
    tolerance: float = 1e-10
    n_restarts: int = 10             # Multi-start optimization
    constraint_type: str = "simplex"  # simplex | non_negative
    regularization: float = 0.0      # L2 regularization on weights


@dataclass
class EvalConfig:
    """Evaluation parameters."""
    n_bootstrap: int = 1000          # Bootstrap samples for CI
    confidence_level: float = 0.95
    conformal_alpha: float = 0.1     # Coverage 1-alpha
    calibration_bins: int = 15       # ECE bins


@dataclass
class HessianConfig:
    """Hessian analysis parameters."""
    n_eigenvalues: int = 20          # Top eigenvalues to compute
    batch_size: int = 128
    tol: float = 1e-4


@dataclass
class NestedCVConfig:
    """Nested k-fold cross-validation parameters."""
    outer_folds: int = 5
    inner_folds: int = 3
    stratified: bool = True


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hessian: HessianConfig = field(default_factory=HessianConfig)
    nested_cv: NestedCVConfig = field(default_factory=NestedCVConfig)

    # Global
    architectures: List[str] = field(default_factory=lambda: list(ARCHITECTURES))
    seeds: List[str] = field(default_factory=lambda: list(SEEDS))
    device: str = "cuda"
    output_dir: str = "./results"
    figure_dir: str = "./figures"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        for section_name, section_dict in raw.items():
            if hasattr(cfg, section_name) and isinstance(section_dict, dict):
                sub = getattr(cfg, section_name)
                if hasattr(sub, "__dataclass_fields__"):
                    for k, v in section_dict.items():
                        if hasattr(sub, k):
                            setattr(sub, k, v)
                else:
                    setattr(cfg, section_name, section_dict)
            elif hasattr(cfg, section_name):
                setattr(cfg, section_name, raw[section_name])
        return cfg

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        import dataclasses
        d = {}
        for fld in dataclasses.fields(self):
            val = getattr(self, fld.name)
            if dataclasses.is_dataclass(val):
                d[fld.name] = dataclasses.asdict(val)
            else:
                d[fld.name] = val
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for d in [self.output_dir, self.figure_dir,
                  self.checkpoint_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
