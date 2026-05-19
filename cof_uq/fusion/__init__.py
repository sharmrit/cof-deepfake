from .cof import CorrelationOptimizedFusion
from .baselines import (
    uniform_average,
    best_single_source,
    pca_fusion,
    entropy_weighted_fusion,
    rank_fusion,
)
from .registry import FusionRegistry, run_all_methods
