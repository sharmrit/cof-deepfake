"""
COF-UQ: Correlation-Optimized Fusion for Uncertainty Quantification
===================================================================

TIFS Extended Version — Architecture-Adaptive Uncertainty Quantification
with Optimization Landscape Analysis for Deepfake Detection.

Modules
-------
models      : Neural network architectures (10-11 models)
data        : Dataset loaders and balanced sampling
uncertainty : Five uncertainty source extractors
fusion      : COF and 13 baseline fusion methods
evaluation  : Metrics, cross-domain, and ablation studies
analysis    : Hessian analysis, nested k-fold CV, stability
training    : Training pipeline with callbacks
visualization : Publication-quality figures
"""

__version__ = "2.0.0"

from .config import Config, ARCHITECTURES, DATASETS, SEEDS, UNCERTAINTY_SOURCES
