# Correlation-Optimized Fusion (COF) for Deepfake Detection

Official code release for the IEEE Transactions on Information Forensics and Security
(TIFS) submission: "Architecture-Adaptive Uncertainty Fusion for Deepfake Detection."

## Authors

- Ritesh Sharma (Virginia Commonwealth University) <sharmar33@vcu.edu>
- Mohammad Ghasemigol (Virginia Commonwealth University)
- Yuichi Motai (Virginia Commonwealth University)

## Citation

If you use this code or pre-computed features, please cite:

    @article{sharma2026cof,
      title={Architecture-Adaptive Uncertainty Fusion for Deepfake Detection},
      author={Sharma, Ritesh and Ghasemigol, Mohammad and Motai, Yuichi},
      journal={IEEE Transactions on Information Forensics and Security},
      year={2026},
      note={Under review}
    }

## Overview

COF is a post-hoc, architecture-adaptive uncertainty quantification framework
that fuses five complementary uncertainty sources (epistemic, aleatoric,
calibration, conformal, distributional) by maximizing Pearson correlation
with prediction errors via constrained simplex optimization (SLSQP).

## Repository Structure

    cof_uq/                 Core COF package
      analysis/             Hessian, stability, nested CV
      data/                 Dataset loaders and transforms
      evaluation/           Cross-domain and ablation metrics
      fusion/               COF and 11 baseline fusion methods
      models/               11 detector architectures
      training/             Training loop and callbacks
      uncertainty/          Five uncertainty sources, extraction, normalization
      visualization/        Plotting utilities
    scripts/                Entry-point scripts
    configs/                Configuration files
    precomputed_features/   Pre-computed uncertainty features (11 archs)
    results_v2/             Symlink to precomputed_features for scripts
    setup.py
    requirements.txt
    LICENSE

## Installation

    pip install -e .
    pip install -r requirements.txt

## Reproducing Paper Results

### Option A: Using pre-computed features (recommended)

Skips training and extraction. Reproduces all Tables III-IX.

    # Reproduces Tables III, IV, VI, IX, XI
    python scripts/aggregate_results.py

    # Reproduces Table V (forensic utility metrics)
    python scripts/compute_forensic_utility.py

    # Reproduces Table VI (UQ baselines)
    python scripts/eval_baselines.py

    # Reproduces Tables IV, XI (cross-domain)
    python scripts/cross_domain_eval.py --all-archs --config configs/tifs.yaml

### Option B: Full reproduction from scratch

Requires datasets downloaded separately.

    # Step 1: Train detector
    python scripts/train_tifs.py --config configs/tifs.yaml --arch xception --seed 42

    # Step 2: Extract uncertainty features
    python scripts/extract_tifs.py --config configs/tifs.yaml --arch xception --dataset faceforensics --seed 42

    # Step 3: Aggregate
    python scripts/aggregate_results.py --config configs/tifs.yaml

## Pre-computed Features

Each .npz file in precomputed_features/ contains:

    uncertainties       (N, 5) normalized uncertainty matrix
    epistemic           epistemic uncertainty (MC Dropout variance)
    aleatoric           aleatoric uncertainty (Bernoulli variance)
    calibration         calibration uncertainty (temperature scaling gap)
    conformal           conformal nonconformity score
    distributional      Mahalanobis distance from training distribution
    predictions         (N,) predicted class
    labels              (N,) ground truth
    errors              (N,) binary error indicator
    probs               (N, 2) mean softmax probabilities

## Datasets

Datasets must be obtained directly from providers (license required):
- FaceForensics++: https://github.com/ondyari/FaceForensics
- CelebDF: https://github.com/yuezunli/celeb-deepfakeforensics
- DFDC: https://ai.facebook.com/datasets/dfdc/

## Hardware

Experiments run on NVIDIA A100 GPUs.
- Training: 4-9h per architecture
- MC Dropout extraction: 18-35 min per architecture (T=20 passes)
- COF fusion weight optimization: 42 seconds (CPU only, 20 restarts)

## Software

- Python 3.9+
- PyTorch 2.8
- scikit-learn, scipy, numpy
- timm (architecture library)

## License

This code is released under the MIT License. See LICENSE file for details.

## Contact

For questions about the code or paper, contact:
Ritesh Sharma <sharmar33@vcu.edu>
