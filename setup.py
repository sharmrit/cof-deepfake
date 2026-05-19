"""
Correlation-Optimized Fusion (COF) for Deepfake Detection
TIFS Extended Version — Comprehensive Uncertainty Quantification Package
"""

from setuptools import setup, find_packages

setup(
    name="cof_uq",
    version="2.0.0",
    description=(
        "Correlation-Optimized Fusion for Deepfake Detection: "
        "Architecture-Adaptive Uncertainty Quantification with "
        "Optimization Landscape Analysis"
    ),
    author="Anonymous",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "timm>=0.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "Pillow>=9.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "pyhessian>=0.1",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "agentic": ["langchain", "langgraph", "langchain-anthropic"],
    },
    entry_points={
        "console_scripts": [
            "cof-train=scripts.train:main",
            "cof-extract=scripts.extract_uncertainty:main",
            "cof-fuse=scripts.run_cof:main",
            "cof-crossdomain=scripts.cross_domain_eval:main",
            "cof-hessian=scripts.run_hessian:main",
            "cof-nestedcv=scripts.run_nested_cv:main",
            "cof-ablation=scripts.run_ablation:main",
        ],
    },
)
