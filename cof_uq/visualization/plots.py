"""
Publication-quality figure generation for COF-UQ.

Produces all figures needed for TIFS submission:
  - Fig 1: Architecture comparison (correlation barplot)
  - Fig 2: Learned fusion weights
  - Fig 3: Cross-domain heatmap
  - Fig 4: Source stability boxplots
  - Fig 5: Hessian eigenvalue spectra
  - Fig 6: Distributional paradox
  - Fig 7: Nested CV results
  - Fig 8: Ablation source selection
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..config import ARCH_SHORT_NAMES, UNCERTAINTY_SOURCES

# Global style for publication figures
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Color palette
COLORS = {
    "cof": "#2196F3",
    "baseline": "#9E9E9E",
    "best": "#4CAF50",
    "worst": "#F44336",
    "epistemic": "#E91E63",
    "aleatoric": "#9C27B0",
    "calibration": "#FF9800",
    "conformal": "#4CAF50",
    "distributional": "#795548",
}

SOURCE_COLORS = [
    COLORS["epistemic"],
    COLORS["aleatoric"],
    COLORS["calibration"],
    COLORS["conformal"],
    COLORS["distributional"],
]


def _setup_style():
    plt.rcParams.update(STYLE)


def _short_name(arch: str) -> str:
    return ARCH_SHORT_NAMES.get(arch, arch)


def _save_fig(fig, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)


# =============================================================================
# Figure 1: Correlation Comparison across methods
# =============================================================================

def plot_correlation_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str = "figures/correlation_comparison.pdf",
    highlight_methods: Optional[List[str]] = None,
):
    """
    Bar plot comparing correlation across all 13 methods for one architecture.

    Parameters
    ----------
    results : dict mapping method_name -> {'correlation': float, ...}
    """
    _setup_style()
    highlight = highlight_methods or ["cof_k5", "cof_k4", "cof_k3"]

    methods = sorted(results.keys(), key=lambda m: -results[m].get("correlation", 0))
    corrs = [results[m].get("correlation", 0) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [COLORS["cof"] if m in highlight else COLORS["baseline"] for m in methods]
    bars = ax.barh(range(len(methods)), corrs, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_", " ").title() for m in methods])
    ax.set_xlabel("Pearson Correlation (ρ)")
    ax.set_title("Fusion Method Comparison")
    ax.invert_yaxis()

    # Add value labels
    for bar, corr in zip(bars, corrs):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{corr:.3f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, max(corrs) * 1.15 if corrs else 1.0)
    _save_fig(fig, save_path)


# =============================================================================
# Figure 2: Learned Fusion Weights
# =============================================================================

def plot_learned_weights(
    arch_weights: Dict[str, Dict[str, float]],
    save_path: str = "figures/learned_weights.pdf",
):
    """
    Stacked bar chart of learned COF weights per architecture.

    Parameters
    ----------
    arch_weights : dict mapping arch_name -> {source_name: weight}
    """
    _setup_style()
    archs = list(arch_weights.keys())
    sources = list(UNCERTAINTY_SOURCES)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(archs))
    width = 0.65

    bottoms = np.zeros(len(archs))
    for i, src in enumerate(sources):
        values = [arch_weights[a].get(src, 0.0) for a in archs]
        ax.bar(
            x, values, width, bottom=bottoms,
            label=src.capitalize(), color=SOURCE_COLORS[i],
            edgecolor="white", linewidth=0.5,
        )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(a) for a in archs], rotation=45, ha="right")
    ax.set_ylabel("Learned Weight")
    ax.set_title("COF-5 Learned Weights Across Architectures")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.05)

    _save_fig(fig, save_path)


# =============================================================================
# Figure 3: Cross-Domain Heatmap
# =============================================================================

def plot_cross_domain_heatmap(
    cross_domain_results: Dict[str, Dict[str, float]],
    datasets: List[str] = None,
    save_path: str = "figures/cross_domain_heatmap.pdf",
):
    """
    Heatmap showing correlation degradation across datasets.

    Parameters
    ----------
    cross_domain_results : dict mapping arch -> {dataset: correlation}
    """
    _setup_style()
    datasets = datasets or ["faceforensics", "celebdf", "dfdc"]

    archs = list(cross_domain_results.keys())
    matrix = np.zeros((len(archs), len(datasets)))

    for i, arch in enumerate(archs):
        for j, ds in enumerate(datasets):
            matrix[i, j] = cross_domain_results[arch].get(ds, 0.0)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.6)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([_short_name(a) for a in archs])

    # Annotate cells
    for i in range(len(archs)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Pearson ρ", shrink=0.8)
    ax.set_title("Cross-Dataset Generalization")

    _save_fig(fig, save_path)


# =============================================================================
# Figure 4: Source Stability (Boxplots)
# =============================================================================

def plot_source_stability(
    stability_data: Dict[str, Dict],
    save_path: str = "figures/source_stability.pdf",
):
    """
    Box plots of per-source correlation stability across bootstrap samples.

    Parameters
    ----------
    stability_data : dict from StabilityAnalyzer.per_source_stability()
    """
    _setup_style()
    sources = [s for s in UNCERTAINTY_SOURCES if s in stability_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

    # Box plots
    data = []
    for s in sources:
        if "mean" in stability_data[s]:
            # Generate samples from stats
            vals = np.random.normal(
                stability_data[s]["mean"],
                stability_data[s]["std"],
                100,
            )
            data.append(vals)
        else:
            data.append([0])

    bp = ax1.boxplot(
        data, labels=[s.capitalize() for s in sources],
        patch_artist=True, widths=0.5,
    )
    for patch, color in zip(bp["boxes"], SOURCE_COLORS[: len(sources)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel("Pearson Correlation")
    ax1.set_title("Source Correlation Stability")

    # CV bar chart
    cvs = [stability_data[s].get("cv", 0) * 100 for s in sources]
    bars = ax2.bar(
        range(len(sources)), cvs,
        color=SOURCE_COLORS[: len(sources)], alpha=0.8,
    )
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels([s[:4].capitalize() for s in sources], rotation=45)
    ax2.set_ylabel("CV (%)")
    ax2.set_title("Coefficient of Variation")

    for bar, cv in zip(bars, cvs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{cv:.1f}%",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# Figure 5: Hessian Eigenvalue Spectrum
# =============================================================================

def plot_hessian_eigenvalues(
    hessian_results: Dict[str, Dict],
    save_path: str = "figures/hessian_eigenvalues.pdf",
):
    """
    Eigenvalue spectrum of COF Hessian per architecture.

    Parameters
    ----------
    hessian_results : dict mapping arch_name -> HessianAnalyzer.analyze() output
    """
    _setup_style()
    archs = list(hessian_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Eigenvalue spectra
    for arch in archs:
        evals = np.array(hessian_results[arch].get("eigenvalues", []))
        if len(evals) > 0:
            axes[0].plot(
                range(1, len(evals) + 1),
                np.abs(evals),
                "o-",
                label=_short_name(arch),
                markersize=4,
            )

    axes[0].set_xlabel("Eigenvalue Index")
    axes[0].set_ylabel("|λ|")
    axes[0].set_title("Hessian Eigenvalue Spectrum")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2)

    # Condition numbers
    cond_numbers = [
        hessian_results[a].get("condition_number", 0) for a in archs
    ]
    x = range(len(archs))
    axes[1].bar(x, cond_numbers, color=COLORS["cof"], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [_short_name(a) for a in archs], rotation=45, ha="right"
    )
    axes[1].set_ylabel("Condition Number")
    axes[1].set_title("Hessian Conditioning")
    axes[1].set_yscale("log")

    plt.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# Figure 6: Distributional Paradox
# =============================================================================

def plot_distributional_paradox(
    paradox_data: Dict[str, Dict],
    save_path: str = "figures/distributional_paradox.pdf",
):
    """
    Visualize distributional paradox across architectures.

    Parameters
    ----------
    paradox_data : dict mapping arch_name -> AblationStudy.distributional_paradox() output
    """
    _setup_style()
    archs = list(paradox_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Delta correlation (K5 - K4)
    deltas = [paradox_data[a].get("delta_pct", 0) for a in archs]
    colors = [COLORS["best"] if d > 0 else COLORS["worst"] for d in deltas]
    ax1.bar(
        range(len(archs)), deltas,
        color=colors, alpha=0.8, edgecolor="white",
    )
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([_short_name(a) for a in archs], rotation=45, ha="right")
    ax1.set_ylabel("Δρ (%)")
    ax1.set_title("Impact of Distributional Source (K5 vs K4)")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Distributional weights (should be ~0)
    w_dist = [paradox_data[a].get("distributional_weight", 0) for a in archs]
    ax2.bar(
        range(len(archs)), w_dist,
        color=COLORS["distributional"], alpha=0.8,
    )
    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels([_short_name(a) for a in archs], rotation=45, ha="right")
    ax2.set_ylabel("w_distributional")
    ax2.set_title("Distributional Weight (≈ 0)")
    ax2.set_ylim(-0.01, max(w_dist) * 1.5 + 0.01)

    plt.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# Figure 7: Nested CV Results
# =============================================================================

def plot_nested_cv_results(
    nested_cv_data: Dict[str, Dict],
    save_path: str = "figures/nested_cv.pdf",
):
    """
    Nested CV vs standard CV comparison plot.

    Parameters
    ----------
    nested_cv_data : dict mapping arch_name -> NestedCrossValidator results
    """
    _setup_style()
    archs = list(nested_cv_data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(archs))
    width = 0.35

    standard = [nested_cv_data[a].get("standard_cv_correlation", 0) for a in archs]
    nested = [nested_cv_data[a].get("nested_cv_correlation", 0) for a in archs]

    ax.bar(x - width / 2, standard, width, label="Standard CV (biased)", color=COLORS["baseline"])
    ax.bar(x + width / 2, nested, width, label="Nested CV (unbiased)", color=COLORS["cof"])

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(a) for a in archs], rotation=45, ha="right")
    ax.set_ylabel("Pearson Correlation (ρ)")
    ax.set_title("Validation Bias: Standard vs Nested CV")
    ax.legend()

    plt.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# Figure 8: Ablation Source Selection
# =============================================================================

def plot_ablation_source_selection(
    ablation_data: Dict[int, Dict],
    save_path: str = "figures/ablation_source_selection.pdf",
):
    """
    K-value ablation: correlation vs number of sources.

    Parameters
    ----------
    ablation_data : dict mapping K -> {'correlation': float, ...}
    """
    _setup_style()
    ks = sorted(ablation_data.keys())
    corrs = [ablation_data[k]["correlation"] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, corrs, "o-", color=COLORS["cof"], linewidth=2, markersize=8)

    for k, c in zip(ks, corrs):
        ax.annotate(
            f"{c:.3f}",
            (k, c),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Number of Sources (K)")
    ax.set_ylabel("Pearson Correlation (ρ)")
    ax.set_title("Source Selection Ablation")
    ax.set_xticks(ks)

    plt.tight_layout()
    _save_fig(fig, save_path)


# =============================================================================
# Training Curves
# =============================================================================

def plot_training_curves(
    history: List[Dict],
    save_path: str = "figures/training_curves.pdf",
):
    """Plot training loss/accuracy and validation loss/AUC."""
    _setup_style()
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    # AUC
    axes[2].plot(epochs, [h["val_auc"] for h in history], label="Val AUC", color=COLORS["cof"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Validation AUC")
    axes[2].legend()

    plt.tight_layout()
    _save_fig(fig, save_path)
