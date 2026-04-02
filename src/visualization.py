"""
visualization.py — Publication-Ready Plotting for the Memory Stress Study

Generates the following key figures:
  1. Accuracy vs Sequence Length (grouped bar chart)
  2. Training Loss Curves per model per sequence length
  3. Gradient Norm Evolution across epochs  ← core research figure
  4. Heatmap of final accuracy (model × seq_len)
  5. Training time comparison

All plots follow a consistent dark research-paper aesthetic with
tight layouts suitable for inclusion in a B.Tech project report.

Author : GRU-vs-LSTM Research Project
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List

from src.metrics import MetricsLogger

# ──────────────────────────────────────────────────────────────────────
#  Global style
# ──────────────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family": "sans-serif",
    "font.size": 11,
}

MODEL_COLORS = {
    "RNN":       "#f85149",   # red — failure
    "GRU":       "#58a6ff",   # blue
    "LSTM":      "#3fb950",   # green — success
    "Attn-LSTM": "#d2a8ff",   # purple — novelty
}

plt.rcParams.update(STYLE)


def _savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
#  1. Accuracy vs Sequence Length  (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_seqlen(
    results: Dict[str, Dict[int, MetricsLogger]],
    seq_lengths: List[int],
    save_path: str = "plots/accuracy_vs_seqlen.png",
) -> None:
    """
    results : {model_name: {seq_len: MetricsLogger}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seq_lengths))
    n_models = len(results)
    width = 0.8 / n_models

    for i, (model_name, seq_dict) in enumerate(results.items()):
        accs = [seq_dict[sl].get_best_val_acc() * 100 for sl in seq_lengths]
        color = MODEL_COLORS.get(model_name, "#ffffff")
        bars = ax.bar(x + i * width, accs, width, label=model_name,
                      color=color, edgecolor="white", linewidth=0.3, alpha=0.9)
        # Value labels on bars
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xlabel("Sequence Length", fontsize=13, fontweight="bold")
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("Memory Retention Under Stress — Accuracy vs Sequence Length",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(seq_lengths)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y")
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  2. Training Loss Curves
# ══════════════════════════════════════════════════════════════════════

def plot_loss_curves(
    results: Dict[str, Dict[int, MetricsLogger]],
    seq_lengths: List[int],
    save_path: str = "plots/loss_curves.png",
) -> None:
    n_cols = len(seq_lengths)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for j, sl in enumerate(seq_lengths):
        ax = axes[j]
        for model_name, seq_dict in results.items():
            logger = seq_dict[sl]
            color = MODEL_COLORS.get(model_name, "#ffffff")
            ax.plot(logger.train_losses, label=model_name, color=color, linewidth=1.8)
        ax.set_title(f"Seq Len = {sl}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Training Loss")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("Training Loss Convergence Across Sequence Lengths",
                 fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  3. Gradient Norm Evolution  — THE KEY RESEARCH FIGURE
# ══════════════════════════════════════════════════════════════════════

def plot_gradient_norms(
    results: Dict[str, Dict[int, MetricsLogger]],
    seq_lengths: List[int],
    save_path: str = "plots/gradient_norms.png",
) -> None:
    """
    Plot gradient norms over epochs for every (model, seq_len) combo.
    This figure visually *proves* vanishing gradients in Vanilla RNNs
    and the stability of LSTM / GRU gating.
    """
    n_cols = len(seq_lengths)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for j, sl in enumerate(seq_lengths):
        ax = axes[j]
        for model_name, seq_dict in results.items():
            logger = seq_dict[sl]
            color = MODEL_COLORS.get(model_name, "#ffffff")
            ax.plot(logger.grad_norms, label=model_name, color=color, linewidth=1.8)
        ax.set_title(f"Seq Len = {sl}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Gradient L2 Norm (recurrent weights)")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("Gradient Norm Dynamics — Evidence of Vanishing / Exploding Gradients",
                 fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  4. Accuracy Heatmap (model × seq_len)
# ══════════════════════════════════════════════════════════════════════

def plot_accuracy_heatmap(
    results: Dict[str, Dict[int, MetricsLogger]],
    seq_lengths: List[int],
    save_path: str = "plots/accuracy_heatmap.png",
) -> None:
    model_names = list(results.keys())
    data = np.zeros((len(model_names), len(seq_lengths)))

    for i, mn in enumerate(model_names):
        for j, sl in enumerate(seq_lengths):
            data[i, j] = results[mn][sl].get_best_val_acc() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=seq_lengths, yticklabels=model_names,
        linewidths=0.5, linecolor="#30363d",
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy Heatmap — Model × Sequence Length",
                 fontsize=13, fontweight="bold", pad=12)
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  5. Training Time Comparison
# ══════════════════════════════════════════════════════════════════════

def plot_training_times(
    times: Dict[str, Dict[int, float]],
    seq_lengths: List[int],
    save_path: str = "plots/training_times.png",
) -> None:
    """
    times : {model_name: {seq_len: seconds}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, sl_dict in times.items():
        color = MODEL_COLORS.get(model_name, "#ffffff")
        t = [sl_dict[sl] for sl in seq_lengths]
        ax.plot(seq_lengths, t, marker="o", label=model_name,
                color=color, linewidth=2, markersize=7)

    ax.set_xlabel("Sequence Length", fontsize=13, fontweight="bold")
    ax.set_ylabel("Training Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_title("Computational Cost — Training Time vs Sequence Length",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True)
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  6. Validation Accuracy Curves over Epochs
# ══════════════════════════════════════════════════════════════════════

def plot_val_accuracy_curves(
    results: Dict[str, Dict[int, MetricsLogger]],
    seq_lengths: List[int],
    save_path: str = "plots/val_accuracy_curves.png",
) -> None:
    n_cols = len(seq_lengths)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for j, sl in enumerate(seq_lengths):
        ax = axes[j]
        for model_name, seq_dict in results.items():
            logger = seq_dict[sl]
            color = MODEL_COLORS.get(model_name, "#ffffff")
            accs_pct = [a * 100 for a in logger.val_accs]
            ax.plot(accs_pct, label=model_name, color=color, linewidth=1.8)
        ax.set_title(f"Seq Len = {sl}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        if j == 0:
            ax.set_ylabel("Validation Accuracy (%)")
        ax.axhline(y=100 / 8 * 1, color="#8b949e", linestyle=":", linewidth=0.8,
                   label="Random baseline" if j == 0 else None)
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("Validation Accuracy Evolution Over Training",
                 fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════
#  Master plot generator
# ══════════════════════════════════════════════════════════════════════

def generate_all_plots(
    results: Dict[str, Dict[int, MetricsLogger]],
    times: Dict[str, Dict[int, float]],
    seq_lengths: List[int],
    output_dir: str = "plots",
) -> None:
    """Generate every figure and save to output_dir."""
    print("\n📊  Generating publication-quality plots...")
    plot_accuracy_vs_seqlen(results, seq_lengths, f"{output_dir}/accuracy_vs_seqlen.png")
    plot_loss_curves(results, seq_lengths, f"{output_dir}/loss_curves.png")
    plot_gradient_norms(results, seq_lengths, f"{output_dir}/gradient_norms.png")
    plot_accuracy_heatmap(results, seq_lengths, f"{output_dir}/accuracy_heatmap.png")
    plot_training_times(times, seq_lengths, f"{output_dir}/training_times.png")
    plot_val_accuracy_curves(results, seq_lengths, f"{output_dir}/val_accuracy_curves.png")
    print("✅  All plots generated!\n")
