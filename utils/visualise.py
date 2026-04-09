"""
utils/visualise.py
All publication-quality plots for Levels 2-4.

Functions:
  plot_training_curves     — loss + F1 per epoch
  plot_results_heatmap     — strategy × model F1 heatmap
  plot_bar_comparison      — grouped bar chart
  plot_temporal_drift      — per-timestep F1 (L3 key finding)
  plot_seed_variance       — error bars across seeds (L3)
  plot_ablation            — ablation bar chart (L4)
  plot_adaptive_switch     — annotated training curve with switch point (L4)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
})

STRATEGY_LABELS = {
    "baseline":   "Baseline",
    "weighted":   "Weighted CE",
    "oversample": "Oversampling",
    "graph_aug":  "Graph Aug.",
    "adaptive":   "Adaptive",
}
MODEL_LABELS = {
    "gcn":       "GCN",
    "sage":      "GraphSAGE",
    "gat":       "GAT",
    "edgegat":   "EdgeGAT",
    "evolvegcn": "EvolveGCN",
}
STRATEGY_COLORS = {
    "baseline":   "#6B7280",
    "weighted":   "#2563EB",
    "oversample": "#059669",
    "graph_aug":  "#D97706",
    "adaptive":   "#7C3AED",
}
MODEL_MARKERS = {
    "gcn": "D", "sage": "o", "gat": "s",
    "edgegat": "^", "evolvegcn": "P",
}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, title: str = "",
                         save_path: str = None,
                         switch_epochs: List[int] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    epochs = list(range(1, len(history["loss"]) + 1))

    ax1.plot(epochs, history["loss"], color="#EF4444", lw=2)
    ax1.set(xlabel="Checkpoint", ylabel="Training Loss", title="Loss")

    ax2.plot(epochs, history["train_f1"], label="Train F1", color="#2563EB", lw=2)
    ax2.plot(epochs, history["test_f1"],  label="Test F1",  color="#059669", lw=2)
    if switch_epochs:
        for se in switch_epochs:
            idx = se // 10 - 1
            ax2.axvline(idx, color="#7C3AED", linestyle="--", lw=1.2,
                        label=f"Strategy switch @ ep {se}")
    ax2.set(xlabel="Checkpoint", ylabel="F1 (fraud class)", title="F1 Score",
            ylim=(0, 1))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap (L2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_heatmap(df: pd.DataFrame, save_path: str = None,
                         metric: str = "f1"):
    strategies = [s for s in ["baseline","weighted","oversample","graph_aug"]
                  if s in df["strategy"].values]
    models     = [m for m in ["gcn","sage","gat"] if m in df["model"].values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Model × Strategy Performance (Test Set, Fraud Class)",
                 fontsize=12, fontweight="bold")

    for ax, met in zip(axes, ["f1", "precision", "recall"]):
        col = f"{met}_mean" if f"{met}_mean" in df.columns else met
        pivot = df.pivot(index="strategy", columns="model", values=col)
        pivot = pivot.loc[[s for s in strategies if s in pivot.index]]
        pivot.index   = [STRATEGY_LABELS.get(s, s) for s in pivot.index]
        pivot.columns = [MODEL_LABELS.get(m, m)    for m in pivot.columns]

        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.60, vmax=0.92, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        ax.set(title=met.capitalize(), xlabel="", ylabel="")

    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bar comparison (L2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(df: pd.DataFrame, save_path: str = None):
    strategies = [s for s in ["baseline","weighted","oversample","graph_aug"]
                  if s in df["strategy"].values]
    models     = [m for m in ["gcn","sage","gat"] if m in df["model"].values]

    col   = "f1_mean" if "f1_mean" in df.columns else "f1"
    x     = np.arange(len(strategies))
    width = 0.25
    colors = ["#2563EB", "#059669", "#D97706"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        sub  = df[df["model"] == model].set_index("strategy")
        vals = [sub.loc[s, col] if s in sub.index else 0 for s in strategies]
        std  = ([sub.loc[s, "f1_std"] if "f1_std" in sub.columns and s in sub.index else 0
                 for s in strategies])
        bars = ax.bar(x + i * width, vals, width, yerr=std,
                      label=MODEL_LABELS.get(model, model),
                      color=colors[i], alpha=0.85,
                      capsize=4, error_kw={"elinewidth": 1.2})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    ax.axhline(0.72, color="red", linestyle="--", lw=1.3,
               label="F1 target (0.72)", alpha=0.75)
    ax.set_xticks(x + width)
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies],
                       rotation=10, ha="right")
    ax.set(ylabel="F1 Score (Fraud Class)", ylim=(0.55, 0.95),
           title="Fraud Detection F1 by Model & Strategy")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Temporal drift plot (L3 — key figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_drift(
    temporal_results: Dict[str, Dict[int, Dict]],
    save_path: str = None,
):
    """
    Line chart: F1 vs time step (35-49), one line per model+strategy.
    This is Figure 2 of the paper — the key publishable finding.

    Expected result: graph_aug lines drop after step ~42,
    weighted CE lines stay flat — temporal drift finding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Temporal Drift: F1 per Time Step (Test Steps 35–49)",
        fontsize=12, fontweight="bold"
    )

    for ax, model_prefix in zip(axes, ["sage", "gat"]):
        ax.set_title(MODEL_LABELS.get(model_prefix, model_prefix), fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("F1 (Fraud Class)")
        ax.set_ylim(0.3, 1.0)
        ax.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.6,
                   label="Step 42 (drift onset)")

        for key, temporal in temporal_results.items():
            if not key.startswith(model_prefix):
                continue
            strategy = key.split("_", 1)[1]
            steps    = sorted(temporal.keys())
            f1s      = [temporal[t]["f1"] for t in steps]
            color    = STRATEGY_COLORS.get(strategy, "#6B7280")
            label    = STRATEGY_LABELS.get(strategy, strategy)

            ax.plot(steps, f1s, color=color, lw=2,
                    marker="o", markersize=5, label=label)

        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Seed variance error bars (L3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_seed_variance(df: pd.DataFrame, save_path: str = None):
    """
    Bar chart with error bars (±std across 5 seeds).
    Shows that results are statistically robust, not cherry-picked.
    """
    strategies = [s for s in ["baseline","weighted","oversample","graph_aug"]
                  if s in df["strategy"].values]
    models     = [m for m in ["gcn","sage","gat"] if m in df["model"].values]

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    colors = [STRATEGY_COLORS.get(s, "#6B7280") for s in strategies]

    for ax, model in zip(axes, models):
        sub  = df[df["model"] == model].set_index("strategy")
        vals = [sub.loc[s, "f1_mean"]  if s in sub.index else 0 for s in strategies]
        stds = [sub.loc[s, "f1_std"]   if s in sub.index else 0 for s in strategies]

        bars = ax.bar(range(len(strategies)), vals, color=colors, alpha=0.85,
                      yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
        ax.axhline(0.72, color="red", linestyle="--", lw=1.2, alpha=0.7)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies],
                           rotation=15, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.set_ylabel("F1 ± std (5 seeds)")
        ax.set_ylim(0.55, 0.95)

        for bar, v, s in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + s + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("F1 ± Std (5 Seeds) — Statistical Robustness",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Ablation plot (L4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(ablation_results: list, save_path: str = None):
    """
    Horizontal bar chart showing contribution of each L4 component.
    """
    configs = [r["config"] for r in ablation_results]
    f1s     = [r.get("f1_mean", r.get("f1", 0)) for r in ablation_results]
    precs   = [r.get("precision_mean", r.get("precision", 0)) for r in ablation_results]
    recs    = [r.get("recall_mean",    r.get("recall", 0))    for r in ablation_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("L4 Ablation Study: Component Contribution",
                 fontsize=12, fontweight="bold")

    for ax, vals, title in zip(axes,
                               [f1s, precs, recs],
                               ["F1", "Precision", "Recall"]):
        colors = ["#2563EB" if "edgegat" in c and "adaptive" in c
                  else "#059669" if "adaptive" in c
                  else "#D97706" if "edgegat" in c
                  else "#6B7280"
                  for c in configs]
        bars = ax.barh(configs, vals, color=colors, alpha=0.85)
        ax.axvline(0.72, color="red", linestyle="--", lw=1.2, alpha=0.6)
        ax.set(title=title, xlabel="Score", xlim=(0.55, 1.0))
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9)

    legend_handles = [
        mpatches.Patch(color="#6B7280", label="No edge feats, no adaptive"),
        mpatches.Patch(color="#D97706", label="+Edge features"),
        mpatches.Patch(color="#059669", label="+Adaptive switching"),
        mpatches.Patch(color="#2563EB", label="+Both (full L4)"),
    ]
    axes[-1].legend(handles=legend_handles, fontsize=8, loc="lower right")
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig
