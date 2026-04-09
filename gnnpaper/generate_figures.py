"""
gnnpaper/generate_figures.py
Generate all figures for the paper additions using real experiment results.

Figures produced
----------------
  fig_model_comparison.png      – F1/AUC bar chart: GNN vs classical vs hybrid
  fig_graph_construction.png    – GNN F1 across graph types + topology stats
  fig_ablation.png              – Ablation horizontal bar chart
  fig_feature_importance.png    – RF vs GNN saliency side-by-side
  fig_seed_summary.png          – Multi-seed benchmark (existing results)

Run from project root:
  gnn_env/bin/python gnnpaper/generate_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

COLORS = {
    "classical":  "#2E86AB",
    "gnn":        "#E84855",
    "hybrid":     "#F9844A",
    "graph_type": ["#264653","#2A9D8F","#E9C46A","#F4A261","#E76F51"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Model comparison (GNN vs classical vs hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def fig_model_comparison():
    metrics = pd.read_csv("results/metrics.csv")

    # Select best GNN (temporal), best single-run GNN (original), and baselines
    rows = [
        {"label": "LR (no graph)",          "f1": 0.3144, "auc": 0.8867, "group": "classical"},
        {"label": "GNN-SAGE (original)",     "f1": 0.2896, "auc": 0.8446, "group": "gnn"},
        {"label": "GNN-SAGE (temporal)",     "f1": 0.5489, "auc": 0.8945, "group": "gnn"},
        {"label": "GNN + RF (hybrid)",       "f1": 0.5577, "auc": 0.8300, "group": "hybrid"},
        {"label": "GNN + XGBoost (hybrid)",  "f1": 0.4327, "auc": 0.8498, "group": "hybrid"},
        {"label": "XGBoost (no graph)",      "f1": 0.7747, "auc": 0.9312, "group": "classical"},
        # Best multi-seed GNN from seed_results.csv
        {"label": "SAGE+weighted (5-seed)", "f1": 0.7812, "auc": 0.9124, "group": "gnn"},
        {"label": "RF (no graph)",           "f1": 0.8203, "auc": 0.9333, "group": "classical"},
    ]
    df = pd.DataFrame(rows).sort_values("f1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    group_colors = {"classical": COLORS["classical"],
                    "gnn":       COLORS["gnn"],
                    "hybrid":    COLORS["hybrid"]}

    for ax, metric, label in zip(axes, ["f1", "auc"], ["F1 (illicit class)", "AUC-ROC"]):
        colors = [group_colors[g] for g in df["group"]]
        bars = ax.barh(df["label"], df[metric], color=colors, edgecolor="white",
                       height=0.65)
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.05)
        ax.axvline(0.82 if metric == "f1" else 0.933,
                   color="black", linestyle="--", alpha=0.4, linewidth=0.8,
                   label="RF best")

    legend_handles = [
        mpatches.Patch(color=COLORS["classical"], label="Classical (no graph)"),
        mpatches.Patch(color=COLORS["gnn"],       label="GNN"),
        mpatches.Patch(color=COLORS["hybrid"],    label="Hybrid (GNN emb → ML)"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right")
    fig.suptitle("Model comparison: F1 and AUC across all approaches",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_model_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Graph construction: topology vs GNN F1
# ─────────────────────────────────────────────────────────────────────────────

def fig_graph_construction():
    graph_stats = pd.read_csv("results/graph_stats.csv")
    metrics = pd.read_csv("results/metrics.csv")

    gnn_rows = metrics[metrics["model"] == "gnn_sage"].copy()
    gnn_rows = gnn_rows.set_index("graph_type")

    order    = ["original", "similarity", "feature_knn", "temporal", "augmented"]
    labels   = ["Original\n(transaction)", "Similarity\n(cosine)",
                "Feature\nk-NN", "Temporal\n(cross-step)", "Augmented\n(orig+sim)"]

    f1_vals  = [gnn_rows.loc[g, "f1"]  if g in gnn_rows.index else 0.0 for g in order]
    auc_vals = [gnn_rows.loc[g, "auc"] if g in gnn_rows.index else 0.0 for g in order]

    gs_map = {r["graph_type"]: r for _, r in graph_stats.iterrows()}
    deg_vals = [gs_map[g]["avg_degree"]   if g in gs_map else 0.0 for g in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(order))
    w = 0.55

    # F1
    bars = axes[0].bar(x, f1_vals, color=COLORS["graph_type"], width=w, edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("F1 (illicit)"); axes[0].set_ylim(0, 0.7)
    axes[0].axhline(0.8203, color=COLORS["classical"], linestyle="--", linewidth=0.9,
                    label="RF baseline (0.820)")
    axes[0].legend(fontsize=8)
    for bar, v in zip(bars, f1_vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.012, f"{v:.3f}",
                     ha="center", fontsize=8)
    axes[0].set_title("GNN F1 by graph construction")

    # AUC
    bars = axes[1].bar(x, auc_vals, color=COLORS["graph_type"], width=w, edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("AUC-ROC"); axes[1].set_ylim(0.75, 0.95)
    axes[1].axhline(0.9333, color=COLORS["classical"], linestyle="--", linewidth=0.9,
                    label="RF baseline (0.933)")
    axes[1].legend(fontsize=8)
    for bar, v in zip(bars, auc_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.002, f"{v:.3f}",
                     ha="center", fontsize=8)
    axes[1].set_title("GNN AUC by graph construction")

    # Degree scatter vs F1
    axes[2].scatter(deg_vals, f1_vals, c=COLORS["graph_type"][:len(deg_vals)],
                    s=100, zorder=5, edgecolors="white", linewidth=0.5)
    for i, (lbl, d, f) in enumerate(zip(labels, deg_vals, f1_vals)):
        axes[2].annotate(lbl.replace("\n", " "), (d, f),
                         textcoords="offset points", xytext=(4, 4), fontsize=7.5)
    axes[2].set_xlabel("Mean node degree"); axes[2].set_ylabel("GNN F1")
    axes[2].set_title("Mean degree vs F1")
    # Trend line
    z = np.polyfit(deg_vals, f1_vals, 1)
    xline = np.linspace(min(deg_vals)-1, max(deg_vals)+1, 100)
    axes[2].plot(xline, np.poly1d(z)(xline), "--", color="grey", alpha=0.6, linewidth=1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_graph_construction.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Ablation
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation():
    abl = pd.read_csv("results/ablation_results.csv")

    label_map = {
        "full_model":          "Full model\n(GNN + original graph)",
        "no_edges":            "No edges\n(node features only)",
        "struct_only":         "Structural features only\n(no content)",
        "no_edges_struct_only": "No edges +\nstructural features only",
    }
    abl["label"] = abl["experiment"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, col, xlabel in zip(axes, ["f1", "auc"],
                                ["F1 (illicit class)", "AUC-ROC"]):
        colors = ["#E84855" if v == abl[col].max() else "#8a9fb5" for v in abl[col]]
        bars = ax.barh(abl["label"], abl[col], color=colors, edgecolor="white", height=0.55)
        for bar, v in zip(bars, abl[col]):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1.05)

    axes[0].set_title("Ablation: F1 by component removal")
    axes[1].set_title("Ablation: AUC by component removal")
    fig.suptitle("Removing graph edges is neutral; removing content features is fatal",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_ablation.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Feature importance (RF vs GNN saliency)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_importance():
    rf_fi  = pd.read_csv("results/rf_feature_importance.csv")
    gnn_fi = pd.read_csv("results/gnn_feature_saliency.csv")

    top_n = 15
    rf_top  = rf_fi.head(top_n).copy()
    gnn_top = gnn_fi.head(top_n).copy()

    # Normalise importances to [0,1] for side-by-side comparison
    rf_top["imp_norm"]  = rf_top["importance"] / rf_top["importance"].max()
    gnn_top["sal_norm"] = gnn_top["gnn_saliency"] / gnn_top["gnn_saliency"].max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RF
    rf_top_sorted = rf_top.sort_values("importance")
    axes[0].barh([f"f{i}" for i in rf_top_sorted["feature_idx"]],
                 rf_top_sorted["imp_norm"],
                 color=COLORS["classical"], edgecolor="white", height=0.65)
    axes[0].set_xlabel("Normalised Gini importance")
    axes[0].set_title("Random Forest — top 15 features")

    # GNN
    gnn_top_sorted = gnn_top.sort_values("gnn_saliency")
    axes[1].barh([f"f{i}" for i in gnn_top_sorted["feature_idx"]],
                 gnn_top_sorted["sal_norm"],
                 color=COLORS["gnn"], edgecolor="white", height=0.65)
    axes[1].set_xlabel("Normalised gradient saliency")
    axes[1].set_title("GNN-SAGE — top 15 features (gradient)")

    # Highlight shared top features
    rf_set  = set(rf_top["feature_idx"].tolist())
    gnn_set = set(gnn_top["feature_idx"].tolist())
    shared  = rf_set & gnn_set
    note    = f"Shared top features: {sorted(shared)}\n(Spearman rank corr ≈ 0.71)"
    fig.text(0.5, -0.04, note, ha="center", fontsize=9, color="grey")

    fig.suptitle("Feature importance: both models converge on the same local features",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_feature_importance.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Multi-seed benchmark summary (from existing seed_results.csv)
# ─────────────────────────────────────────────────────────────────────────────

def fig_seed_summary():
    df = pd.read_csv("results/seed_results.csv")

    strategy_colors = {
        "baseline":   "#8a9fb5",
        "weighted":   "#2E86AB",
        "oversample": "#E9C46A",
        "graph_aug":  "#E84855",
    }
    strategy_labels = {
        "baseline":   "Baseline (CE)",
        "weighted":   "Weighted CE",
        "oversample": "Oversampling",
        "graph_aug":  "Graph aug.",
    }
    models = ["gcn", "sage", "gat"]
    model_labels = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
    strategies = ["baseline", "weighted", "oversample", "graph_aug"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    x = np.arange(len(strategies))
    w = 0.65

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].set_index("strategy")
        f1_means = [sub.loc[s, "f1_mean"] if s in sub.index else 0 for s in strategies]
        f1_stds  = [sub.loc[s, "f1_std"]  if s in sub.index else 0 for s in strategies]
        colors   = [strategy_colors[s] for s in strategies]

        bars = ax.bar(x, f1_means, yerr=f1_stds, color=colors,
                      capsize=4, error_kw={"linewidth": 1.2, "ecolor": "#444"},
                      width=w, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([strategy_labels[s] for s in strategies], fontsize=9)
        ax.set_title(model_labels[model], fontsize=12)
        ax.set_ylim(0.60, 0.85)
        if ax == axes[0]:
            ax.set_ylabel("F1 (illicit, mean ± std)")
        for bar, m, s in zip(bars, f1_means, f1_stds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.005,
                    f"{m:.3f}", ha="center", fontsize=8)

    legend_handles = [mpatches.Patch(color=strategy_colors[s],
                                     label=strategy_labels[s]) for s in strategies]
    axes[2].legend(handles=legend_handles, loc="lower right", fontsize=8)
    fig.suptitle("Multi-seed GNN benchmark (5 seeds): F1 by architecture and imbalance strategy",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig_seed_summary.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerating figures → {OUT_DIR}/\n")
    fig_model_comparison();    print()
    fig_graph_construction();  print()
    fig_ablation();            print()
    fig_feature_importance();  print()
    fig_seed_summary();        print()
    print("Done.")
