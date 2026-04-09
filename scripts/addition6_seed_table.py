"""
scripts/addition6_seed_table.py
Generates the proper 3-seed main results table.

Runs ALL configs (MLP, GCN, SAGE, GAT) × ALL strategies
(baseline, weighted, graph_aug) × 3 seeds.
Outputs a publication-ready table with mean ± std.

Codepath : scripts/addition6_seed_table.py
Runtime  : ~45 min total
           MLP × 3 strategies × 3 seeds × ~2.5 min = ~22 min
           GCN × 3 strategies × 3 seeds × ~2.5 min = ~22 min
           (SAGE/GAT already have checkpoints — skipped or fast)
Output   : results/additions/main_results_table.csv
           results/additions/main_results_table.png
"""

import sys, os, json, time, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.mlp_baseline import MLP
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.trainer_minibatch import train
from utils.metrics import aggregate_seed_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/additions", exist_ok=True)

CONFIG = dict(lr=1e-3, weight_decay=5e-4, epochs=300, patience=30,
              eval_every=5, batch_size=512)
SEEDS = [42, 123, 456]

CONFIGS = [
    # (model_name, strategy, display_name, is_mlp, hidden, layers)
    ("mlp",  "baseline",  "MLP",        True,  128, 3),
    ("mlp",  "weighted",  "MLP",        True,  128, 3),
    ("gcn",  "baseline",  "GCN",        False, 256, 3),
    ("gcn",  "weighted",  "GCN",        False, 256, 3),
    ("sage", "baseline",  "GraphSAGE",  False, 256, 3),
    ("sage", "weighted",  "GraphSAGE",  False, 256, 3),
    ("sage", "graph_aug", "GraphSAGE",  False, 256, 3),
    ("gat",  "baseline",  "GAT",        False, 256, 3),
    ("gat",  "weighted",  "GAT",        False, 256, 3),
    ("gat",  "graph_aug", "GAT",        False, 256, 3),
]

STRATEGY_LABELS = {
    "baseline":  "Baseline CE",
    "weighted":  "Weighted CE",
    "graph_aug": "Graph Aug.",
}


def run_seeds(model_name, strategy, display_name, is_mlp,
              hidden, layers, data, cw):
    cache_path = (f"results/additions/cache_"
                  f"{model_name}_{strategy}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded from cache: {display_name} + {strategy}")
        return cached

    all_metrics = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        aug_data, criterion = apply_strategy(data, strategy, cw)
        if is_mlp:
            model = MLP(in_channels=data.num_node_features,
                        hidden_channels=hidden, out_channels=3,
                        num_layers=layers, dropout=0.5)
        else:
            model = build_model(model_name,
                                in_channels=data.num_node_features,
                                hidden_channels=hidden,
                                num_layers=layers,
                                dropout=0.5, out_channels=3)
        t0 = time.time()
        result = train(model=model, data=aug_data,
                       criterion=criterion, config=CONFIG,
                       device=torch.device("cpu"), verbose=False)
        m = result["best_metrics"]
        elapsed = (time.time() - t0) / 60
        print(f"    seed {seed}: F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} "
              f"({elapsed:.1f} min)")
        all_metrics.append(m)

    agg = aggregate_seed_metrics(all_metrics)
    with open(cache_path, "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data = preprocess(f, c, e)
    cw   = get_class_weights(data)

    rows = []
    for model_name, strategy, display, is_mlp, hidden, layers in CONFIGS:
        print(f"\n── {display} + {STRATEGY_LABELS[strategy]} ──")
        agg = run_seeds(model_name, strategy, display,
                        is_mlp, hidden, layers, data, cw)
        rows.append({
            "Model":     display,
            "Strategy":  STRATEGY_LABELS[strategy],
            "F1":        agg["f1"]["mean"],
            "F1_std":    agg["f1"]["std"],
            "Precision": agg["precision"]["mean"],
            "P_std":     agg["precision"]["std"],
            "Recall":    agg["recall"]["mean"],
            "R_std":     agg["recall"]["std"],
            "AUC":       agg["auc"]["mean"],
            "AUC_std":   agg["auc"]["std"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/additions/main_results_table.csv", index=False)

    # Print table
    print("\n" + "="*78)
    print(f"{'Model':<12} {'Strategy':<14} "
          f"{'F1':>10} {'Precision':>12} {'Recall':>10} {'AUC':>10}")
    print("-"*78)
    for _, row in df.iterrows():
        print(f"{row['Model']:<12} {row['Strategy']:<14} "
              f"{row['F1']:>6.4f}±{row['F1_std']:.3f}  "
              f"{row['Precision']:>6.4f}±{row['P_std']:.3f}  "
              f"{row['Recall']:>6.4f}±{row['R_std']:.3f}  "
              f"{row['AUC']:>6.4f}±{row['AUC_std']:.3f}")

    # Publication-quality plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.suptitle(
        "Main Results: F1, Precision, Recall (mean ± std, 3 seeds)",
        fontsize=13, fontweight="bold"
    )

    MODEL_COLORS = {
        "MLP":       "#6B7280",
        "GCN":       "#94A3B8",
        "GraphSAGE": "#2563EB",
        "GAT":       "#D97706",
    }
    STRAT_HATCHES = {
        "Baseline CE": "",
        "Weighted CE": "//",
        "Graph Aug.":  "xx",
    }

    for ax, (metric, std_col, target, title) in zip(axes, [
        ("F1",        "F1_std",  0.72, "F1 Score"),
        ("Precision", "P_std",   0.70, "Precision"),
        ("Recall",    "R_std",   0.75, "Recall"),
    ]):
        x     = range(len(df))
        colors  = [MODEL_COLORS[m] for m in df["Model"]]
        hatches = [STRAT_HATCHES[s] for s in df["Strategy"]]

        for i, (_, row) in enumerate(df.iterrows()):
            ax.bar(i, row[metric], yerr=row[std_col],
                   color=MODEL_COLORS[row["Model"]],
                   hatch=STRAT_HATCHES[row["Strategy"]],
                   alpha=0.82, capsize=4,
                   error_kw={"elinewidth": 1.2},
                   edgecolor="white")
            ax.text(i, row[metric] + row[std_col] + 0.008,
                    f"{row[metric]:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, fontweight="bold",
                    rotation=90)

        ax.axhline(target, color="red", linestyle="--",
                   lw=1.2, alpha=0.7, label=f"Target ({target})")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{r['Model']}\n{r['Strategy']}" for _, r in df.iterrows()],
            fontsize=7, rotation=30, ha="right"
        )
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.3, 1.02)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend for model colors
    legend_patches = [
        mpatches.Patch(color=c, label=m)
        for m, c in MODEL_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9,
               title="Model", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("results/additions/main_results_table.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: results/additions/main_results_table.csv")
    print("Saved: results/additions/main_results_table.png")


if __name__ == "__main__":
    main()