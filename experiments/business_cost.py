"""
experiments/business_cost.py
Business cost evaluation with asymmetric FP/FN penalties.

Motivation
----------
In fraud detection, a False Negative (missed fraud) is far more costly
than a False Positive (flagging a legitimate transaction).

Cost model
----------
  C_FN = cost of missing one illicit transaction   (default: 10)
  C_FP = cost of flagging one licit transaction    (default: 1)

  Total cost = C_FN * FN + C_FP * FP

The cost ratio C_FN / C_FP represents the relative business severity.
We sweep over multiple ratios to show model rankings across settings.

Inputs
------
  results/metrics.csv      — overall test metrics
  results/seed_results.csv — multi-seed F1/Precision/Recall per model

Outputs
-------
  results/business_cost.csv       (cost matrix, absolute and normalised)
  results/business_cost.png       (figure: cost vs cost-ratio sweep)

Usage:
    python experiments/business_cost.py
    python experiments/business_cost.py --c_fn 20 --c_fp 1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Cost helpers
# ─────────────────────────────────────────────────────────────────────────────

def tp_fp_fn_from_metrics(precision: float, recall: float, n_illicit: int):
    """
    Reconstruct (approximate) TP, FP, FN from precision and recall.
    TP = recall * n_illicit
    FN = n_illicit - TP
    FP = TP * (1 - precision) / precision   [from precision = TP / (TP + FP)]
    """
    TP = recall * n_illicit
    FN = n_illicit - TP
    if precision > 0:
        FP = TP * (1.0 - precision) / precision
    else:
        FP = 0.0
    return TP, FP, FN


def total_cost(precision: float, recall: float,
               n_illicit: int, c_fn: float, c_fp: float) -> float:
    _, FP, FN = tp_fp_fn_from_metrics(precision, recall, n_illicit)
    return c_fn * FN + c_fp * FP


def normalised_cost(cost: float, n_illicit: int, c_fn: float) -> float:
    """Cost relative to the worst-case (catching nothing, all FN)."""
    worst = c_fn * n_illicit
    return cost / max(worst, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Build model table from available CSVs
# ─────────────────────────────────────────────────────────────────────────────

def build_model_table(results_dir: str) -> pd.DataFrame:
    """
    Combine metrics.csv and seed_results.csv into a unified table
    with model, precision, recall, f1, source columns.
    """
    rows = []

    # From metrics.csv (per-run GNN + baseline results)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        for _, r in df.iterrows():
            rows.append({
                "model":     r["model"],
                "variant":   r.get("graph_type", "—"),
                "f1":        r["f1"],
                "precision": r["precision"],
                "recall":    r["recall"],
                "source":    "metrics.csv",
            })

    # From seed_results.csv (mean values across 5 seeds)
    seed_path = os.path.join(results_dir, "seed_results.csv")
    if os.path.exists(seed_path):
        df = pd.read_csv(seed_path)
        for _, r in df.iterrows():
            rows.append({
                "model":     f"{r['model']}_{r['strategy']}",
                "variant":   "multi-seed",
                "f1":        r["f1_mean"],
                "precision": r["precision_mean"],
                "recall":    r["recall_mean"],
                "source":    "seed_results.csv",
            })

    return pd.DataFrame(rows).drop_duplicates(subset=["model", "variant"])


# ─────────────────────────────────────────────────────────────────────────────
# Representative model selection (de-duplicate for clean table)
# ─────────────────────────────────────────────────────────────────────────────

REPRESENTATIVE_MODELS = [
    # (display_label, model_key, variant_key, source)
    ("Logistic Regression",     "logistic_regression",  "none",         "metrics.csv"),
    ("Random Forest",           "random_forest",        "none",         "metrics.csv"),
    ("XGBoost",                 "xgboost",              "none",         "metrics.csv"),
    ("GNN+RF (Hybrid)",         "gnn+rf",               "original",     "metrics.csv"),
    ("GCN (weighted)",          "gcn_weighted",         "multi-seed",   "seed_results.csv"),
    ("GraphSAGE (weighted)",    "sage_weighted",        "multi-seed",   "seed_results.csv"),
    ("GraphSAGE (graph_aug)",   "sage_graph_aug",       "multi-seed",   "seed_results.csv"),
    ("GAT (weighted)",          "gat_weighted",         "multi-seed",   "seed_results.csv"),
    ("GAT (graph_aug)",         "gat_graph_aug",        "multi-seed",   "seed_results.csv"),
]

# Approximate number of illicit test nodes (Elliptic dataset, steps 35-49)
# From dataset: ~2,700 illicit in test split (steps 35-49, labeled)
N_ILLICIT_TEST = 2_700


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print("Loading model metrics …")
    all_models = build_model_table(results_dir)

    # Build lookup: (model, variant, source) → {precision, recall, f1}
    def lookup(model_key, variant, source):
        mask = (
            (all_models["model"]   == model_key) &
            (all_models["variant"] == variant)   &
            (all_models["source"]  == source)
        )
        rows = all_models[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    # ── Default cost analysis (C_FN / C_FP = args.c_fn / args.c_fp) ──────
    print(f"\nCost model: C_FN={args.c_fn}  C_FP={args.c_fp}  "
          f"(ratio {args.c_fn/args.c_fp:.0f}:1)")
    print(f"N illicit test nodes (estimate): {N_ILLICIT_TEST:,}")

    cost_rows = []
    for display_label, model_key, variant, source in REPRESENTATIVE_MODELS:
        row = lookup(model_key, variant, source)
        if row is None:
            continue
        p, r, f1 = row["precision"], row["recall"], row["f1"]
        _, FP, FN = tp_fp_fn_from_metrics(p, r, N_ILLICIT_TEST)
        cost = total_cost(p, r, N_ILLICIT_TEST, args.c_fn, args.c_fp)
        norm = normalised_cost(cost, N_ILLICIT_TEST, args.c_fn)
        cost_rows.append({
            "model":      display_label,
            "f1":         round(f1,   4),
            "precision":  round(p,    4),
            "recall":     round(r,    4),
            "FP_est":     round(FP,   0),
            "FN_est":     round(FN,   0),
            "cost":       round(cost, 1),
            "cost_norm":  round(norm, 4),
            "c_fn":       args.c_fn,
            "c_fp":       args.c_fp,
        })

    df_cost = pd.DataFrame(cost_rows)

    if df_cost.empty:
        print("ERROR: No matching model data found in metrics.csv / seed_results.csv.")
        print("       Run experiments/baselines.py and experiments/run_seeds.py first.")
        sys.exit(1)

    df_cost = df_cost.sort_values("cost")
    out_csv = os.path.join(results_dir, "business_cost.csv")
    df_cost.to_csv(out_csv, index=False)

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'Model':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'FP':>7} {'FN':>7} {'Cost':>9} {'Norm':>6}")
    print("─" * 76)
    for _, row in df_cost.iterrows():
        print(f"  {row['model']:<23} {row['f1']:>6.4f} {row['precision']:>6.4f} "
              f"{row['recall']:>6.4f} {row['FP_est']:>7.0f} {row['FN_est']:>7.0f} "
              f"{row['cost']:>9,.0f} {row['cost_norm']:>6.4f}")

    # ── Cost-ratio sweep ───────────────────────────────────────────────────
    sweep_ratios = [1, 2, 5, 10, 20, 50, 100]
    sweep_rows   = []
    for display_label, model_key, variant, source in REPRESENTATIVE_MODELS:
        row = lookup(model_key, variant, source)
        if row is None:
            continue
        p, r = row["precision"], row["recall"]
        for ratio in sweep_ratios:
            cost = total_cost(p, r, N_ILLICIT_TEST, c_fn=ratio, c_fp=1.0)
            norm = normalised_cost(cost, N_ILLICIT_TEST, c_fn=ratio)
            sweep_rows.append({
                "model":     display_label,
                "fn_fp_ratio": ratio,
                "cost":      cost,
                "cost_norm": norm,
            })
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(os.path.join(results_dir, "business_cost_sweep.csv"), index=False)

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot(df_cost, df_sweep, sweep_ratios, results_dir)

    print(f"\nSaved → {out_csv}")
    print(f"        {results_dir}/business_cost_sweep.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
]


def _plot(df_cost: pd.DataFrame, df_sweep: pd.DataFrame,
          sweep_ratios: list, results_dir: str):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: bar chart at default cost ratio ──────────────────────────────
    ax = axes[0]
    models = df_cost["model"].tolist()
    costs  = df_cost["cost"].tolist()
    colors = MODEL_COLORS[:len(models)]
    bars   = ax.barh(models, costs, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Total Cost  (C_FN × FN + C_FP × FP)", fontsize=10)
    ax.set_title(f"Business Cost per Model\n"
                 f"(C_FN={df_cost['c_fn'].iloc[0]}, C_FP={df_cost['c_fp'].iloc[0]})",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # best (lowest cost) at top

    # Annotate with normalised cost
    for bar, row in zip(bars, df_cost.itertuples()):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{row.cost_norm:.3f}", va="center", fontsize=8)

    # ── Right: cost vs. FN/FP ratio sweep ─────────────────────────────────
    ax2 = axes[1]
    model_names = df_sweep["model"].unique()

    for i, model_name in enumerate(model_names):
        sub    = df_sweep[df_sweep["model"] == model_name]
        ratios = sub["fn_fp_ratio"].tolist()
        norms  = sub["cost_norm"].tolist()
        ax2.plot(ratios, norms,
                 color=MODEL_COLORS[i % len(MODEL_COLORS)],
                 linewidth=2.0, marker="o", markersize=4,
                 label=model_name)

    ax2.axvline(10, color="gray", linestyle=":", linewidth=1.2, alpha=0.6,
                label="C_FN/C_FP = 10 (default)")
    ax2.set_xscale("log")
    ax2.set_xlabel("FN/FP Cost Ratio  (C_FN / C_FP)", fontsize=10)
    ax2.set_ylabel("Normalised Cost  (0 = perfect, 1 = worst)", fontsize=10)
    ax2.set_title("Cost vs. Cost Ratio Sweep\n(lower = better)", fontsize=11)
    ax2.legend(fontsize=7.5, loc="upper left", ncol=1,
               framealpha=0.85, edgecolor="#ccc")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(min(sweep_ratios) * 0.8, max(sweep_ratios) * 1.2)

    fig.suptitle("Business Cost Evaluation: Asymmetric FP/FN Penalties",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(results_dir, "business_cost.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_fn", type=float, default=10.0,
                        help="Cost of a False Negative (missed fraud)")
    parser.add_argument("--c_fp", type=float, default=1.0,
                        help="Cost of a False Positive (flagged licit tx)")
    main(parser.parse_args())
