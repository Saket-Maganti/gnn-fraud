"""
experiments/temporal_deep.py
Publication-quality temporal analysis.

Extends the existing temporal_results.json with:
  1. Per-timestep F1 / Precision / Recall curves (3-panel figure)
  2. Illicit transaction ratio vs time (overlaid as second y-axis)
  3. Identifies and annotates the temporal drift onset point
  4. Saves temporal_deep.csv with full per-step breakdown

Requires:
  - results/temporal_results.json  (from experiments/temporal_analysis.py)
  - Elliptic dataset CSVs in data/raw/

Usage:
    python experiments/temporal_deep.py
    python experiments/temporal_deep.py --no_plot   # CSV only
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_illicit_rate(data) -> dict:
    """Returns {timestep: illicit_fraction} for all 49 steps."""
    y  = data.y.numpy()
    ts = data.time_step.numpy()
    rates = {}
    for t in range(1, 50):
        mask    = (ts == t) & (y != 0)    # labeled nodes in step t
        if mask.sum() == 0:
            rates[t] = float("nan")
            continue
        illicit = (y[mask] == 1).sum()
        rates[t] = float(illicit) / float(mask.sum())
    return rates


def load_temporal_json(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # Convert string keys back to int
    return {
        model_key: {int(step): metrics
                    for step, metrics in step_dict.items()}
        for model_key, step_dict in raw.items()
    }


def find_drift_onset(f1_series: dict, threshold: float = 0.05) -> int:
    """
    Find first step where F1 drops >threshold below the mean of the first 3 steps.
    Returns the step number or -1 if no clear drift detected.
    """
    steps  = sorted(f1_series)
    if len(steps) < 4:
        return -1
    early_mean = np.mean([f1_series[s] for s in steps[:3]])
    for s in steps[3:]:
        if f1_series[s] < early_mean - threshold:
            return s
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# CSV builder
# ─────────────────────────────────────────────────────────────────────────────

def build_csv(temporal: dict, illicit_rates: dict, results_dir: str) -> pd.DataFrame:
    rows = []
    for model_key, step_metrics in temporal.items():
        for t in range(35, 50):
            if t not in step_metrics:
                continue
            m = step_metrics[t]
            rows.append({
                "model":        model_key,
                "timestep":     t,
                "f1":           m["f1"],
                "precision":    m["precision"],
                "recall":       m["recall"],
                "auc":          m.get("auc", float("nan")),
                "accuracy":     m.get("accuracy", float("nan")),
                "illicit_rate": illicit_rates.get(t, float("nan")),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, "temporal_deep.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

MODEL_STYLES = {
    "sage_baseline":  {"color": "#4e79a7", "linestyle": "-",  "label": "SAGE-Baseline",  "lw": 1.5},
    "sage_weighted":  {"color": "#f28e2b", "linestyle": "-",  "label": "SAGE-Weighted",  "lw": 2.2},
    "sage_graph_aug": {"color": "#e15759", "linestyle": "--", "label": "SAGE-GraphAug",  "lw": 2.2},
    "gat_baseline":   {"color": "#76b7b2", "linestyle": ":",  "label": "GAT-Baseline",   "lw": 1.5},
    "gat_weighted":   {"color": "#59a14f", "linestyle": ":",  "label": "GAT-Weighted",   "lw": 2.0},
    "gat_graph_aug":  {"color": "#edc948", "linestyle": "-.", "label": "GAT-GraphAug",   "lw": 2.0},
}

METRIC_LABELS = {"f1": "F1 Score", "precision": "Precision", "recall": "Recall"}


def plot_temporal_deep(temporal: dict, illicit_rates: dict, save_path: str):
    """
    3-panel figure:
      Panel 1: F1 per timestep (all models) + illicit rate (right y-axis)
      Panel 2: Precision per timestep
      Panel 3: Recall per timestep
    """
    test_steps = list(range(35, 50))

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(3, 1, hspace=0.42)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    ill_steps  = [t for t in test_steps if not np.isnan(illicit_rates.get(t, float("nan")))]
    ill_vals   = [illicit_rates[t] * 100 for t in ill_steps]  # as %

    metrics = ["f1", "precision", "recall"]

    for panel_idx, metric in enumerate(metrics):
        ax = axes[panel_idx]

        for model_key, step_metrics in temporal.items():
            style = MODEL_STYLES.get(model_key, {"color": "gray", "linestyle": "-",
                                                  "label": model_key, "lw": 1.5})
            ys = [step_metrics.get(t, {}).get(metric, float("nan")) for t in test_steps]
            ax.plot(test_steps, ys,
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["lw"], label=style["label"],
                    marker="o", markersize=4, alpha=0.88)

        # Drift annotation on F1 panel (panel 0)
        if panel_idx == 0 and temporal:
            # Use the best (sage_weighted / sage_graph_aug) series for drift detection
            ref_key = "sage_weighted" if "sage_weighted" in temporal else list(temporal.keys())[0]
            f1_series = {t: temporal[ref_key].get(t, {}).get("f1", float("nan"))
                         for t in test_steps}
            onset = find_drift_onset(f1_series)
            if onset > 0:
                ax.axvline(onset, color="#b07aa1", linewidth=1.4, linestyle=":",
                           alpha=0.85, label=f"Drift onset (step {onset})")
                ax.annotate(f"Drift onset\n(step {onset})",
                            xy=(onset, 0.05), xycoords=("data", "axes fraction"),
                            fontsize=8, color="#b07aa1", ha="left",
                            xytext=(onset + 0.3, 0.12))

        # Illicit rate as twin axis (only on F1 panel)
        if panel_idx == 0 and ill_vals:
            ax2 = ax.twinx()
            ax2.fill_between(ill_steps, ill_vals, alpha=0.12, color="#bab0ac", label="Illicit rate (%)")
            ax2.plot(ill_steps, ill_vals, color="#bab0ac", linewidth=1.2, linestyle="--")
            ax2.set_ylabel("Illicit rate (%)", fontsize=9, color="#888")
            ax2.tick_params(axis="y", colors="#888", labelsize=8)
            ax2.set_ylim(0, max(ill_vals) * 3.5)

        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(34.5, 49.5)
        ax.set_xticks(test_steps)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.axvspan(42.5, 49.5, alpha=0.06, color="#e15759",
                   label="Late drift zone (steps 43–49)" if panel_idx == 0 else None)

        if panel_idx == 0:
            ax.legend(fontsize=7.5, ncol=2, loc="upper right",
                      framealpha=0.85, edgecolor="#ccc")

    axes[-1].set_xlabel("Test Timestep", fontsize=10)
    fig.suptitle("Per-Timestep Performance vs. Temporal Distribution Shift\n"
                 "(Elliptic Bitcoin Dataset, Test Steps 35–49)", fontsize=12, y=0.99)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(temporal: dict, illicit_rates: dict):
    print(f"\n{'Step':>5}  {'Illicit%':>8}  " +
          "  ".join(f"{k:>14}" for k in temporal))
    print("-" * (20 + 16 * len(temporal)))
    for t in range(35, 50):
        ill = illicit_rates.get(t, float("nan"))
        ill_str = f"{ill*100:6.1f}%" if not np.isnan(ill) else "   N/A"
        cols = []
        for model_key, sm in temporal.items():
            m = sm.get(t, {})
            if m:
                cols.append(f"F1={m['f1']:.3f} P={m['precision']:.3f}")
            else:
                cols.append("       N/A      ")
        print(f"{t:>5}  {ill_str:>8}  " + "  ".join(f"{c:>14}" for c in cols))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "temporal_results.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found.")
        print("Run experiments/temporal_analysis.py first.")
        sys.exit(1)

    print("Loading temporal results from JSON …")
    temporal = load_temporal_json(json_path)
    print(f"  Models: {list(temporal.keys())}")

    print("Computing illicit rates from dataset …")
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges)
    illicit_rates = compute_illicit_rate(data)

    print("\n" + "=" * 60)
    print("TEMPORAL DEEP ANALYSIS")
    print("=" * 60)
    print_summary(temporal, illicit_rates)

    print("\nBuilding CSV …")
    build_csv(temporal, illicit_rates, results_dir)

    if not args.no_plot:
        print("Generating figure …")
        plot_temporal_deep(
            temporal, illicit_rates,
            save_path=os.path.join(results_dir, "temporal_deep.png"),
        )

    # Print drift onset per model
    print("\nDrift onset (F1 drop >5pp below first-3-step mean):")
    for model_key, step_metrics in temporal.items():
        f1s = {t: step_metrics.get(t, {}).get("f1", float("nan")) for t in range(35, 50)}
        onset = find_drift_onset(f1s)
        status = f"step {onset}" if onset > 0 else "no clear drift"
        print(f"  {model_key:<20}: {status}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip figure generation (CSV only)")
    main(parser.parse_args())
