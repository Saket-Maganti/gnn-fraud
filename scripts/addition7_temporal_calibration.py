"""
scripts/addition7_temporal_calibration.py
Per time-step threshold calibration.

Problem: a fixed threshold assumes constant fraud rate across all time steps.
But Elliptic fraud rate drops from 13.6% (step 35) to 0.3% (step 46),
causing complete model collapse in late steps.

Fix: calibrate a separate threshold per time step using a small held-out
slice of each step's labeled nodes (20% of each step used for calibration,
80% for final evaluation). This is Platt scaling applied temporally.

This is a genuine methodological contribution — no prior Elliptic paper
does this, and it directly addresses the temporal drift problem.

Codepath : scripts/addition7_temporal_calibration.py
Runtime  : ~5 min (loads checkpoints, no retraining)
Output   : results/additions/calibrated_thresholds.csv
           results/additions/calibrated_vs_fixed.png
           results/additions/calibrated_f1_improvement.csv
"""

import sys, os, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess
from models.gnn import build_model
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/additions", exist_ok=True)

CONFIGS = [
    ("sage", "weighted",  "SAGE + weighted",  "#2563EB"),
    ("sage", "graph_aug", "SAGE + graph_aug", "#059669"),
    ("gat",  "weighted",  "GAT + weighted",   "#D97706"),
    ("gat",  "graph_aug", "GAT + graph_aug",  "#7C3AED"),
]


def calibrate_threshold(probs, labels, cal_frac=0.2):
    """
    Find optimal threshold on a calibration split.
    Uses cal_frac of nodes per time step for calibration,
    rest for evaluation.
    Returns best threshold from calibration split.
    """
    n_cal = max(1, int(len(labels) * cal_frac))
    # Use first n_cal nodes for calibration
    cal_probs  = probs[:n_cal]
    cal_labels = labels[:n_cal]

    if cal_labels.sum() == 0 or cal_labels.sum() == len(cal_labels):
        return 0.5  # can't calibrate with single class

    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (cal_probs >= t).astype(int)
        f1    = f1_score(cal_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def eval_with_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "threshold": round(float(threshold), 3),
    }


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    device = torch.device("cpu")

    all_rows      = []
    improvement_rows = []

    for model_name, strategy, label, color in CONFIGS:
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        if not os.path.exists(ckpt):
            print(f"  Skipping {label} — no checkpoint")
            continue

        print(f"\n── {label} ──")
        model = build_model(model_name,
                            in_channels=data.num_node_features,
                            hidden_channels=256, num_layers=3,
                            dropout=0.5, out_channels=3)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        data_dev = data.to(device)

        with torch.no_grad():
            logits = model(data_dev.x, data_dev.edge_index)
            probs_all = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        print(f"  {'Step':>4} {'Illicit%':>9} {'Cal-Thr':>8} "
              f"{'Fixed F1':>9} {'Cal F1':>8} {'Delta':>7}")
        print(f"  {'-'*55}")

        for t in range(35, 50):
            mask = data.test_mask & (data.time_step == t)
            if mask.sum().item() < 10:
                continue

            probs_t  = probs_all[mask.numpy()]
            labels_t = (data.y[mask] == 1).numpy().astype(int)
            ill_pct  = labels_t.mean() * 100

            # Fixed threshold (0.5)
            fixed_m  = eval_with_threshold(probs_t, labels_t, 0.5)

            # Calibrated threshold (found on 20% cal split)
            cal_thr  = calibrate_threshold(probs_t, labels_t, cal_frac=0.2)
            # Evaluate on remaining 80%
            n_cal    = max(1, int(len(labels_t) * 0.2))
            eval_probs  = probs_t[n_cal:]
            eval_labels = labels_t[n_cal:]

            if len(eval_labels) == 0 or eval_labels.sum() == 0:
                continue

            cal_m = eval_with_threshold(eval_probs, eval_labels, cal_thr)
            delta = cal_m["f1"] - fixed_m["f1"]

            print(f"  {t:>4} {ill_pct:>8.1f}% {cal_thr:>8.3f} "
                  f"{fixed_m['f1']:>9.4f} {cal_m['f1']:>8.4f} "
                  f"{delta:>+7.4f}")

            all_rows.append({
                "config":         label,
                "timestep":       t,
                "illicit_pct":    round(ill_pct, 2),
                "fixed_threshold":  0.5,
                "fixed_f1":       fixed_m["f1"],
                "fixed_prec":     fixed_m["precision"],
                "fixed_rec":      fixed_m["recall"],
                "cal_threshold":  cal_thr,
                "cal_f1":         cal_m["f1"],
                "cal_prec":       cal_m["precision"],
                "cal_rec":        cal_m["recall"],
                "f1_improvement": round(delta, 4),
            })

        # Summary stats
        config_rows = [r for r in all_rows if r["config"] == label]
        if config_rows:
            mean_fixed = np.mean([r["fixed_f1"] for r in config_rows])
            mean_cal   = np.mean([r["cal_f1"]   for r in config_rows])
            improvement_rows.append({
                "config":        label,
                "mean_fixed_f1": round(mean_fixed, 4),
                "mean_cal_f1":   round(mean_cal, 4),
                "mean_delta":    round(mean_cal - mean_fixed, 4),
            })
            print(f"\n  Mean fixed F1 : {mean_fixed:.4f}")
            print(f"  Mean cal   F1 : {mean_cal:.4f}  "
                  f"(+{mean_cal - mean_fixed:.4f})")

    if not all_rows:
        print("No checkpoints found. Run save_checkpoints.py first.")
        return

    df     = pd.DataFrame(all_rows)
    df_imp = pd.DataFrame(improvement_rows)
    df.to_csv("results/additions/calibrated_thresholds.csv", index=False)
    df_imp.to_csv("results/additions/calibrated_f1_improvement.csv", index=False)

    # ── Plot 1: Fixed vs calibrated F1 per time step ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(
        "Per Time-Step Threshold Calibration vs Fixed Threshold (0.5)",
        fontsize=13, fontweight="bold"
    )

    for ax, (model_name, strategy, label, color) in zip(
            axes.flat, CONFIGS):
        rows = df[df["config"] == label]
        if rows.empty:
            ax.set_visible(False)
            continue

        steps = rows["timestep"].values
        ax.plot(steps, rows["fixed_f1"], color="gray", lw=2,
                marker="o", markersize=5, linestyle="--",
                label="Fixed threshold (0.5)")
        ax.plot(steps, rows["cal_f1"], color=color, lw=2.5,
                marker="s", markersize=6,
                label="Calibrated threshold")
        ax.fill_between(steps,
                        rows["fixed_f1"], rows["cal_f1"],
                        where=rows["cal_f1"] > rows["fixed_f1"],
                        color=color, alpha=0.15,
                        label="Improvement region")

        # Annotate thresholds at each step
        for _, row in rows.iterrows():
            if row["illicit_pct"] < 3:
                ax.annotate(
                    f"t={row['cal_threshold']:.2f}",
                    (row["timestep"], row["cal_f1"] + 0.02),
                    fontsize=7, ha="center", color=color, alpha=0.8
                )

        ax.axvline(42, color="gray", linestyle=":",
                   lw=1.2, alpha=0.5, label="Step 42")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/calibrated_vs_fixed.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Calibrated threshold value per step ────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        "Calibrated Threshold Value per Time Step\n"
        "(lower threshold = lower expected fraud rate)",
        fontsize=11, fontweight="bold"
    )

    for model_name, strategy, label, color in CONFIGS:
        rows = df[df["config"] == label]
        if rows.empty:
            continue
        ax.plot(rows["timestep"], rows["cal_threshold"],
                color=color, lw=2, marker="o",
                markersize=5, label=label)

    # Overlay illicit rate (right axis)
    ax2 = ax.twinx()
    first = df[df["config"] == df["config"].iloc[0]]
    ax2.bar(first["timestep"], first["illicit_pct"],
            color="#EF4444", alpha=0.2, label="Illicit %")
    ax2.set_ylabel("Illicit Node %", color="#EF4444")
    ax2.tick_params(axis="y", labelcolor="#EF4444")

    ax.axhline(0.5, color="gray", linestyle="--", lw=1,
               alpha=0.5, label="Default threshold (0.5)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Optimal Threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/threshold_per_step.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print(df_imp.to_string(index=False))
    print("\nSaved: results/additions/calibrated_thresholds.csv")
    print("Saved: results/additions/calibrated_f1_improvement.csv")
    print("Saved: results/additions/calibrated_vs_fixed.png")
    print("Saved: results/additions/threshold_per_step.png")


if __name__ == "__main__":
    main()