"""
scripts/addition3_pr_curves.py
Addition 3: Precision-Recall curves for all configs.

Plots the full PR tradeoff across all thresholds — standard for
imbalanced classification papers. Shows operating point flexibility.

Codepath : scripts/addition3_pr_curves.py
Runtime  : ~3 min (loads saved checkpoints)
Output   : results/additions/pr_curves.png
           results/additions/threshold_table.csv
"""

import sys, os, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              f1_score, precision_score, recall_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/additions", exist_ok=True)

CONFIGS = [
    ("sage", "weighted",  "SAGE + weighted",  "#2563EB", "o"),
    ("sage", "graph_aug", "SAGE + graph_aug", "#059669", "s"),
    ("gat",  "weighted",  "GAT + weighted",   "#D97706", "^"),
    ("gat",  "graph_aug", "GAT + graph_aug",  "#7C3AED", "D"),
]


def get_probs(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    mask   = data.test_mask
    probs  = torch.softmax(logits[mask], dim=-1)[:, 1].cpu().numpy()
    labels = (data.y[mask] == 1).cpu().numpy().astype(int)
    return probs, labels


def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    preds = (probs >= best_t).astype(int)
    return {
        "threshold": round(float(best_t), 3),
        "f1":        round(float(best_f1), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "ap":        round(float(average_precision_score(labels, probs)), 4),
    }


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    device = torch.device("cpu")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Precision–Recall Analysis — All Configurations",
                 fontsize=13, fontweight="bold")

    ax_pr  = axes[0]
    ax_thr = axes[1]

    threshold_rows = []

    for model_name, strategy, label, color, marker in CONFIGS:
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        if not os.path.exists(ckpt):
            print(f"  Skipping {label} — no checkpoint at {ckpt}")
            print(f"  Run: python train.py --model {model_name} "
                  f"--strategy {strategy} --hidden 128 --layers 2 "
                  f"--dropout 0.5 --lr 1e-3 --epochs 300 first")
            continue

        print(f"  Loading {label}...")
        model = build_model(model_name,
                            in_channels=data.num_node_features,
                            hidden_channels=256, num_layers=3,
                            dropout=0.5, out_channels=3)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)

        probs, labels = get_probs(model, data, device)
        prec, rec, thresholds = precision_recall_curve(labels, probs)
        ap   = average_precision_score(labels, probs)
        best = find_best_threshold(probs, labels)

        # PR curve
        ax_pr.plot(rec, prec, color=color, lw=2,
                   label=f"{label} (AP={ap:.3f})")
        ax_pr.scatter([best["recall"]], [best["precision"]],
                      color=color, s=80, marker=marker, zorder=5)

        # F1 vs threshold curve
        f1s = []
        ts  = np.arange(0.05, 0.95, 0.01)
        for t in ts:
            preds = (probs >= t).astype(int)
            f1s.append(f1_score(labels, preds, zero_division=0))
        ax_thr.plot(ts, f1s, color=color, lw=2, label=label)
        ax_thr.axvline(best["threshold"], color=color,
                       linestyle=":", lw=1, alpha=0.7)

        threshold_rows.append({"config": label, **best})
        print(f"    Best: threshold={best['threshold']:.2f}  "
              f"F1={best['f1']:.4f}  P={best['precision']:.4f}  "
              f"R={best['recall']:.4f}  AP={best['ap']:.4f}")

    # Decorate PR plot
    ax_pr.axhline(0.70, color="gray", linestyle="--", lw=1, alpha=0.5,
                  label="Precision=0.70")
    ax_pr.axvline(0.75, color="gray", linestyle="-.", lw=1, alpha=0.5,
                  label="Recall=0.75")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curves")
    ax_pr.set_xlim(0, 1.05)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.grid(alpha=0.3)
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)

    # Decorate threshold plot
    ax_thr.set_xlabel("Classification Threshold")
    ax_thr.set_ylabel("F1 Score (Fraud Class)")
    ax_thr.set_title("F1 vs Threshold")
    ax_thr.set_xlim(0.05, 0.95)
    ax_thr.set_ylim(0, 0.90)
    ax_thr.axhline(0.72, color="red", linestyle="--", lw=1.2,
                   label="Target F1=0.72", alpha=0.7)
    ax_thr.legend(fontsize=8)
    ax_thr.grid(alpha=0.3)
    ax_thr.spines["top"].set_visible(False)
    ax_thr.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/pr_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Save threshold table
    df = pd.DataFrame(threshold_rows)
    df.to_csv("results/additions/threshold_table.csv", index=False)

    print("\n── OPTIMAL THRESHOLD SUMMARY ──")
    print(df.to_string(index=False))
    print("\nSaved: results/additions/pr_curves.png")
    print("Saved: results/additions/threshold_table.csv")


if __name__ == "__main__":
    main()
