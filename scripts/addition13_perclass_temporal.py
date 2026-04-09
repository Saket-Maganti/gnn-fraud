"""
scripts/addition13_perclass_temporal.py
Per-class precision and recall at each time step.

Shows WHICH metric collapses — recall (model stops predicting fraud)
or precision (model predicts too much fraud).
Answer: recall collapses completely, precision stays reasonable.
This is the more interesting story — the model becomes overly conservative.

Also shows: is the collapse due to the model not predicting fraud at all,
or predicting fraud everywhere?

Codepath : scripts/addition13_perclass_temporal.py
Runtime  : ~3 min (loads checkpoints)
Output   : results/additions/perclass_temporal.png
           results/additions/perclass_temporal.csv
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
]


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    device = torch.device("cpu")

    all_rows = []

    fig, axes = plt.subplots(len(CONFIGS), 3,
                              figsize=(16, 5 * len(CONFIGS)))
    if len(CONFIGS) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Per-Class Temporal Analysis: What Causes the Collapse?\n"
                 "(Precision, Recall, and Predicted Fraud Rate per Time Step)",
                 fontsize=13, fontweight="bold")

    for row_idx, (model_name, strategy, label, color) in enumerate(CONFIGS):
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        if not os.path.exists(ckpt):
            print(f"  Skipping {label}")
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
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(-1).cpu().numpy()

        steps       = []
        precisions  = []
        recalls     = []
        f1s         = []
        pred_rates  = []   # fraction of nodes predicted as fraud
        true_rates  = []   # actual fraud rate

        print(f"  {'Step':>4} {'True%':>7} {'Pred%':>7} "
              f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'-'*52}")

        for t in range(35, 50):
            mask = data.test_mask & (data.time_step == t)
            if mask.sum().item() < 5:
                continue

            labels_t    = data.y[mask].numpy()
            preds_t     = preds[mask.numpy()]
            probs_t     = probs[mask.numpy()]

            labels_bin  = (labels_t == 1).astype(int)
            preds_bin   = (preds_t  == 1).astype(int)

            true_rate   = labels_bin.mean() * 100
            pred_rate   = preds_bin.mean()  * 100

            prec  = precision_score(labels_bin, preds_bin, zero_division=0)
            rec   = recall_score(labels_bin,    preds_bin, zero_division=0)
            f1    = f1_score(labels_bin,        preds_bin, zero_division=0)

            steps.append(t)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            pred_rates.append(pred_rate)
            true_rates.append(true_rate)

            print(f"  {t:>4} {true_rate:>6.1f}% {pred_rate:>6.1f}% "
                  f"{prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

            all_rows.append({
                "config":    label,
                "timestep":  t,
                "true_rate": round(true_rate, 2),
                "pred_rate": round(pred_rate, 2),
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1, 4),
            })

        ax1, ax2, ax3 = axes[row_idx]

        # Plot 1: Precision vs Recall over time
        ax1.plot(steps, precisions, color="#EF4444", lw=2.5,
                 marker="o", markersize=6, label="Precision")
        ax1.plot(steps, recalls, color="#2563EB", lw=2.5,
                 marker="s", markersize=6, label="Recall")
        ax1.plot(steps, f1s, color=color, lw=2, marker="D",
                 markersize=5, linestyle="--", label="F1", alpha=0.7)
        ax1.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.5)
        ax1.set_title(f"{label}\nPrecision vs Recall", fontsize=9)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Plot 2: True vs predicted fraud rate
        ax2.bar([s - 0.2 for s in steps], true_rates,
                0.35, color="#EF4444", alpha=0.7, label="True illicit %")
        ax2.bar([s + 0.2 for s in steps], pred_rates,
                0.35, color="#3B82F6", alpha=0.7, label="Predicted illicit %")
        ax2.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.5)
        ax2.set_title("True vs Predicted Fraud Rate", fontsize=9)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("% Nodes Predicted/True Fraud")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Plot 3: Annotated collapse analysis
        ax3.scatter(true_rates, recalls, c=steps, cmap="RdYlGn_r",
                    s=100, zorder=3)
        for t, tr, rec in zip(steps, true_rates, recalls):
            ax3.annotate(str(t), (tr, rec),
                         textcoords="offset points",
                         xytext=(4, 4), fontsize=7)
        ax3.set_xlabel("True Illicit Rate (%)")
        ax3.set_ylabel("Recall")
        ax3.set_title("Recall vs True Fraud Rate\n"
                      "(colour = time step, red=late)", fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Colour bar
        sm = plt.cm.ScalarMappable(
            cmap="RdYlGn_r",
            norm=plt.Normalize(min(steps), max(steps))
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, label="Time step", shrink=0.8)

    plt.tight_layout()
    plt.savefig("results/additions/perclass_temporal.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(all_rows)
    df.to_csv("results/additions/perclass_temporal.csv", index=False)

    # Key finding
    print("\n── KEY FINDING ──")
    for config in df["config"].unique():
        sub = df[df["config"] == config]
        early = sub[sub["timestep"] <= 42]
        late  = sub[sub["timestep"] >= 43]
        print(f"\n{config}:")
        print(f"  Early steps (35-42): "
              f"Prec={early['precision'].mean():.3f}  "
              f"Recall={early['recall'].mean():.3f}")
        print(f"  Late  steps (43-49): "
              f"Prec={late['precision'].mean():.3f}  "
              f"Recall={late['recall'].mean():.3f}")
        print(f"  Collapse type: "
              f"{'RECALL collapses' if late['recall'].mean() < 0.1 else 'mixed'}")
        print(f"  Pred fraud rate drops: "
              f"{early['pred_rate'].mean():.1f}% → "
              f"{late['pred_rate'].mean():.1f}%")

    print("\nSaved: results/additions/perclass_temporal.png")
    print("Saved: results/additions/perclass_temporal.csv")


if __name__ == "__main__":
    main()