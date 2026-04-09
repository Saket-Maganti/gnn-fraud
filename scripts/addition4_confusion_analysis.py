"""
scripts/addition4_confusion_analysis.py
Addition 4: Confusion matrix + business cost analysis.

Quantifies TP/FP/FN/TN and their business implications.
In financial fraud: FN (missed fraud) >> FP (false alert) in cost.
This contextualises the precision-recall tradeoff as a deliberate choice.

Codepath : scripts/addition4_confusion_analysis.py
Runtime  : ~3 min (loads saved checkpoints)
Output   : results/additions/confusion_analysis.png
           results/additions/business_cost_analysis.csv
"""

import sys, os, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess
from models.gnn import build_model
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("results/additions", exist_ok=True)

CONFIGS = [
    ("sage", "weighted",  "SAGE\n+weighted"),
    ("sage", "graph_aug", "SAGE\n+graph_aug"),
    ("gat",  "weighted",  "GAT\n+weighted"),
    ("gat",  "graph_aug", "GAT\n+graph_aug"),
]

# Business cost assumptions (in arbitrary units, e.g. USD thousands)
COST_FN = 50   # missed fraud: avg loss per undetected fraud transaction
COST_FP = 2    # false alert: analyst review cost per false alarm


def load_predictions(model_name, strategy, data, device):
    ckpt = f"results/{model_name}_{strategy}_model.pt"
    if not os.path.exists(ckpt):
        return None, None
    model = build_model(model_name,
                        in_channels=data.num_node_features,
                        hidden_channels=256, num_layers=3,
                        dropout=0.5, out_channels=3)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    mask   = data.test_mask
    preds  = logits[mask].argmax(-1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    known  = labels != 0
    return preds[known], labels[known]


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    device = torch.device("cpu")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Confusion Matrix & Business Cost Analysis",
                 fontsize=14, fontweight="bold")

    cost_rows = []

    for i, (model_name, strategy, label) in enumerate(CONFIGS):
        preds, labels = load_predictions(model_name, strategy, data, device)
        if preds is None:
            print(f"  Skipping {label.replace(chr(10), ' ')} — no checkpoint")
            continue

        cm = confusion_matrix(labels, preds, labels=[1, 2])
        tn_val = cm[1, 1]; fp_val = cm[1, 0]
        fn_val = cm[0, 1]; tp_val = cm[0, 0]

        # Confusion matrix heatmap
        ax_cm = fig.add_subplot(gs[0, i])
        im = ax_cm.imshow([[tp_val, fn_val], [fp_val, tn_val]],
                           cmap="Blues", aspect="auto")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred Illicit", "Pred Licit"], fontsize=8)
        ax_cm.set_yticklabels(["True Illicit", "True Licit"], fontsize=8)
        ax_cm.set_title(label.replace("\n", " "), fontsize=9, fontweight="bold")

        for (r, c_), val, color_text in [
            ((0,0), tp_val, "white"), ((0,1), fn_val, "black"),
            ((1,0), fp_val, "black"), ((1,1), tn_val, "white")
        ]:
            ax_cm.text(c_, r, str(val), ha="center", va="center",
                       fontsize=11, fontweight="bold", color=color_text)

        # Business cost
        total_cost    = fn_val * COST_FN + fp_val * COST_FP
        fn_cost       = fn_val * COST_FN
        fp_cost       = fp_val * COST_FP
        fraud_caught  = tp_val / (tp_val + fn_val) * 100
        f1            = f1_score((labels==1).astype(int),
                                  (preds==1).astype(int), zero_division=0)

        cost_rows.append({
            "config":       label.replace("\n", " "),
            "TP":           tp_val, "FP": fp_val,
            "FN":           fn_val, "TN": tn_val,
            "F1":           round(f1, 4),
            "fraud_caught%": round(fraud_caught, 1),
            "fn_cost_ku":   fn_cost,
            "fp_cost_ku":   fp_cost,
            "total_cost_ku":total_cost,
        })
        print(f"  {label.replace(chr(10),' ')}: "
              f"TP={tp_val} FP={fp_val} FN={fn_val} TN={tn_val} | "
              f"cost={total_cost:,}k | fraud_caught={fraud_caught:.1f}%")

    # Business cost bar chart
    if cost_rows:
        ax_cost = fig.add_subplot(gs[1, :2])
        df_cost = pd.DataFrame(cost_rows)
        x       = range(len(df_cost))
        width   = 0.35
        ax_cost.bar([i - width/2 for i in x], df_cost["fn_cost_ku"],
                    width, label=f"FN cost (${COST_FN}k each)",
                    color="#EF4444", alpha=0.85)
        ax_cost.bar([i + width/2 for i in x], df_cost["fp_cost_ku"],
                    width, label=f"FP cost (${COST_FP}k each)",
                    color="#F59E0B", alpha=0.85)
        ax_cost.set_xticks(list(x))
        ax_cost.set_xticklabels(df_cost["config"], fontsize=9)
        ax_cost.set_ylabel("Business Cost (arbitrary units)")
        ax_cost.set_title("FN vs FP Business Cost Breakdown\n"
                          f"(FN=${COST_FN}k per missed fraud, "
                          f"FP=${COST_FP}k per false alert)",
                          fontsize=10)
        ax_cost.legend(fontsize=9)
        ax_cost.grid(axis="y", alpha=0.3)
        ax_cost.spines["top"].set_visible(False)
        ax_cost.spines["right"].set_visible(False)

        # Fraud caught % chart
        ax_caught = fig.add_subplot(gs[1, 2:])
        colors_c  = ["#059669" if v >= 90 else
                     "#D97706" if v >= 80 else "#EF4444"
                     for v in df_cost["fraud_caught%"]]
        bars = ax_caught.bar(df_cost["config"], df_cost["fraud_caught%"],
                             color=colors_c, alpha=0.85, edgecolor="white")
        ax_caught.axhline(90, color="green", linestyle="--", lw=1.2,
                          label="90% catch rate", alpha=0.7)
        ax_caught.axhline(80, color="orange", linestyle="--", lw=1.2,
                          label="80% catch rate", alpha=0.7)
        for bar, v in zip(bars, df_cost["fraud_caught%"]):
            ax_caught.text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + 0.5,
                           f"{v:.1f}%", ha="center", va="bottom",
                           fontsize=9, fontweight="bold")
        ax_caught.set_ylabel("Fraud Cases Caught (%)")
        ax_caught.set_title("Fraud Detection Rate per Config", fontsize=10)
        ax_caught.set_ylim(0, 105)
        ax_caught.legend(fontsize=8)
        ax_caught.grid(axis="y", alpha=0.3)
        ax_caught.spines["top"].set_visible(False)
        ax_caught.spines["right"].set_visible(False)

        df_cost.to_csv("results/additions/business_cost_analysis.csv", index=False)

    plt.savefig("results/additions/confusion_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/additions/confusion_analysis.png")
    print("Saved: results/additions/business_cost_analysis.csv")


if __name__ == "__main__":
    main()
