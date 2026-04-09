"""
scripts/addition2_temporal_analysis.py
Addition 2: Per time-step F1 table and plot.

Evaluates saved checkpoints on each test time step (35-49) individually.
Reveals temporal degradation — the key publishable finding.
No retraining needed if checkpoints exist.

Codepath : scripts/addition2_temporal_analysis.py
Runtime  : ~5 min (loads saved checkpoints, no training)
           ~25 min if --retrain flag used
Output   : results/additions/temporal_f1_table.csv
           results/additions/temporal_drift.png
"""

import sys, os, json, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.trainer_minibatch import train
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
CONFIG = dict(lr=1e-3, weight_decay=5e-4, epochs=300, patience=30,
              eval_every=5, batch_size=512)


def eval_per_timestep(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    results = {}
    for t in range(35, 50):
        mask = data.test_mask & (data.time_step == t)
        if mask.sum().item() < 5:
            continue
        preds  = logits[mask].argmax(-1).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
        pb = (preds==1).astype(int); lb = (labels==1).astype(int)
        results[t] = {
            "f1":        round(float(f1_score(lb, pb, zero_division=0)), 4),
            "precision": round(float(precision_score(lb, pb, zero_division=0)), 4),
            "recall":    round(float(recall_score(lb, pb, zero_division=0)), 4),
            "n_illicit": int(lb.sum()),
            "n_total":   int(len(lb)),
        }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain models instead of loading checkpoints")
    args = parser.parse_args()

    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data = preprocess(f, c, e)
    cw   = get_class_weights(data)
    device = torch.device("cpu")

    all_temporal = {}

    for model_name, strategy, label, color in CONFIGS:
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        print(f"\n── {label} ──")

        model = build_model(model_name,
                            in_channels=data.num_node_features,
                            hidden_channels=256, num_layers=3,
                            dropout=0.5, out_channels=3)

        if os.path.exists(ckpt) and not args.retrain:
            print(f"  Loading: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"  Training ({CONFIG['epochs']} epochs)...")
            torch.manual_seed(42)
            aug_data, criterion = apply_strategy(data, strategy, cw)
            result = train(model=model, data=aug_data, criterion=criterion,
                           config=CONFIG, device=device, verbose=True)
            model.load_state_dict(torch.load("best_model.pt", map_location=device))
            torch.save(model.state_dict(), ckpt)
            print(f"  Best F1: {result['best_metrics']['f1']:.4f}")

        model = model.to(device)
        temporal = eval_per_timestep(model, data, device)
        all_temporal[label] = temporal

        print(f"  {'Step':>4} {'F1':>7} {'Prec':>7} {'Recall':>7} "
              f"{'Illicit%':>9}")
        for t in sorted(temporal):
            m = temporal[t]
            pct = m["n_illicit"] / m["n_total"] * 100
            print(f"  {t:>4} {m['f1']:>7.4f} {m['precision']:>7.4f} "
                  f"{m['recall']:>7.4f} {pct:>8.1f}%")

    # Save CSV
    rows = []
    for label, temporal in all_temporal.items():
        for t, m in temporal.items():
            rows.append({"config": label, "timestep": t, **m})
    df = pd.DataFrame(rows)
    df.to_csv("results/additions/temporal_f1_table.csv", index=False)

    # Plot — two subplots: F1 and illicit rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Temporal Drift: Per Time-Step Analysis (Test Steps 35–49)",
                 fontsize=13, fontweight="bold")

    for (model_name, strategy, label, color) in CONFIGS:
        temporal = all_temporal[label]
        steps = sorted(temporal.keys())
        f1s   = [temporal[t]["f1"] for t in steps]
        ax1.plot(steps, f1s, color=color, lw=2.5,
                 marker="o", markersize=6, label=label)

    # Illicit rate per timestep (same for all configs)
    first_label = list(all_temporal.keys())[0]
    first_temp  = all_temporal[first_label]
    steps   = sorted(first_temp.keys())
    ill_pct = [first_temp[t]["n_illicit"]/first_temp[t]["n_total"]*100
               for t in steps]
    ax2.bar(steps, ill_pct, color="#EF4444", alpha=0.7, label="Illicit %")
    ax2.axhline(11.6, color="gray", linestyle="--", lw=1.2,
                label="Train illicit rate (11.6%)")

    ax1.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.6)
    ax2.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.6,
                label="Step 42 (drift onset)")

    ax1.set_ylabel("F1 Score (Fraud Class)")
    ax1.set_ylim(0.3, 1.0)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Illicit Node %")
    ax2.set_ylim(0, 20)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/temporal_drift.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: results/additions/temporal_f1_table.csv")
    print("Saved: results/additions/temporal_drift.png")

    # Key finding summary
    print("\n── KEY FINDING ──")
    for label, temporal in all_temporal.items():
        steps  = sorted(temporal.keys())
        early  = np.mean([temporal[t]["f1"] for t in steps if t <= 39])
        late   = np.mean([temporal[t]["f1"] for t in steps if t >= 44])
        drop   = early - late
        print(f"{label:<22}: early F1={early:.3f}  late F1={late:.3f}  "
              f"drop={drop:.3f} {'← significant drift' if drop>0.05 else ''}")


if __name__ == "__main__":
    main()
