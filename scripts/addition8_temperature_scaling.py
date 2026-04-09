"""
scripts/addition8_temperature_scaling.py
Temperature scaling calibration.

Learns a single scalar T that divides logits before softmax:
    p(y|x) = softmax(logits / T)

T > 1 = soften predictions (lower confidence)
T < 1 = sharpen predictions (higher confidence)

Unlike threshold tuning, temperature scaling:
- Is a principled probabilistic calibration method (Guo et al. 2017)
- Preserves the ranking of predictions (only changes confidence)
- Can be fitted on a small validation set
- Works even when calibration set has few samples

We fit T on time steps 30-34 (validation period) and evaluate on 35-49.
This is the correct way to handle temporal calibration.

Codepath : scripts/addition8_temperature_scaling.py
Runtime  : ~5 min (loads checkpoints, fits T, evaluates)
Output   : results/additions/temperature_scaling.csv
           results/additions/temperature_scaling.png
"""

import sys, os, torch
import torch.nn as nn
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


class TemperatureScaler(nn.Module):
    """Single scalar temperature applied to logits before softmax."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.01)

    def fit(self, logits, labels, mask, lr=0.01, max_iter=100):
        """Fit temperature on validation set using NLL loss."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr,
                                       max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits[mask])
            loss   = nll(scaled, labels[mask])
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


def get_metrics_at_threshold(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
    }


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    device = torch.device("cpu")

    # Validation mask: steps 30-34 (labeled, from training period)
    labeled  = data.y != 0
    val_mask = labeled & (data.time_step >= 30) & (data.time_step <= 34)
    print(f"Val nodes (steps 30-34): {val_mask.sum().item():,}")

    all_rows     = []
    summary_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle("Temperature Scaling Calibration\n"
                 "(fitted on steps 30-34, evaluated on steps 35-49)",
                 fontsize=13, fontweight="bold")

    for ax, (model_name, strategy, label, color) in zip(
            axes.flat, CONFIGS):
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        if not os.path.exists(ckpt):
            print(f"  Skipping {label} — no checkpoint")
            ax.set_visible(False)
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

        # Fit temperature on val set
        scaler = TemperatureScaler()
        val_labels = data_dev.y.clone()
        T = scaler.fit(logits.detach(), val_labels, val_mask)
        print(f"  Fitted temperature T = {T:.4f}")

        # Scale logits
        with torch.no_grad():
            scaled_logits = scaler(logits.detach())

        # Evaluate per time step
        steps_f1_fixed  = []
        steps_f1_scaled = []
        steps_list      = []

        print(f"  {'Step':>4} {'Illicit%':>9} {'T=1.0 F1':>10} "
              f"{'T={:.2f} F1'.format(T):>12} {'Delta':>8}")
        print(f"  {'-'*52}")

        for t in range(35, 50):
            mask_t = data.test_mask & (data.time_step == t)
            if mask_t.sum().item() < 5:
                continue

            labels_t = (data.y[mask_t] == 1).numpy().astype(int)
            ill_pct  = labels_t.mean() * 100

            # Original probabilities
            probs_orig   = torch.softmax(
                logits[mask_t], dim=-1)[:, 1].detach().numpy()
            # Scaled probabilities
            probs_scaled = torch.softmax(
                scaled_logits[mask_t], dim=-1)[:, 1].detach().numpy()

            m_orig   = get_metrics_at_threshold(probs_orig,   labels_t)
            m_scaled = get_metrics_at_threshold(probs_scaled, labels_t)
            delta    = m_scaled["f1"] - m_orig["f1"]

            print(f"  {t:>4} {ill_pct:>8.1f}% "
                  f"{m_orig['f1']:>10.4f} "
                  f"{m_scaled['f1']:>12.4f} "
                  f"{delta:>+8.4f}")

            steps_list.append(t)
            steps_f1_fixed.append(m_orig["f1"])
            steps_f1_scaled.append(m_scaled["f1"])

            all_rows.append({
                "config":       label,
                "timestep":     t,
                "illicit_pct":  round(ill_pct, 2),
                "temperature":  round(T, 4),
                "f1_original":  m_orig["f1"],
                "f1_scaled":    m_scaled["f1"],
                "prec_scaled":  m_scaled["precision"],
                "rec_scaled":   m_scaled["recall"],
                "delta":        round(delta, 4),
            })

        mean_orig   = np.mean(steps_f1_fixed)
        mean_scaled = np.mean(steps_f1_scaled)
        improvement = mean_scaled - mean_orig
        print(f"\n  Mean F1 original : {mean_orig:.4f}")
        print(f"  Mean F1 scaled   : {mean_scaled:.4f}  "
              f"({improvement:+.4f})")

        summary_rows.append({
            "config":        label,
            "temperature":   round(T, 4),
            "mean_f1_orig":  round(mean_orig, 4),
            "mean_f1_scaled":round(mean_scaled, 4),
            "improvement":   round(improvement, 4),
        })

        # Plot
        ax.plot(steps_list, steps_f1_fixed, color="gray", lw=2,
                marker="o", markersize=5, linestyle="--",
                label=f"Original (T=1.0)")
        ax.plot(steps_list, steps_f1_scaled, color=color, lw=2.5,
                marker="s", markersize=6,
                label=f"Temperature scaled (T={T:.2f})")
        ax.fill_between(steps_list,
                        steps_f1_fixed, steps_f1_scaled,
                        where=[s > f for s, f in
                               zip(steps_f1_scaled, steps_f1_fixed)],
                        color=color, alpha=0.2, label="Improvement")
        ax.fill_between(steps_list,
                        steps_f1_fixed, steps_f1_scaled,
                        where=[s < f for s, f in
                               zip(steps_f1_scaled, steps_f1_fixed)],
                        color="red", alpha=0.1, label="Degradation")
        ax.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.5)
        ax.set_title(f"{label}\nT={T:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/temperature_scaling.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summary_rows)
    df.to_csv("results/additions/temperature_scaling.csv", index=False)

    print("\n" + "="*60)
    print("TEMPERATURE SCALING SUMMARY")
    print("="*60)
    print(df_sum.to_string(index=False))
    print("\nSaved: results/additions/temperature_scaling.csv")
    print("Saved: results/additions/temperature_scaling.png")


if __name__ == "__main__":
    main()