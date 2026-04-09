"""
experiments/calibration_study.py
Calibration analysis for GNN fraud detection models.

Computes
--------
  1. Expected Calibration Error (ECE) before and after temperature scaling
  2. Reliability diagrams (calibration curves) per model
  3. Brier score (overall probabilistic accuracy)

Temperature scaling (Guo et al. 2017): learn scalar T on a held-out
validation split (timesteps 30-34) that divides logits before softmax.
Preserves ranking; only adjusts confidence levels.

Outputs
-------
  results/calibration_results.csv    (ECE, Brier, optimal T per model)
  results/calibration_curves.png     (reliability diagrams, 2×2 grid)

Usage:
    python experiments/calibration_study.py
    python experiments/calibration_study.py --n_bins 15 --epochs 200
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.metrics import compute_metrics


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(device_arg)


# ─────────────────────────────────────────────────────────────────────────────
# ECE + Brier score
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    probs:  predicted probability for the positive (illicit) class [N]
    labels: binary ground truth (1=illicit) [N]
    """
    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    ece         = 0.0
    N           = len(probs)
    bin_data    = []

    for i in range(n_bins):
        lo, hi   = bins[i], bins[i + 1]
        in_bin   = (probs >= lo) & (probs < hi)
        if in_bin.sum() == 0:
            bin_data.append((0.5 * (lo + hi), 0.0, 0.0, 0))
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc  = labels[in_bin].mean()
        n_b      = in_bin.sum()
        ece     += (n_b / N) * abs(avg_conf - avg_acc)
        bin_data.append((avg_conf, avg_acc, n_b / N, n_b))

    return float(ece), bin_data


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T.clamp(min=0.05)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50) -> float:
        """Fit T on validation set using NLL loss with LBFGS."""
        self.to(logits.device)   # ensure T is on the same device as logits
        self.train()
        optimizer = torch.optim.LBFGS([self.T], lr=lr, max_iter=max_iter)

        logits_val = logits[mask].detach()
        labels_val = labels[mask].detach()

        def closure():
            optimizer.zero_grad()
            scaled = logits_val / self.T.clamp(min=0.05)
            loss   = F.cross_entropy(scaled, labels_val)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()
        return float(self.T.item())


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(data, model_name: str, strategy: str, device: torch.device,
                epochs: int = 200, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cw             = get_class_weights(data)
    aug_data, crit = apply_strategy(data, strategy, cw)
    aug_data       = aug_data.to(device)
    crit           = crit.to(device)

    model = build_model(
        model_name,
        in_channels     = aug_data.num_node_features,
        hidden_channels = 256,
        num_layers      = 3,
        dropout         = 0.5,
        out_channels    = 3,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1, best_state, no_improve = 0.0, None, 0
    patience = 30

    for ep in range(1, epochs + 1):
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(aug_data.x, aug_data.edge_index,
                           getattr(aug_data, "edge_attr", None))
            loss = crit(logits, aug_data.y, aug_data.train_mask)
        opt.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if hasattr(crit, "step_epoch"):
            crit.step_epoch()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                le = model(aug_data.x, aug_data.edge_index,
                           getattr(aug_data, "edge_attr", None))
            f1 = compute_metrics(le, aug_data.y, aug_data.test_mask)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Extract probabilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs_labels(model, data, mask, device, scaler=None):
    data   = data.to(device)
    logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    if scaler is not None:
        logits = scaler(logits)
    probs  = torch.softmax(logits[mask], dim=-1)[:, 1].cpu().numpy()
    labels = (data.y[mask] == 1).cpu().numpy().astype(int)
    return probs, labels


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability(bin_data_before, bin_data_after, ece_before, ece_after,
                     title: str, ax):
    """Draw a reliability diagram on ax (before and after scaling)."""
    confs_b = [b[0] for b in bin_data_before]
    accs_b  = [b[1] for b in bin_data_before]
    confs_a = [b[0] for b in bin_data_after]
    accs_a  = [b[1] for b in bin_data_after]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration", alpha=0.5)
    ax.bar(confs_b, accs_b, width=0.09, alpha=0.4, color="#4e79a7",
           label=f"Before scaling  (ECE={ece_before:.4f})", align="center")
    ax.plot(confs_a, accs_a, "o-", color="#e15759", linewidth=2.0, markersize=5,
            label=f"After scaling   (ECE={ece_after:.4f})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction positives", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("sage",  "weighted"),
    ("sage",  "graph_aug"),
    ("gat",   "weighted"),
    ("gat",   "graph_aug"),
]


def main(args):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print("Loading data …")
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)

    # Validation mask: steps 30-34 (labeled), for fitting temperature
    val_mask = (
        (data.time_step >= 30) & (data.time_step <= 34) & (data.y != 0)
    )
    test_mask = data.test_mask

    print(f"  Val nodes (steps 30-34): {val_mask.sum().item():,}")
    print(f"  Test nodes (steps 35-49): {test_mask.sum().item():,}")

    rows         = []
    fig, axes    = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat    = axes.flatten()

    for idx, (model_name, strategy) in enumerate(EXPERIMENTS):
        label = f"{model_name}_{strategy}"
        print(f"\n[{idx+1}/{len(EXPERIMENTS)}] {label} …")

        # Train
        model = train_model(data, model_name, strategy, device,
                            epochs=args.epochs, seed=42)

        # Uncalibrated probs (test set)
        p_before, y_true = get_probs_labels(model, data, test_mask, device)

        # Fit temperature scaler on validation set
        data_dev  = data.to(device)
        with torch.no_grad():
            logits_all = model(data_dev.x, data_dev.edge_index,
                               getattr(data_dev, "edge_attr", None))
        scaler = TemperatureScaler()
        T_opt  = scaler.fit(logits_all, data_dev.y, val_mask.to(device))
        print(f"  Optimal T = {T_opt:.4f}")

        # Calibrated probs
        p_after, _ = get_probs_labels(model, data, test_mask, device, scaler=scaler)

        # ECE
        ece_before, bin_before = compute_ece(p_before, y_true, n_bins=args.n_bins)
        ece_after,  bin_after  = compute_ece(p_after,  y_true, n_bins=args.n_bins)

        # Brier
        brier_before = brier_score(p_before, y_true)
        brier_after  = brier_score(p_after,  y_true)

        print(f"  ECE   before={ece_before:.4f}  after={ece_after:.4f}  "
              f"Δ={ece_after-ece_before:+.4f}")
        print(f"  Brier before={brier_before:.4f}  after={brier_after:.4f}")

        rows.append({
            "model":          label,
            "T_optimal":      round(T_opt, 4),
            "ece_before":     round(ece_before,    4),
            "ece_after":      round(ece_after,     4),
            "ece_reduction":  round(ece_before - ece_after, 4),
            "brier_before":   round(brier_before,  4),
            "brier_after":    round(brier_after,   4),
            "brier_reduction":round(brier_before - brier_after, 4),
        })

        # Reliability diagram
        plot_reliability(
            bin_before, bin_after, ece_before, ece_after,
            title=label.replace("_", " ").title(),
            ax=axes_flat[idx],
        )

    # ── Save CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "calibration_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # ── Print table ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"{'Model':<22} {'T':>6} {'ECE↓':>8} {'ECE↑':>8} {'ΔECE':>8} {'ΔBrier':>8}")
    print("─" * 65)
    for row in rows:
        print(f"  {row['model']:<20} {row['T_optimal']:>6.3f} "
              f"{row['ece_before']:>8.4f} {row['ece_after']:>8.4f} "
              f"{row['ece_reduction']:>+8.4f} {row['brier_reduction']:>+8.4f}")

    # ── Save figure ────────────────────────────────────────────────────────
    fig.suptitle("Reliability Diagrams: Calibration Before/After Temperature Scaling",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out_fig = os.path.join(results_dir, "calibration_curves.png")
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    main(parser.parse_args())
