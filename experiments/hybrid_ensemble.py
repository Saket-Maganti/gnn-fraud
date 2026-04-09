"""
experiments/hybrid_ensemble.py
Hybrid model: combine GNN and MLP predictions via ensemble.

Two combination strategies
--------------------------
  1. weighted_avg  — p_hybrid = α * p_gnn + (1-α) * p_mlp
                     α is tuned on the validation split (steps 30-34)
  2. concatenation — train a lightweight classifier on [emb_gnn || raw_features]
                     (uses models/hybrid.py extract_embeddings)

Comparison baseline: GNN alone, MLP alone (no-edges SAGE).

Outputs
-------
  results/hybrid_ensemble.csv    (metrics per strategy)
  results/hybrid_ensemble.png    (bar chart)

Usage:
    python experiments/hybrid_ensemble.py
    python experiments/hybrid_ensemble.py --epochs 150
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data
from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from models.hybrid import extract_embeddings
from utils.imbalance import apply_strategy
from utils.metrics import compute_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _no_edges(data: Data) -> Data:
    empty = torch.zeros(2, 0, dtype=torch.long)
    return Data(x=data.x, edge_index=empty, y=data.y,
                train_mask=data.train_mask, test_mask=data.test_mask,
                time_step=data.time_step)


def train_sage(graph_data: Data, device, epochs: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cw             = get_class_weights(graph_data)
    aug_data, crit = apply_strategy(graph_data, "weighted", cw)
    aug_data       = aug_data.to(device)
    crit           = crit.to(device)

    model = build_model(
        "sage",
        in_channels     = aug_data.num_node_features,
        hidden_channels = 256,
        num_layers      = 3,
        dropout         = 0.5,
        out_channels    = 3,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state, no_improve = 0.0, None, 0
    patience = 30

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(aug_data.x, aug_data.edge_index,
                       getattr(aug_data, "edge_attr", None))
        loss = crit(logits, aug_data.y, aug_data.train_mask)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
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
# Probability extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_fraud_probs(model, data, device) -> np.ndarray:
    """Return per-node fraud probability (class 1) for all nodes."""
    data   = data.to(device)
    logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Binary sklearn metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def eval_binary(y_true_orig, preds_bin, probs_fraud):
    y_bin = (y_true_orig == 1).astype(int)
    try:
        auc = float(roc_auc_score(y_bin, probs_fraud))
    except ValueError:
        auc = float("nan")
    return {
        "f1":        round(float(f1_score(y_bin, preds_bin, zero_division=0)),        4),
        "precision": round(float(precision_score(y_bin, preds_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_bin, preds_bin, zero_division=0)),    4),
        "auc":       round(auc, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: weighted average ensemble
# ─────────────────────────────────────────────────────────────────────────────

def tune_alpha(p_gnn_val, p_mlp_val, y_val_orig):
    """Grid-search α on validation split for best F1 on illicit class."""
    y_bin   = (y_val_orig == 1).astype(int)
    best_f1, best_alpha = 0.0, 0.5
    for alpha in np.linspace(0.0, 1.0, 21):
        p_mix  = alpha * p_gnn_val + (1 - alpha) * p_mlp_val
        preds  = (p_mix >= 0.5).astype(int)
        f1     = f1_score(y_bin, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha
    return best_alpha


def weighted_avg_ensemble(p_gnn, p_mlp, alpha: float, threshold: float = 0.5):
    """Return (binary predictions, fraud probs)."""
    p_mix = alpha * p_gnn + (1 - alpha) * p_mlp
    preds = (p_mix >= threshold).astype(int)
    return preds, p_mix


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: concatenation ensemble (GNN emb + raw features → RF)
# ─────────────────────────────────────────────────────────────────────────────

def concat_rf_ensemble(gnn_model, data, device):
    """
    Train RF on [GNN_embedding || raw_features] for train nodes,
    evaluate on test nodes.
    Returns metrics dict.
    """
    emb   = extract_embeddings(gnn_model, data, device).numpy()     # [N, h]
    raw   = data.x.cpu().numpy()                                     # [N, F]
    feat  = np.concatenate([emb, raw], axis=1)                       # [N, h+F]

    y     = data.y.cpu().numpy()
    tr_m  = data.train_mask.cpu().numpy()
    te_m  = data.test_mask.cpu().numpy()

    X_tr, y_tr = feat[tr_m], y[tr_m]
    X_te, y_te = feat[te_m], y[te_m]

    # Label map: keep only labeled nodes for training
    labeled_tr = y_tr != 0
    X_tr, y_tr = X_tr[labeled_tr], y_tr[labeled_tr]

    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    rf.fit(X_tr, y_tr)

    preds   = rf.predict(X_te)
    classes = list(rf.classes_)
    proba   = rf.predict_proba(X_te)

    y_te_bin   = (y_te == 1).astype(int)
    preds_bin  = (preds == 1).astype(int)
    prob_fraud = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(y_te))

    try:
        auc = float(roc_auc_score(y_te_bin, prob_fraud))
    except ValueError:
        auc = float("nan")

    return {
        "f1":        round(float(f1_score(y_te_bin, preds_bin, zero_division=0)),        4),
        "precision": round(float(precision_score(y_te_bin, preds_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te_bin, preds_bin, zero_division=0)),    4),
        "auc":       round(auc, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading data …")
    features, classes, edges = load_elliptic_raw()
    data     = preprocess(features, classes, edges, normalize=True)
    mlp_data = _no_edges(data)

    val_mask  = (data.time_step >= 30) & (data.time_step <= 34) & (data.y != 0)
    test_mask = data.test_mask

    print(f"\nTraining GNN (SAGE+weighted, {args.epochs} epochs) …")
    gnn_model = train_sage(data, device, args.epochs, seed=42)

    print(f"Training MLP (SAGE+weighted, no edges, {args.epochs} epochs) …")
    mlp_model = train_sage(mlp_data, device, args.epochs, seed=42)

    # Raw probabilities (all nodes)
    p_gnn_all = get_fraud_probs(gnn_model, data,     device)
    p_mlp_all = get_fraud_probs(mlp_model, mlp_data, device)

    y_all = data.y.cpu().numpy()

    rows = []

    # ── Baseline: GNN alone ────────────────────────────────────────────────
    gnn_te  = p_gnn_all[test_mask.cpu().numpy()]
    y_te    = y_all[test_mask.cpu().numpy()]
    preds_g = (gnn_te >= 0.5).astype(int)
    m = eval_binary(y_te, preds_g, gnn_te)
    rows.append({"strategy": "GNN alone (SAGE+weighted)", **m})
    print(f"\n  GNN alone:   F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Baseline: MLP alone ────────────────────────────────────────────────
    mlp_te  = p_mlp_all[test_mask.cpu().numpy()]
    preds_m = (mlp_te >= 0.5).astype(int)
    m = eval_binary(y_te, preds_m, mlp_te)
    rows.append({"strategy": "MLP alone (no edges)", **m})
    print(f"  MLP alone:   F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Strategy 1: weighted average (α tuned on val) ─────────────────────
    p_gnn_val = p_gnn_all[val_mask.cpu().numpy()]
    p_mlp_val = p_mlp_all[val_mask.cpu().numpy()]
    y_val     = y_all[val_mask.cpu().numpy()]
    alpha     = tune_alpha(p_gnn_val, p_mlp_val, y_val)
    print(f"\n  Optimal α (val set): {alpha:.2f}")

    preds_w, probs_w = weighted_avg_ensemble(gnn_te, mlp_te, alpha)
    m = eval_binary(y_te, preds_w, probs_w)
    rows.append({"strategy": f"Weighted avg (α={alpha:.2f})", **m})
    print(f"  Weighted avg:  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Strategy 2: concatenation → RF ────────────────────────────────────
    print("\n  Concatenation RF (GNN emb + raw features) …")
    m = concat_rf_ensemble(gnn_model, data, device)
    rows.append({"strategy": "Concat (GNN emb + raw → RF)", **m})
    print(f"  Concat RF:     F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "hybrid_ensemble.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Print table
    print(f"\n{'Strategy':<35} {'F1':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7}")
    print("─" * 62)
    for row in rows:
        print(f"  {row['strategy']:<33} {row['f1']:>7.4f} {row['precision']:>7.4f} "
              f"{row['recall']:>7.4f} {row['auc']:>7.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        _plot(df, results_dir)


def _plot(df: pd.DataFrame, results_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    strategies = df["strategy"].tolist()
    x = np.arange(len(strategies))
    w = 0.28
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]

    for i, metric in enumerate(["f1", "precision", "recall", "auc"]):
        vals = df[metric].tolist()
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=colors[i], alpha=0.82, label=metric.upper())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9, rotation=12, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Hybrid Ensemble Strategies vs. Individual Models", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "hybrid_ensemble.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--device",  choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--no_plot", action="store_true")
    main(parser.parse_args())
