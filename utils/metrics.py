"""
utils/metrics.py
Evaluation metrics for fraud detection.
All metrics are computed on the fraud (illicit=1) class only.
"""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, Optional
from torch_geometric.data import Data


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask:   torch.Tensor,
) -> Dict[str, float]:
    """
    Compute F1, precision, recall, accuracy on masked nodes.
    Positive class = illicit (label=1).
    """
    preds  = logits[mask].argmax(dim=-1).cpu().numpy()
    y_true = labels[mask].cpu().numpy()

    preds_bin  = (preds  == 1).astype(int)
    y_true_bin = (y_true == 1).astype(int)

    f1        = f1_score(y_true_bin,        preds_bin, zero_division=0)
    precision = precision_score(y_true_bin, preds_bin, zero_division=0)
    recall    = recall_score(y_true_bin,    preds_bin, zero_division=0)
    acc       = (preds == y_true).mean()

    # AUC-ROC on fraud probability
    probs = torch.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
    try:
        auc = roc_auc_score(y_true_bin, probs)
    except ValueError:
        auc = float("nan")

    return {
        "f1":        round(float(f1),        4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "accuracy":  round(float(acc),       4),
        "auc":       round(float(auc),       4),
    }


def aggregate_seed_metrics(seed_results: list) -> Dict[str, Dict[str, float]]:
    """
    Given a list of metric dicts (one per seed), compute mean ± std.
    Returns { metric_name: {"mean": x, "std": y} }
    """
    keys = seed_results[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in seed_results if not np.isnan(r[k])]
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
        }
    return agg


def format_metrics_table(agg: Dict) -> str:
    """Pretty-print aggregated metrics table."""
    lines = [f"{'Metric':<12} {'Mean':>8} {'±Std':>8}"]
    lines.append("-" * 30)
    for k, v in agg.items():
        lines.append(f"{k:<12} {v['mean']:>8.4f} {v['std']:>8.4f}")
    return "\n".join(lines)


@torch.no_grad()
def full_report(model, data: Data, device: torch.device,
                model_type: str = "static"):
    """Print a detailed test-set classification report."""
    model.eval()
    data = data.to(device)

    if model_type == "evolve":
        # Build snapshots for EvolveGCN
        from utils.temporal import build_snapshots
        snapshots = build_snapshots(data, time_range=range(35, 50))
        logits = model(snapshots)
        mask   = data.test_mask
    else:
        logits = model(data.x, data.edge_index,
                       getattr(data, "edge_attr", None))
        mask   = data.test_mask

    mask   = mask.to(device)
    preds  = logits[mask].argmax(-1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    print("\n── CLASSIFICATION REPORT (test set) ──")
    print(classification_report(
        y_true, preds,
        labels=[1, 2],
        target_names=["Illicit (fraud)", "Licit (legit)"],
        digits=4,
    ))
    cm = confusion_matrix(y_true, preds, labels=[1, 2])
    tn, fp, fn, tp = cm.ravel()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Fraud Precision : {tp/(tp+fp+1e-8):.4f}")
    print(f"Fraud Recall    : {tp/(tp+fn+1e-8):.4f}")
