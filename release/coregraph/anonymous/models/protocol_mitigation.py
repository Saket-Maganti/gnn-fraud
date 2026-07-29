"""
Model-agnostic protocol mitigation utilities.

Training-free helpers for threshold selection, review-budget evaluation, and
prior-shift adjustment. Consumed by ``scripts/run_protocol_mitigation_study.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


Number = Union[int, float]


def _to_numpy(arr) -> np.ndarray:
    return np.asarray(arr)


def _binary_labels(y_true, positive_label: int = 1) -> np.ndarray:
    y = _to_numpy(y_true).reshape(-1)
    return (y == positive_label).astype(int)


def select_threshold_on_validation(
    y_val,
    scores_val,
    metric: str = "f1",
    *,
    positive_label: int = 1,
    thresholds: Sequence[float] | None = None,
) -> float:
    """Pick a score threshold maximizing *metric* on validation labels."""
    y = _binary_labels(y_val, positive_label)
    scores = _to_numpy(scores_val).reshape(-1)
    if y.shape[0] != scores.shape[0]:
        raise ValueError("y_val and scores_val length mismatch")
    if y.shape[0] == 0:
        raise ValueError("Validation set is empty")

    grid = (
        list(thresholds)
        if thresholds is not None
        else [i / 100 for i in range(1, 100)]
    )
    best_t, best_score = 0.5, float("-inf")
    for t in grid:
        preds = (scores >= t).astype(int)
        if metric in {"f1", "val_f1"}:
            score = f1_score(y, preds, zero_division=0)
        elif metric in {"auprc", "val_auprc"}:
            score = f1_score(y, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def evaluate_threshold_policy(
    y_true,
    scores,
    threshold: float,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    """Apply a fixed score threshold and return fraud-detection metrics."""
    y = _binary_labels(y_true, positive_label)
    s = _to_numpy(scores).reshape(-1)
    preds = (s >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }


def top_k_review_predictions(scores, k: Number) -> np.ndarray:
    """Return indices of the top-*k* scored items (higher = more suspicious)."""
    s = _to_numpy(scores).reshape(-1)
    n = s.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    if isinstance(k, float) and 0.0 < k < 1.0:
        k_eff = max(1, int(np.ceil(k * n)))
    else:
        k_eff = min(n, max(0, int(k)))
    order = np.lexsort((np.arange(n), -s))
    return order[:k_eff]


def evaluate_review_budget(
    y_true,
    scores,
    ks: Iterable[Number],
    *,
    positive_label: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Precision/recall at fixed review budgets."""
    y = _binary_labels(y_true, positive_label)
    s = _to_numpy(scores).reshape(-1)
    n_pos = int(y.sum())
    out: Dict[str, Dict[str, float]] = {}
    for k in ks:
        key = str(k)
        top = top_k_review_predictions(s, k)
        if top.size == 0:
            out[key] = {"precision": 0.0, "recall": 0.0, "k_eff": 0}
            continue
        tp = int(y[top].sum())
        out[key] = {
            "precision": float(tp / top.size),
            "recall": float(tp / max(n_pos, 1)),
            "k_eff": int(top.size),
        }
    return out


def apply_prior_shift_threshold(
    base_threshold: float,
    train_prior: float,
    val_prior: float,
    test_prior: float | None = None,
) -> float:
    """Adjust a probability/score threshold under a simple prior shift.

    Uses log-odds correction:
        logit(t') = logit(t) + log(p_target / (1-p_target)) - log(p_ref / (1-p_ref))
    where *p_ref* is ``val_prior`` and *p_target* is ``test_prior`` when given,
    otherwise ``val_prior`` relative to ``train_prior``.
    """
    eps = 1e-6

    def _clamp(p: float) -> float:
        return float(np.clip(p, eps, 1.0 - eps))

    def _logit(p: float) -> float:
        p = _clamp(p)
        return float(np.log(p / (1.0 - p)))

    ref = _clamp(val_prior)
    target = _clamp(test_prior if test_prior is not None else val_prior)
    source = _clamp(train_prior)
    if test_prior is None:
        correction = _logit(target) - _logit(source)
    else:
        correction = _logit(target) - _logit(ref)
    adjusted_logit = _logit(base_threshold) + correction
    return float(1.0 / (1.0 + np.exp(-adjusted_logit)))
