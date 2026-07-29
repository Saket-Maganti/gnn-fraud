"""Ranking, threshold, top-K, and budget-curve metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeAlias

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

IntegerArrayLike: TypeAlias = Sequence[int] | np.ndarray
FloatArrayLike: TypeAlias = Sequence[float] | np.ndarray


def _binary(labels: IntegerArrayLike, positive_label: int = 1) -> np.ndarray:
    y = np.asarray(labels).reshape(-1)
    keep = y != 0
    return (y[keep] == positive_label).astype(int)


def _prepare(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    positive_label: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(labels).reshape(-1)
    score = np.asarray(scores, dtype=float).reshape(-1)
    if len(raw) != len(score):
        raise ValueError("labels and scores length mismatch")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite")
    keep = raw != 0
    y = (raw[keep] == positive_label).astype(int)
    return y, score[keep]


def average_ranks(scores: FloatArrayLike, *, ascending: bool = True) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    ranked = rankdata(values if ascending else -values, method="average")
    return np.asarray(ranked, dtype=float)


def deterministic_top_k(
    scores: FloatArrayLike,
    k: int,
    *,
    identifiers: Sequence[object] | None = None,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if k < 0:
        raise ValueError("k cannot be negative")
    if identifiers is None:
        tie_breaker = np.arange(len(values))
    else:
        if len(identifiers) != len(values):
            raise ValueError("identifiers must align with scores")
        tie_breaker = np.asarray([str(value) for value in identifiers])
    order = np.lexsort((tie_breaker, -values))
    return order[: min(k, len(values))]


def precision_at_k(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    k: int | float,
    positive_label: int = 1,
) -> float:
    y, s = _prepare(labels, scores, positive_label)
    k_eff = _resolve_k(k, len(y))
    if k_eff == 0:
        return 0.0
    return float(y[deterministic_top_k(s, k_eff)].mean())


def recall_at_k(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    k: int | float,
    positive_label: int = 1,
) -> float:
    y, s = _prepare(labels, scores, positive_label)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    k_eff = _resolve_k(k, len(y))
    return float(y[deterministic_top_k(s, k_eff)].sum() / positives)


def _resolve_k(k: int | float, n: int) -> int:
    if isinstance(k, float) and 0 < k <= 1:
        return min(n, max(1, int(np.ceil(k * n)))) if n else 0
    return min(n, max(0, int(k)))


def budget_curve_auc(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    budgets: Sequence[float],
    *,
    metric: str = "recall",
) -> float:
    x = np.asarray(budgets, dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0) or x[0] < 0 or x[-1] > 1:
        raise ValueError("budgets must be strictly increasing fractions in [0,1]")
    fn = recall_at_k if metric == "recall" else precision_at_k if metric == "precision" else None
    if fn is None:
        raise ValueError("budget curve metric must be recall or precision")
    values = np.asarray([fn(labels, scores, float(budget)) for budget in x])
    # NumPy 1.x remains pinned for compatibility with the legacy PyG stack.
    return float(np.trapz(values, x) / (x[-1] - x[0]))


def binary_metrics(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> dict[str, float]:
    y, s = _prepare(labels, scores, positive_label)
    if len(np.unique(y)) < 2:
        auroc = float("nan")
        auprc = float(y.mean()) if len(y) else float("nan")
    else:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    pred = (s >= threshold).astype(int)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    objective: str
    validation_value: float
    tie_policy: str = "smallest_threshold"


def select_threshold_on_validation(
    labels: IntegerArrayLike,
    scores: FloatArrayLike,
    *,
    objective: str,
    thresholds: Iterable[float] | None = None,
    cost_matrix: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> ThresholdSelection:
    """Select only genuine threshold metrics; ranking objectives are rejected."""

    y, s = _prepare(labels, scores)
    if objective in {"auprc", "auroc", "average_precision", "ranking"}:
        raise ValueError(
            f"{objective} is threshold-free; use an explicitly named validation "
            "surrogate or budget-specific selection"
        )
    grid = np.asarray(
        list(thresholds)
        if thresholds is not None
        else sorted(set([0.0, 0.5, 1.0, *s.tolist()])),
        dtype=float,
    )
    if grid.size == 0 or np.any((grid < 0) | (grid > 1)):
        raise ValueError("threshold grid must be non-empty and lie in [0,1]")
    best_t = float(grid[0])
    best_value = -np.inf
    for threshold in np.sort(grid):
        pred = (s >= threshold).astype(int)
        if objective in {"f1", "val_f1"}:
            value = float(f1_score(y, pred, zero_division=0))
        elif objective in {"balanced_accuracy", "val_balanced_accuracy"}:
            value = float(balanced_accuracy_score(y, pred))
        elif objective in {"cost_risk", "declared_cost_risk"}:
            if cost_matrix is None:
                raise ValueError("cost-risk threshold selection requires cost_matrix")
            costs = np.asarray(cost_matrix, dtype=float)
            value = -float(costs[y, pred].mean())
        else:
            raise ValueError(f"unknown threshold objective: {objective}")
        if value > best_value:
            best_t, best_value = float(threshold), value
    return ThresholdSelection(
        threshold=best_t,
        objective=objective,
        validation_value=-best_value if objective in {"cost_risk", "declared_cost_risk"} else best_value,
    )


def select_budget_cutoff(
    scores: FloatArrayLike,
    *,
    budget: int | float,
) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    k = _resolve_k(budget, len(values))
    if k == 0:
        return float("inf"), 0
    selected = deterministic_top_k(values, k)
    return float(values[selected[-1]]), k
