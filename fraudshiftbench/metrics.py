"""Reusable CPU-only metrics for FraudShiftBench.

The functions here are intentionally small, deterministic, and independent of
training frameworks. They accept plain mappings/sequences so scripts can reuse
them on RB01/RB02 artifacts, synthetic stress tests, and reviewer fixtures.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Sequence

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 2

EVIDENCE_LEVELS = {
    "supported",
    "diagnostic",
    "sensitivity",
    "scaffold",
    "blocked",
}


def _clip01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _common_score_maps(
    scores_a: Mapping[str, float], scores_b: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    common = sorted(set(scores_a) & set(scores_b))
    return (
        {m: float(scores_a[m]) for m in common},
        {m: float(scores_b[m]) for m in common},
    )


def rank_reversal_score(scores_a: Mapping[str, float], scores_b: Mapping[str, float]) -> Optional[float]:
    """Fraction of pairwise model orderings that reverse between two protocols.

    Returns ``None`` when fewer than two models have scores under both protocols.
    Ties are ignored. A score of 0 means no reversals; 1 means complete reversal.
    """

    a, b = _common_score_maps(scores_a, scores_b)
    models = sorted(a)
    if len(models) < 2:
        return None
    total = 0
    reversed_pairs = 0
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            da = a[left] - a[right]
            db = b[left] - b[right]
            if da == 0 or db == 0:
                continue
            total += 1
            if da * db < 0:
                reversed_pairs += 1
    if total == 0:
        return None
    return reversed_pairs / total


def leaderboard_instability_score(
    scores_a: Mapping[str, float], scores_b: Mapping[str, float]
) -> Optional[float]:
    """Alias for rank reversal score, named for paper/toolkit readability."""

    return rank_reversal_score(scores_a, scores_b)


def protocol_risk_index(
    *,
    leaderboard_flip_probability: Optional[float] = None,
    rank_instability: Optional[float] = None,
    temporal_prior_drift: Optional[float] = None,
    protocol_metric_gap: Optional[float] = None,
    prediction_disagreement: Optional[float] = None,
    confidence_instability: Optional[float] = None,
) -> Optional[float]:
    """Transparent composite protocol-risk index.

    All available components are clipped to [0, 1] and averaged. Missing
    components are ignored; ``None`` is returned only when no component exists.
    Negative metric gaps do not count as risk because they are not optimistic
    inflation.
    """

    components: list[float] = []
    for value in (
        leaderboard_flip_probability,
        rank_instability,
        temporal_prior_drift,
        protocol_metric_gap,
        prediction_disagreement,
        confidence_instability,
    ):
        if value is None:
            continue
        components.append(_clip01(value))
    if not components:
        return None
    return sum(components) / len(components)


def graph_harm_rate(labels: Iterable[str]) -> float:
    """Share of MLP-vs-GNN comparison labels equal to ``graph_harm``."""

    counts = Counter(labels)
    total = sum(counts.values())
    return counts.get("graph_harm", 0) / total if total else 0.0


def graph_help_rate(labels: Iterable[str]) -> float:
    """Share of MLP-vs-GNN comparison labels equal to ``graph_help``."""

    counts = Counter(labels)
    total = sum(counts.values())
    return counts.get("graph_help", 0) / total if total else 0.0


def high_confidence_harm_rate(
    rows: Iterable[Mapping[str, Any]],
    *,
    margin_key: str = "gnn_margin",
    label_key: str = "category",
    threshold: float = 0.4,
) -> float:
    """High-confidence graph-harm share among comparison rows."""

    total = 0
    high_conf_harm = 0
    for row in rows:
        total += 1
        try:
            margin = float(row.get(margin_key, 0.0))
        except (TypeError, ValueError):
            margin = 0.0
        if row.get(label_key) == "graph_harm" and margin >= threshold:
            high_conf_harm += 1
    return high_conf_harm / total if total else 0.0


def _binary_labels(labels: Sequence[int]) -> list[int]:
    return [1 if int(y) == POSITIVE_LABEL else 0 for y in labels]


def _top_indices(scores: Sequence[float], budget: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return order[: max(0, min(int(budget), len(order)))]


def fraud_recall_at_budget(labels: Sequence[int], scores: Sequence[float], budget: int) -> float:
    """Fraud recall when reviewing the top-scored ``budget`` items."""

    y = _binary_labels(labels)
    positives = sum(y)
    if positives == 0:
        return 0.0
    hits = sum(y[i] for i in _top_indices(scores, budget))
    return hits / positives


def false_positive_workload(labels: Sequence[int], scores: Sequence[float], budget: int) -> int:
    """Number of non-fraud items sent to review in top-scored ``budget`` rows."""

    y = _binary_labels(labels)
    return sum(1 for i in _top_indices(scores, budget) if y[i] == 0)


def protocol_regret(scores: Mapping[str, float], selected_model: str) -> float:
    """Metric regret of selecting ``selected_model`` under one protocol."""

    if not scores or selected_model not in scores:
        return 0.0
    best = max(float(v) for v in scores.values())
    return max(0.0, best - float(scores[selected_model]))


def protocol_robust_selection_regret(
    scores_by_protocol: Mapping[str, Mapping[str, float]],
    selected_model: str,
) -> dict[str, float]:
    """Per-protocol, worst-case, and average regret for a selected model."""

    per_protocol = {
        protocol: protocol_regret(scores, selected_model)
        for protocol, scores in sorted(scores_by_protocol.items())
    }
    vals = list(per_protocol.values())
    return {
        **per_protocol,
        "worst_case_regret": max(vals) if vals else 0.0,
        "average_regret": sum(vals) / len(vals) if vals else 0.0,
    }


def evidence_bound_claim_status(
    evidence_level: str,
    *,
    has_real_results: bool = False,
    has_second_dataset_results: bool = False,
    is_simulation: bool = False,
) -> str:
    """Map evidence state to a claim-safe status string."""

    level = str(evidence_level).strip().lower()
    if level not in EVIDENCE_LEVELS:
        return "BLOCKED_UNKNOWN_EVIDENCE"
    if level == "blocked":
        return "BLOCKED_NOT_CLAIMED"
    if level == "scaffold":
        return "SCAFFOLD_ONLY"
    if level == "sensitivity" or is_simulation:
        return "SENSITIVITY_ONLY"
    if level == "diagnostic":
        return "SUPPORTED_DIAGNOSTIC"
    if level == "supported" and has_real_results:
        if has_second_dataset_results:
            return "SUPPORTED_MULTI_DATASET"
        return "SUPPORTED_SINGLE_DATASET"
    return "PENDING_RESULTS"
