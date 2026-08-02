"""GraphSafe-V2 prediction-level selective reliability utilities.

The implementation is intentionally prediction-level: it consumes saved
prediction rows and computes selective-risk and review-budget summaries without
requiring model retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class GraphSafeV2Config:
    mode: str = "graphsafe_v2_default"
    decision_threshold: float = 0.5
    conservative_margin: float = 0.1
    coverage_levels: tuple[float, ...] = (0.5, 0.7, 0.9, 0.95)
    review_budgets: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)


REQUIRED_COLUMNS = {
    "dataset",
    "protocol",
    "model",
    "seed",
    "split",
    "node_id",
    "timestep",
    "y_true",
    "score",
    "label_known",
    "artifact_source",
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def validate_prediction_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["empty_prediction_rows"]
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    return [f"missing_columns:{','.join(missing)}"] if missing else []


def _confidence(row: dict[str, Any], config: GraphSafeV2Config) -> float:
    score = _float(row.get("score"))
    if config.mode == "score_threshold":
        return score
    if config.mode == "uncertainty_margin":
        return abs(score - config.decision_threshold)
    if config.mode == "temporal_stability":
        return max(score, 1.0 - score)
    if config.mode == "conformal_quantile":
        return max(score, 1.0 - score)
    return max(score, 1.0 - score) + min(abs(score - config.decision_threshold), config.conservative_margin)


def _prediction(row: dict[str, Any], config: GraphSafeV2Config) -> int:
    return 1 if _float(row.get("score")) >= config.decision_threshold else 0


def confidence_scores(
    scores: Iterable[float],
    config: GraphSafeV2Config | None = None,
) -> np.ndarray:
    """Expose the label-free GraphSafe confidence used for selective routing."""

    selected = config or GraphSafeV2Config()
    return np.asarray(
        [_confidence({"score": float(score)}, selected) for score in scores],
        dtype=float,
    )


def selective_risk_rows(rows: list[dict[str, Any]], config: GraphSafeV2Config | None = None) -> list[dict[str, Any]]:
    config = config or GraphSafeV2Config()
    errors = validate_prediction_rows(rows)
    if errors:
        raise ValueError(";".join(errors))
    ordered = sorted(rows, key=lambda row: _confidence(row, config), reverse=True)
    total = len(ordered)
    out: list[dict[str, Any]] = []
    for coverage in config.coverage_levels:
        keep = max(1, min(total, round(total * coverage)))
        accepted = ordered[:keep]
        false = sum(1 for row in accepted if _prediction(row, config) != _int(row.get("y_true")))
        positives_accepted = sum(1 for row in accepted if _int(row.get("y_true")) == 1)
        false_negative = sum(1 for row in accepted if _int(row.get("y_true")) == 1 and _prediction(row, config) == 0)
        out.append(
            {
                "mode": config.mode,
                "coverage": round(keep / total, 6),
                "target_coverage": coverage,
                "accepted_count": keep,
                "total_count": total,
                "selective_risk": round(false / keep, 6),
                "false_negative_risk_accepted": round(false_negative / positives_accepted, 6) if positives_accepted else 0.0,
                "abstention_fraction": round(1.0 - keep / total, 6),
            }
        )
    return out


def review_budget_rows(rows: list[dict[str, Any]], config: GraphSafeV2Config | None = None) -> list[dict[str, Any]]:
    config = config or GraphSafeV2Config()
    errors = validate_prediction_rows(rows)
    if errors:
        raise ValueError(";".join(errors))
    ordered = sorted(rows, key=lambda row: _float(row.get("score")), reverse=True)
    total = len(ordered)
    total_pos = sum(1 for row in ordered if _int(row.get("y_true")) == 1)
    out: list[dict[str, Any]] = []
    for budget in config.review_budgets:
        keep = max(1, min(total, round(total * budget)))
        reviewed = ordered[:keep]
        captured = sum(1 for row in reviewed if _int(row.get("y_true")) == 1)
        out.append(
            {
                "budget": budget,
                "review_count": keep,
                "precision_at_budget": round(captured / keep, 6),
                "recall_at_budget": round(captured / total_pos, 6) if total_pos else 0.0,
                "captured_positives": captured,
                "total_positives": total_pos,
            }
        )
    return out


def coverage_at_risk(rows: list[dict[str, Any]], risk_targets: Iterable[float] = (0.02, 0.05, 0.1)) -> list[dict[str, Any]]:
    curve = selective_risk_rows(rows, GraphSafeV2Config(coverage_levels=tuple(i / 100 for i in range(5, 101, 5))))
    out = []
    for target in risk_targets:
        feasible = [row for row in curve if float(row["selective_risk"]) <= target]
        best = max(feasible, key=lambda row: float(row["coverage"])) if feasible else None
        out.append(
            {
                "target_risk": target,
                "coverage_at_target_risk": "" if best is None else best["coverage"],
                "status": "PENDING_OR_UNMET" if best is None else "SUPPORTED_BY_INPUT_ROWS",
            }
        )
    return out


def safe_claim_rows() -> list[dict[str, str]]:
    return [
        {
            "claim": "GraphSafe-V2 can be evaluated as selective decision support using coverage-risk and review-budget metrics.",
            "status": "ALLOWABLE_METHOD_CLAIM",
            "reason": "It is a prediction-level wrapper and does not assert rank-metric gains.",
        },
        {
            "claim": "GraphSafe-V2 improves AUPRC/AUROC or universally wins.",
            "status": "FORBIDDEN_WITHOUT_EVIDENCE",
            "reason": "The wrapper is not a rank-metric improvement claim and V24 has not executed experiments.",
        },
    ]
