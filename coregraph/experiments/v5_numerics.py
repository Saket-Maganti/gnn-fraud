"""V5.2 float64 numerical contracts for feasible saved-output routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


NUMERICAL_IMPLEMENTATION_VERSION = "coregraph_v5_2_float64_simplex_v1"
SCIENTIFIC_COMPUTE_DTYPE = "float64"
WEIGHT_NEGATIVE_TOLERANCE = 1e-12
SIMPLEX_TOLERANCE = 1e-12
HULL_PROJECTION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SimplexDiagnostics:
    max_abs_weight_sum_error_before_residual: float
    max_abs_weight_sum_error_after_residual: float
    rows_with_unavailable_nonzero_weight: int
    input_rows_with_unavailable_nonzero_weight: int
    rows_forced_to_abstain_no_feasible_expert: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HullDiagnostics:
    max_pre_projection_hull_violation: float
    rows_projected_to_hull: int
    max_projection_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_normalized_feasible_weights(
    weights: np.ndarray,
    feasible: np.ndarray,
    active: np.ndarray,
    tolerance: float,
) -> float:
    active_sum = np.sum(weights[active], axis=1, dtype=np.float64)
    maximum_error = (
        float(np.max(np.abs(active_sum - 1.0))) if active_sum.size else 0.0
    )
    if maximum_error > tolerance:
        raise ValueError(
            "float64 feasible weight normalization failed: "
            f"maximum_sum_error={maximum_error}"
        )
    if np.any(weights < -tolerance) or np.any(weights > 1.0 + tolerance):
        raise ValueError("normalized feasible weights lie outside [0, 1]")
    if np.any((~feasible) & (weights != 0.0)):
        raise RuntimeError("unavailable expert retained nonzero normalized weight")
    return maximum_error


def normalize_feasible_weights_float64(
    raw_weights: np.ndarray,
    availability: np.ndarray,
    *,
    negative_tolerance: float = WEIGHT_NEGATIVE_TOLERANCE,
    simplex_tolerance: float = SIMPLEX_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, SimplexDiagnostics]:
    """Mask and normalize feasible weights with deterministic residual correction."""

    weights = np.asarray(raw_weights, dtype=np.float64)
    feasible = np.asarray(availability, dtype=bool)
    if weights.ndim != 2 or weights.shape != feasible.shape:
        raise ValueError("raw weights and availability must be aligned matrices")
    if not np.isfinite(weights).all():
        raise ValueError("raw routing weights must be finite")
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0:
        raise ValueError("negative weight tolerance must be finite and non-negative")
    if not np.isfinite(simplex_tolerance) or simplex_tolerance <= 0:
        raise ValueError("simplex tolerance must be finite and positive")
    if np.any(weights < -negative_tolerance):
        minimum = float(np.min(weights))
        raise ValueError(f"materially negative routing weight: minimum={minimum}")

    input_unavailable = np.any((~feasible) & (weights != 0.0), axis=1)
    weights = np.where(weights < 0.0, 0.0, weights)
    masked = np.where(feasible, weights, 0.0)
    feasible_sum = np.sum(masked, axis=1, dtype=np.float64)
    active = feasible.any(axis=1) & (feasible_sum > 0.0)
    normalized = np.zeros_like(masked, dtype=np.float64)
    normalized[active] = masked[active] / feasible_sum[active, None]

    before_sum = np.sum(normalized[active], axis=1, dtype=np.float64)
    before_error = (
        float(np.max(np.abs(before_sum - 1.0))) if before_sum.size else 0.0
    )
    active_rows = np.flatnonzero(active)
    if active_rows.size:
        largest = np.argmax(normalized[active], axis=1)
        residual = 1.0 - before_sum
        normalized[active_rows, largest] += residual

    normalized[~feasible] = 0.0
    after_error = _validate_normalized_feasible_weights(
        normalized, feasible, active, simplex_tolerance
    )
    unavailable_after = np.any((~feasible) & (normalized != 0.0), axis=1)
    forced_abstain = ~active
    return (
        normalized,
        forced_abstain,
        SimplexDiagnostics(
            max_abs_weight_sum_error_before_residual=before_error,
            max_abs_weight_sum_error_after_residual=after_error,
            rows_with_unavailable_nonzero_weight=int(unavailable_after.sum()),
            input_rows_with_unavailable_nonzero_weight=int(input_unavailable.sum()),
            rows_forced_to_abstain_no_feasible_expert=int(forced_abstain.sum()),
        ),
    )


def routed_scores_in_feasible_hull_float64(
    normalized_weights: np.ndarray,
    expert_scores: np.ndarray,
    availability: np.ndarray,
    *,
    projection_tolerance: float = HULL_PROJECTION_TOLERANCE,
) -> tuple[np.ndarray, HullDiagnostics]:
    """Compute float64 routed scores and enforce the feasible score hull."""

    weights = np.asarray(normalized_weights, dtype=np.float64)
    scores = np.asarray(expert_scores, dtype=np.float64)
    feasible = np.asarray(availability, dtype=bool)
    if weights.ndim != 2 or weights.shape != scores.shape or scores.shape != feasible.shape:
        raise ValueError("weights, expert scores, and availability must align")
    if not np.isfinite(weights).all() or not np.isfinite(scores).all():
        raise ValueError("routing weights and expert scores must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("expert scores must be probabilities")
    if not np.isfinite(projection_tolerance) or projection_tolerance < 0:
        raise ValueError("hull projection tolerance must be finite and non-negative")
    if np.any((~feasible) & (weights != 0.0)):
        raise ValueError("unavailable expert contributes to routed score")

    routed = np.sum(weights * scores, axis=1, dtype=np.float64)
    has_weight = np.sum(weights, axis=1, dtype=np.float64) > 0.0
    lower = np.min(np.where(feasible, scores, np.inf), axis=1)
    upper = np.max(np.where(feasible, scores, -np.inf), axis=1)
    lower_violation = np.where(has_weight, np.maximum(lower - routed, 0.0), 0.0)
    upper_violation = np.where(has_weight, np.maximum(routed - upper, 0.0), 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    maximum_violation = float(np.max(violation)) if violation.size else 0.0
    if maximum_violation > projection_tolerance:
        raise ValueError(
            "routed score substantively violates feasible hull: "
            f"maximum_violation={maximum_violation}, tolerance={projection_tolerance}"
        )
    projected = routed.copy()
    projected[has_weight] = np.minimum(
        np.maximum(routed[has_weight], lower[has_weight]), upper[has_weight]
    )
    delta = np.abs(projected - routed)
    return (
        projected,
        HullDiagnostics(
            max_pre_projection_hull_violation=maximum_violation,
            rows_projected_to_hull=int(np.sum(delta > 0.0)),
            max_projection_delta=float(np.max(delta)) if delta.size else 0.0,
        ),
    )
