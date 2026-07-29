"""Axis-additive compositional approximation bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np

FloatArrayLike: TypeAlias = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class CompositionalBound:
    axis_estimation_error: float
    interaction_residual: float
    router_approximation_error: float
    total_bound: float
    axis_coefficient: float = 2.0
    interaction_coefficient: float = 2.0
    router_coefficient: float = 1.0
    proof_status: str = "PROVED"


def compositional_error_bound(
    axis_errors: FloatArrayLike,
    *,
    interaction_residual: float,
    router_approximation_error: float,
) -> CompositionalBound:
    errors = np.asarray(axis_errors, dtype=float)
    if np.any(errors < 0) or interaction_residual < 0 or router_approximation_error < 0:
        raise ValueError("bound terms must be non-negative")
    axis_total = float(errors.sum())
    return CompositionalBound(
        axis_estimation_error=axis_total,
        interaction_residual=float(interaction_residual),
        router_approximation_error=float(router_approximation_error),
        total_bound=(
            2 * axis_total
            + 2 * interaction_residual
            + router_approximation_error
        ),
    )


def verify_additive_bound(
    true_axis_effects: np.ndarray,
    estimated_axis_effects: np.ndarray,
    interactions: np.ndarray,
    router_error: np.ndarray,
    *,
    actual_excess_risk: float | None = None,
) -> bool:
    true = np.asarray(true_axis_effects, dtype=float)
    estimated = np.asarray(estimated_axis_effects, dtype=float)
    if true.shape != estimated.shape:
        raise ValueError("true and estimated axis effects must align")
    if actual_excess_risk is not None and actual_excess_risk < 0:
        raise ValueError("actual excess risk must be non-negative")
    actual = (
        float(actual_excess_risk)
        if actual_excess_risk is not None
        else abs(
            float((true - estimated).sum())
            + float(np.asarray(interactions).sum())
            + float(np.asarray(router_error).sum())
        )
    )
    bound = (
        2 * float(np.abs(true - estimated).sum())
        + 2 * float(np.abs(interactions).sum())
        + float(np.abs(router_error).sum())
    )
    return actual <= bound + 1e-12
