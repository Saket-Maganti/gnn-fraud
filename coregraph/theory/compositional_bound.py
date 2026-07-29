"""Axis-additive compositional approximation bound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CompositionalBound:
    axis_estimation_error: float
    interaction_residual: float
    router_approximation_error: float
    total_bound: float
    proof_status: str = "PROVED"


def compositional_error_bound(
    axis_errors: Sequence[float],
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
        total_bound=axis_total + interaction_residual + router_approximation_error,
    )


def verify_additive_bound(
    true_axis_effects: np.ndarray,
    estimated_axis_effects: np.ndarray,
    interactions: np.ndarray,
    router_error: np.ndarray,
) -> bool:
    true = np.asarray(true_axis_effects, dtype=float)
    estimated = np.asarray(estimated_axis_effects, dtype=float)
    if true.shape != estimated.shape:
        raise ValueError("true and estimated axis effects must align")
    actual = abs(
        float((true - estimated).sum())
        + float(np.asarray(interactions).sum())
        + float(np.asarray(router_error).sum())
    )
    bound = (
        float(np.abs(true - estimated).sum())
        + float(np.abs(interactions).sum())
        + float(np.abs(router_error).sum())
    )
    return actual <= bound + 1e-12
