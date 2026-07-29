"""Two-contract fixed-mixture lower bound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedMixtureResult:
    optimal_weight_expert_one: float
    worst_contract_regret: float
    lower_bound: float
    contract_aware_regret: float
    proof_status: str = "PROVED"


def fixed_mixture_lower_bound(delta_one: float, delta_two: float) -> FixedMixtureResult:
    """Exact minimax regret for two experts whose ordering crosses.

    Contract 1 risks are ``(0, delta_one)`` and contract 2 risks are
    ``(delta_two, 0)``. A contract-independent randomized mixture selects
    expert one with probability ``w``. Its regrets are
    ``(1-w) delta_one`` and ``w delta_two``. Equalising them gives the exact
    positive lower bound ``delta_one*delta_two/(delta_one+delta_two)``.
    """

    if delta_one <= 0 or delta_two <= 0:
        raise ValueError("crossing gaps must be strictly positive")
    weight = delta_one / (delta_one + delta_two)
    bound = delta_one * delta_two / (delta_one + delta_two)
    return FixedMixtureResult(weight, bound, bound, 0.0)


def fixed_mixture_regret_curve(
    delta_one: float,
    delta_two: float,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("mixture weights must lie in [0,1]")
    return np.maximum((1 - values) * delta_one, values * delta_two)
