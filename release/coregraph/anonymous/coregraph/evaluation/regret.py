"""Loss-parametric regret summaries.

Callers must supply router and oracle risks measured with one identical,
explicitly declared loss. Pilot result schemas use loss-specific names rather
than exporting these generic helper labels.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


V5_REGRET_NUMERIC_TOLERANCE = -1e-12


def feasible_row_oracle_brier_with_abstention(
    *,
    labels: np.ndarray,
    expert_scores: np.ndarray,
    availability: np.ndarray,
    abstention_cost: float,
) -> np.ndarray:
    """Return the matched-action row oracle over feasible experts and abstention."""

    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(expert_scores, dtype=np.float64)
    feasible = np.asarray(availability, dtype=bool)
    if scores.ndim != 2 or scores.shape != feasible.shape:
        raise ValueError("expert scores and availability must be aligned matrices")
    if scores.shape[0] != len(y):
        raise ValueError("expert scores do not align with labels")
    if np.any((y < 0) | (y > 1)) or not np.isfinite(y).all():
        raise ValueError("Brier labels must be finite binary values")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("expert scores must be finite probabilities")
    if not np.isfinite(abstention_cost) or abstention_cost < 0:
        raise ValueError("abstention cost must be finite and non-negative")
    expert_loss = (scores - y[:, None]) ** 2
    best_feasible_expert = np.min(np.where(feasible, expert_loss, np.inf), axis=1)
    return np.minimum(best_feasible_expert, float(abstention_cost))


def v5_matched_action_brier_metrics(
    *,
    labels: np.ndarray,
    method_scores: np.ndarray,
    method_abstains: np.ndarray,
    expert_scores: np.ndarray,
    availability: np.ndarray,
    abstention_cost: float,
    tolerance: float = V5_REGRET_NUMERIC_TOLERANCE,
) -> dict[str, float | None]:
    """Compute V5 primary regret and the separate best-fixed diagnostic."""

    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(method_scores, dtype=np.float64).reshape(-1)
    abstains = np.asarray(method_abstains, dtype=bool).reshape(-1)
    experts = np.asarray(expert_scores, dtype=np.float64)
    feasible = np.asarray(availability, dtype=bool)
    if len(y) == 0:
        raise ValueError("cannot evaluate an empty target")
    if scores.shape != y.shape or abstains.shape != y.shape:
        raise ValueError("method outputs do not align with labels")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("method scores must be finite probabilities")
    oracle_loss = feasible_row_oracle_brier_with_abstention(
        labels=y,
        expert_scores=experts,
        availability=feasible,
        abstention_cost=abstention_cost,
    )
    method_loss = np.where(abstains, abstention_cost, (scores - y) ** 2)
    row_regret = method_loss - oracle_loss
    minimum = float(np.min(row_regret))
    if minimum < tolerance:
        raise ValueError(
            "contract regret below the frozen numeric tolerance: "
            f"minimum_row_regret={minimum}, tolerance={tolerance}"
        )
    canonical_row_regret = np.maximum(row_regret, 0.0)
    primary_regret = float(np.mean(canonical_row_regret, dtype=np.float64))
    fixed_feasible = feasible.all(axis=0)
    if fixed_feasible.any():
        fixed_risks = np.mean((experts - y[:, None]) ** 2, axis=0)
        best_fixed = float(np.min(fixed_risks[fixed_feasible]))
        excess_fixed: float | None = float(np.mean(method_loss) - best_fixed)
    else:
        best_fixed = None
        excess_fixed = None
    return {
        "contract_brier_risk": float(np.mean(method_loss)),
        "feasible_row_oracle_loss_with_abstention": float(np.mean(oracle_loss)),
        "contract_regret_vs_feasible_row_oracle": primary_regret,
        "best_fixed_nonabstaining_expert_brier": best_fixed,
        "excess_cost_vs_best_fixed_nonabstaining_expert": excess_fixed,
        "minimum_raw_row_regret": minimum,
        "rows_with_raw_regret_below_zero": int(np.sum(row_regret < 0.0)),
        "rows_with_raw_regret_below_tolerance": int(np.sum(row_regret < tolerance)),
    }


def contract_regrets(
    router_risk: Mapping[str, float],
    expert_risk: Mapping[str, Mapping[str, float]],
    availability: Mapping[str, Mapping[str, bool]],
) -> dict[str, float]:
    regrets: dict[str, float] = {}
    for contract, risk in router_risk.items():
        feasible = [
            value
            for expert, value in expert_risk[contract].items()
            if availability[contract].get(expert, False)
        ]
        if not feasible:
            raise ValueError(f"contract {contract} has no feasible oracle expert")
        regrets[contract] = float(risk - min(feasible))
    return regrets


def regret_summary(regrets: Mapping[str, float], *, alpha: float = 0.8) -> dict[str, float]:
    values = np.asarray(list(regrets.values()), dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarise an empty regret set")
    tail_count = max(1, int(np.ceil((1 - alpha) * len(values))))
    tail = np.sort(values)[-tail_count:]
    return {
        "mean_contract_regret": float(values.mean()),
        "maximum_contract_regret": float(values.max()),
        "median_contract_regret": float(np.median(values)),
        "cvar_contract_regret": float(tail.mean()),
    }
