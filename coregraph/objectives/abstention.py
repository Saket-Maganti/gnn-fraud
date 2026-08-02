"""Differentiable abstention objective with explicit zero-coverage semantics."""

from __future__ import annotations

import torch


def abstention_objective(
    per_example_loss: torch.Tensor,
    abstention_probability: torch.Tensor,
    *,
    abstention_cost: float,
    coverage_floor: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if per_example_loss.shape != abstention_probability.shape or not per_example_loss.numel():
        raise ValueError("abstention objective requires non-empty aligned vectors")
    if abstention_cost < 0 or not 0 <= coverage_floor <= 1:
        raise ValueError("invalid abstention cost or coverage floor")
    if bool(((abstention_probability < 0) | (abstention_probability > 1)).any()):
        raise ValueError("abstention probabilities must lie in [0,1]")
    acceptance = 1 - abstention_probability
    effective = acceptance * per_example_loss + abstention_probability * abstention_cost
    coverage = acceptance.mean()
    penalty = torch.relu(coverage.new_tensor(coverage_floor) - coverage)
    return effective.mean() + penalty, {
        "effective_risk": effective.mean(),
        "coverage": coverage,
        "coverage_penalty": penalty,
    }
