"""Deterministic label-free fallback strategies."""

from __future__ import annotations

from enum import Enum

import torch


class FallbackStrategy(str, Enum):
    FEATURE_ONLY_SAFE = "feature_only_safe"
    BEST_SOURCE_VALIDATION = "best_source_validation"
    STATIC_AVERAGE = "static_average"
    ABSTAIN = "abstain"
    TOP_TWO_MIXTURE = "top_two_mixture"


def fallback_weights(
    availability: torch.Tensor,
    *,
    strategy: FallbackStrategy,
    feature_expert_index: int = 0,
    source_validation_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feasible weights and an abstention flag per row."""

    mask = availability.bool()
    batch, experts = mask.shape
    weights = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    abstain = torch.zeros(batch, dtype=torch.bool, device=mask.device)
    for row in range(batch):
        feasible = torch.where(mask[row])[0]
        if feasible.numel() == 0:
            abstain[row] = True
            continue
        if strategy is FallbackStrategy.ABSTAIN:
            abstain[row] = True
        elif strategy is FallbackStrategy.FEATURE_ONLY_SAFE:
            chosen = (
                feature_expert_index
                if feature_expert_index in feasible.tolist()
                else int(feasible[0])
            )
            weights[row, chosen] = 1
        elif strategy is FallbackStrategy.BEST_SOURCE_VALIDATION:
            if source_validation_order is None:
                raise ValueError("source validation fallback needs a fixed expert order")
            chosen = next(
                (int(index) for index in source_validation_order if mask[row, int(index)]),
                int(feasible[0]),
            )
            weights[row, chosen] = 1
        elif strategy is FallbackStrategy.STATIC_AVERAGE:
            weights[row, feasible] = 1 / feasible.numel()
        elif strategy is FallbackStrategy.TOP_TWO_MIXTURE:
            chosen_pair = feasible[:2]
            weights[row, chosen_pair] = 1 / chosen_pair.numel()
    return weights, abstain
