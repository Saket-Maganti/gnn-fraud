"""Target-label-free counterfactual routing checks with fixed expert scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from coregraph.routing.masks import apply_feasible_mask


@dataclass(frozen=True)
class CounterfactualRouting:
    weights: torch.Tensor
    blended_scores: torch.Tensor
    selected_expert: torch.Tensor
    all_unavailable: torch.Tensor


def reroute_fixed_scores(
    expert_scores: torch.Tensor,
    routing_logits: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> CounterfactualRouting:
    if expert_scores.shape != routing_logits.shape:
        raise ValueError("counterfactual scores and logits must align")
    routing = apply_feasible_mask(routing_logits, feasible_mask)
    blended = (routing.probabilities * expert_scores).sum(dim=-1)
    return CounterfactualRouting(
        routing.probabilities,
        blended,
        routing.selected_expert,
        routing.all_unavailable,
    )
