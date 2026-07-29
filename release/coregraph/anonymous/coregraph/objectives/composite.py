"""Configurable CoReGraph multi-objective loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from coregraph.objectives.budget import soft_recall_at_k_loss
from coregraph.objectives.calibration import soft_brier_loss
from coregraph.objectives.classification import binary_cross_entropy
from coregraph.objectives.compute import expected_compute_cost
from coregraph.objectives.cvar import empirical_cvar
from coregraph.objectives.ranking import pairwise_logistic_ranking_loss
from coregraph.objectives.regret import contract_regret


@dataclass(frozen=True)
class ObjectiveWeights:
    average: float = 1.0
    ranking: float = 0.0
    robust_regret: float = 0.0
    budget: float = 0.0
    stability: float = 0.0
    compute: float = 0.0
    calibration: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("objective weights must be non-negative")


class CompositeObjective:
    def __init__(self, weights: ObjectiveWeights, *, cvar_alpha: float = 0.8):
        self.weights = weights
        self.cvar_alpha = cvar_alpha

    def __call__(
        self,
        *,
        router_logits: torch.Tensor,
        targets: torch.Tensor,
        contract_router_risk: torch.Tensor | None = None,
        contract_expert_risks: torch.Tensor | None = None,
        contract_availability: torch.Tensor | None = None,
        expert_weights: torch.Tensor | None = None,
        expert_costs: torch.Tensor | None = None,
        stability_penalty: torch.Tensor | None = None,
        review_k: int | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        zero = router_logits.sum() * 0
        terms: dict[str, torch.Tensor] = {
            "average": binary_cross_entropy(router_logits, targets),
            "ranking": pairwise_logistic_ranking_loss(router_logits, targets),
            "budget": (
                soft_recall_at_k_loss(router_logits, targets, review_k)
                if review_k is not None
                else zero
            ),
            "calibration": soft_brier_loss(router_logits, targets),
            "stability": stability_penalty if stability_penalty is not None else zero,
            "compute": (
                expected_compute_cost(expert_weights, expert_costs)
                if expert_weights is not None and expert_costs is not None
                else zero
            ),
            "robust_regret": zero,
        }
        if (
            contract_router_risk is not None
            and contract_expert_risks is not None
            and contract_availability is not None
        ):
            regrets = contract_regret(
                contract_router_risk,
                contract_expert_risks,
                contract_availability,
            )
            terms["robust_regret"] = empirical_cvar(regrets, self.cvar_alpha)
        total = sum(
            getattr(self.weights, name) * value
            for name, value in terms.items()
        )
        return total, terms
