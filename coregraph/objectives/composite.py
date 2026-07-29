"""End-to-end CoReGraph multi-objective loss.

Normalisation is explicit:

- classification, calibration, compute, stability, and abstention are means;
- contract risks are example means within each declared group and then
  balanced equally across groups;
- headline regret is relative to one best feasible expert over each whole
  contract; the instance-clairvoyant oracle is an optional diagnostic only;
- CVaR is a mean of the worst empirical ``1-alpha`` group regrets;
- budget losses are computed separately within each constrained contract and
  balanced over constrained groups.

Consequently every default term is order one; coefficients remain scientific
hyperparameters and must be frozen on source validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from coregraph.objectives.budget import soft_recall_at_k_loss
from coregraph.objectives.calibration import soft_brier_loss
from coregraph.objectives.classification import binary_cross_entropy_values
from coregraph.objectives.compute import expected_compute_cost
from coregraph.objectives.cvar import empirical_cvar
from coregraph.objectives.ranking import pairwise_logistic_ranking_loss
from coregraph.objectives.scores import ScoreType
from coregraph.routing.abstention import abstention_capacity_penalty


@dataclass(frozen=True)
class ObjectiveWeights:
    average: float = 1.0
    ranking: float = 0.0
    robust_regret: float = 0.0
    budget: float = 0.0
    stability: float = 0.0
    compute: float = 0.0
    calibration: float = 0.0
    abstention: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("objective weights must be non-negative")


def _group_means(
    values: torch.Tensor,
    group_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] != group_indices.numel():
        raise ValueError("group indices must align with example tensors")
    groups = torch.unique(group_indices, sorted=True)
    if groups.numel() == 0:
        raise ValueError("at least one contract group is required")
    return (
        torch.stack(
            [values[group_indices == group].mean(dim=0) for group in groups],
            dim=0,
        ),
        groups,
    )


class CompositeObjective:
    def __init__(
        self,
        weights: ObjectiveWeights,
        *,
        cvar_alpha: float = 0.8,
        include_instance_oracle_diagnostic: bool = False,
    ):
        self.weights = weights
        self.cvar_alpha = cvar_alpha
        self.include_instance_oracle_diagnostic = (
            include_instance_oracle_diagnostic
        )

    def __call__(
        self,
        *,
        router_scores: torch.Tensor,
        score_type: ScoreType,
        targets: torch.Tensor,
        group_indices: torch.Tensor,
        expert_scores: torch.Tensor,
        availability_mask: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_costs: torch.Tensor,
        stability_penalty: torch.Tensor | None = None,
        review_k_by_group: Mapping[int, int | None] | None = None,
        abstention_probability: torch.Tensor | None = None,
        forced_abstention: torch.Tensor | None = None,
        abstention_capacity_by_group: (
            Mapping[int, float | None] | None
        ) = None,
        abstention_cost_value: float = 0.0,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        if router_scores.ndim != 1 or targets.shape != router_scores.shape:
            raise ValueError("router scores and targets must be aligned vectors")
        if expert_scores.ndim != 2 or expert_scores.shape[0] != len(targets):
            raise ValueError("expert scores must have shape [examples,experts]")
        if (
            availability_mask.shape != expert_scores.shape
            or expert_weights.shape != expert_scores.shape
        ):
            raise ValueError("expert scores, masks, and weights must align")
        if group_indices.shape != targets.shape:
            raise ValueError("group indices must align with targets")
        if abstention_cost_value < 0:
            raise ValueError("abstention cost cannot be negative")

        router_losses = binary_cross_entropy_values(
            router_scores,
            targets,
            score_type=score_type,
        )
        expanded_targets = targets[:, None].expand_as(expert_scores)
        expert_losses = binary_cross_entropy_values(
            expert_scores.reshape(-1),
            expanded_targets.reshape(-1),
            score_type=score_type,
        ).reshape_as(expert_scores)

        if abstention_probability is None:
            abstain_probability = torch.zeros_like(router_losses)
        else:
            if abstention_probability.shape != router_losses.shape:
                raise ValueError("abstention probability must align with examples")
            abstain_probability = abstention_probability
        if forced_abstention is not None:
            if forced_abstention.shape != router_losses.shape:
                raise ValueError("forced abstention must align with examples")
            abstain_probability = torch.where(
                forced_abstention.bool(),
                torch.ones_like(abstain_probability),
                abstain_probability,
            )
        effective_router_losses = (
            (1 - abstain_probability) * router_losses
            + abstain_probability * abstention_cost_value
        )
        contract_router_risk, groups = _group_means(
            effective_router_losses,
            group_indices,
        )
        contract_expert_risks, _ = _group_means(
            expert_losses,
            group_indices,
        )
        contract_availability = torch.stack(
            [
                availability_mask[group_indices == group].bool().all(dim=0)
                for group in groups
            ],
            dim=0,
        )
        masked_contract_expert_risks = torch.where(
            contract_availability,
            contract_expert_risks,
            torch.full_like(contract_expert_risks, torch.inf),
        )
        contract_feasible = contract_availability.any(dim=-1)
        contract_oracle_risk = masked_contract_expert_risks.min(dim=-1).values
        contract_oracle_risk = torch.where(
            contract_feasible,
            contract_oracle_risk,
            contract_oracle_risk.new_full(
                contract_oracle_risk.shape,
                abstention_cost_value,
            ),
        )
        regrets = contract_router_risk - contract_oracle_risk
        zero = router_scores.sum() * 0
        budget_losses: list[torch.Tensor] = []
        if review_k_by_group is not None:
            group_ids = {int(group.item()) for group in groups}
            if set(review_k_by_group) != group_ids:
                raise ValueError(
                    "review budget map must cover every contract group exactly"
                )
            for group in groups:
                group_id = int(group.item())
                review_k = review_k_by_group[group_id]
                if review_k is not None:
                    if not isinstance(review_k, int):
                        raise TypeError("per-contract review K must be an integer")
                    keep = group_indices == group
                    budget_losses.append(
                        soft_recall_at_k_loss(
                            router_scores[keep],
                            targets[keep],
                            review_k,
                        )
                    )
        budget_loss = (
            torch.stack(budget_losses).mean() if budget_losses else zero
        )
        terms: dict[str, torch.Tensor] = {
            "average": contract_router_risk.mean(),
            "ranking": pairwise_logistic_ranking_loss(
                router_scores,
                targets,
                score_type=score_type,
            ),
            "budget": budget_loss,
            "calibration": (
                soft_brier_loss(
                    router_scores,
                    targets,
                    score_type=score_type,
                )
                if self.weights.calibration > 0
                else zero
            ),
            "stability": (
                stability_penalty
                if stability_penalty is not None
                else zero
            ),
            "compute": expected_compute_cost(
                expert_weights,
                expert_costs,
            ),
            "robust_regret": empirical_cvar(regrets, self.cvar_alpha),
            "abstention": effective_router_losses.mean(),
            "contract_router_risk": contract_router_risk,
            "contract_expert_risks": contract_expert_risks,
            "contract_availability": contract_availability.float(),
            "contract_feasible_oracle_risk": contract_oracle_risk,
            "bce_surrogate_contract_regret": regrets,
        }
        if self.include_instance_oracle_diagnostic:
            masked_example_expert_risks = torch.where(
                availability_mask.bool(),
                expert_losses,
                torch.full_like(expert_losses, torch.inf),
            )
            example_feasible = availability_mask.bool().any(dim=-1)
            instance_oracle = masked_example_expert_risks.min(dim=-1).values
            instance_oracle = torch.where(
                example_feasible,
                instance_oracle,
                instance_oracle.new_full(
                    instance_oracle.shape,
                    abstention_cost_value,
                ),
            )
            instance_by_contract, _ = _group_means(
                instance_oracle,
                group_indices,
            )
            terms["instance_clairvoyant_oracle_risk"] = instance_by_contract
        if abstention_capacity_by_group is not None:
            group_ids = {int(group.item()) for group in groups}
            if set(abstention_capacity_by_group) != group_ids:
                raise ValueError(
                    "abstention capacity map must cover every contract group exactly"
                )
            capacity_penalties = []
            for group in groups:
                group_id = int(group.item())
                capacity = abstention_capacity_by_group[group_id]
                if capacity is not None:
                    capacity_penalties.append(
                        abstention_capacity_penalty(
                            abstain_probability[group_indices == group],
                            capacity=capacity,
                        )
                    )
            if capacity_penalties:
                terms["abstention"] = terms["abstention"] + torch.stack(
                    capacity_penalties
                ).mean()
        total = (
            self.weights.average * terms["average"]
            + self.weights.ranking * terms["ranking"]
            + self.weights.robust_regret * terms["robust_regret"]
            + self.weights.budget * terms["budget"]
            + self.weights.stability * terms["stability"]
            + self.weights.compute * terms["compute"]
            + self.weights.calibration * terms["calibration"]
            + self.weights.abstention * terms["abstention"]
        )
        return total, terms
