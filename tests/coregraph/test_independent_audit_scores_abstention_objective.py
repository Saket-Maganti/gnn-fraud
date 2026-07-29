from __future__ import annotations

import numpy as np
import pytest
import torch

from coregraph.method import CoReGraph
from coregraph.objectives.calibration import soft_brier_loss
from coregraph.objectives.classification import binary_cross_entropy
from coregraph.objectives.composite import CompositeObjective, ObjectiveWeights
from coregraph.objectives.scores import ScoreType, probabilities
from coregraph.routing.abstention import (
    abstention_capacity_penalty,
    abstention_cost,
    area_under_risk_coverage_curve,
    coverage,
    select_abstention_threshold,
    selective_risk,
)
from coregraph.routing.diagnostics import score_diagnostics


def test_probability_and_logit_bce_have_exact_parity() -> None:
    logits = torch.tensor([-2.0, 0.0, 1.5, 4.0])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    probability_loss = binary_cross_entropy(
        torch.sigmoid(logits),
        targets,
        score_type=ScoreType.PROBABILITY,
    )
    logit_loss = binary_cross_entropy(
        logits,
        targets,
        score_type=ScoreType.LOGIT,
    )
    assert torch.allclose(probability_loss, logit_loss, atol=1e-7)


def test_incompatible_score_semantics_fail_closed() -> None:
    targets = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="probabilities"):
        binary_cross_entropy(
            torch.tensor([-0.1, 1.1]),
            targets,
            score_type=ScoreType.PROBABILITY,
        )
    with pytest.raises(ValueError, match="rank"):
        binary_cross_entropy(
            torch.tensor([1.0, 2.0]),
            targets,
            score_type=ScoreType.RANK_SCORE,
        )
    with pytest.raises(ValueError, match="rank"):
        soft_brier_loss(
            torch.tensor([1.0, 2.0]),
            targets,
            score_type=ScoreType.RANK_SCORE,
        )
    with pytest.raises(ValueError, match="rank"):
        score_diagnostics(
            np.asarray([[1.0, 2.0]]),
            score_type=ScoreType.RANK_SCORE,
        )
    assert torch.allclose(
        probabilities(torch.tensor([0.0]), ScoreType.LOGIT),
        torch.tensor([0.5]),
    )


def test_abstention_metrics_threshold_and_capacity_are_functional() -> None:
    losses = torch.tensor([0.9, 0.1, 0.8, 0.2])
    abstention_probability = torch.tensor([0.9, 0.1, 0.8, 0.2])
    selection = select_abstention_threshold(
        losses,
        abstention_probability,
        capacity=0.5,
    )
    abstain = abstention_probability >= selection.threshold
    assert coverage(abstain).item() == 0.5
    assert selective_risk(losses, abstain).item() == pytest.approx(0.15)
    assert 0 <= area_under_risk_coverage_curve(
        losses,
        abstention_probability,
    ).item() <= 1
    assert abstention_cost(abstain, cost=0.25).item() == 0.125
    assert abstention_capacity_penalty(
        abstention_probability,
        capacity=0.25,
    ).item() > 0


def test_composite_objective_derives_group_risks_and_backpropagates(
    contract_factory,
) -> None:
    torch.manual_seed(13)
    contracts = [contract_factory(f"source_{index // 3}") for index in range(6)]
    model = CoReGraph(
        num_experts=2,
        diagnostic_dim=2,
        per_expert_diagnostic_dim=1,
        axis_dropout=0.0,
        contract_noise_std=0.0,
    )
    expert_scores = torch.tensor(
        [
            [0.9, 0.6],
            [0.2, 0.4],
            [0.8, 0.7],
            [0.4, 0.9],
            [0.6, 0.1],
            [0.3, 0.8],
        ],
        requires_grad=True,
    )
    availability = torch.tensor(
        [
            [True, True],
            [True, True],
            [True, True],
            [True, False],
            [True, False],
            [True, False],
        ]
    )
    output = model(
        contracts=contracts,
        expert_scores=expert_scores,
        score_type=ScoreType.PROBABILITY,
        shared_diagnostics=torch.zeros((6, 2)),
        per_expert_diagnostics=torch.zeros((6, 2, 1)),
        availability_mask=availability,
        expert_costs=torch.tensor([1.0, 3.0]),
    )
    objective = CompositeObjective(
        ObjectiveWeights(
            average=1.0,
            ranking=0.1,
            robust_regret=1.0,
            budget=0.1,
            stability=0.1,
            compute=0.1,
            calibration=0.1,
            abstention=0.2,
        ),
        cvar_alpha=0.5,
    )
    total, terms = objective(
        router_scores=output.blended_score,
        score_type=output.score_type,
        targets=torch.tensor([1, 0, 1, 1, 0, 0]),
        group_indices=torch.tensor([0, 0, 0, 1, 1, 1]),
        expert_scores=expert_scores,
        availability_mask=availability,
        expert_weights=output.expert_weights,
        expert_costs=torch.tensor([1.0, 3.0]),
        stability_penalty=output.expert_weights.square().mean(),
        review_k=2,
        abstention_probability=output.abstention_probability,
        forced_abstention=~availability.any(dim=1),
        abstention_capacity=0.25,
        abstention_cost_value=0.2,
    )
    assert terms["contract_router_risk"].shape == (2,)
    assert terms["contract_expert_risks"].shape == (2, 2)
    assert torch.isfinite(total)
    total.backward()
    assert expert_scores.grad is not None
    assert torch.isfinite(expert_scores.grad).all()
    encoder_grads = [
        parameter.grad for parameter in model.contract_encoder.parameters()
    ]
    router_grads = [parameter.grad for parameter in model.router.parameters()]
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in encoder_grads)
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in router_grads)


def test_feasible_oracle_respects_example_level_availability() -> None:
    objective = CompositeObjective(ObjectiveWeights(robust_regret=1.0))
    _, terms = objective(
        router_scores=torch.tensor([0.9, 0.1]),
        score_type=ScoreType.PROBABILITY,
        targets=torch.tensor([1.0, 0.0]),
        group_indices=torch.tensor([0, 0]),
        expert_scores=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        availability_mask=torch.tensor([[True, False], [False, True]]),
        expert_weights=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        expert_costs=torch.tensor([1.0, 1.0]),
    )
    assert terms["feasible_oracle_risk"].item() == pytest.approx(
        -np.log(0.9),
    )
