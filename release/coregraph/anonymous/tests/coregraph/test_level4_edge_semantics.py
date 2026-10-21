from __future__ import annotations

import numpy as np
import pytest
import torch

from coregraph.contracts.interactions import bounded_pairwise_interactions
from coregraph.evaluation.selective import selective_metrics
from coregraph.objectives.abstention import abstention_objective
from coregraph.objectives.cvar import empirical_cvar, variational_cvar
from coregraph.objectives.ranking import pairwise_logistic_ranking_loss
from coregraph.objectives.regret import aggregate_regret, contract_regret, feasible_oracle_risk
from coregraph.objectives.scores import ScoreType
from coregraph.objectives.task import binary_cross_entropy
from coregraph.resources.profiles import ResourceProfile, standard_profiles
from coregraph.routing.contract_router import ContractRouter
from coregraph.theory.selective_risk import selective_risk_transfer_bound


def test_interaction_validation_and_single_axis_semantics() -> None:
    with pytest.raises(ValueError, match="shape"):
        bounded_pairwise_interactions(torch.ones(2, 3))
    with pytest.raises(ValueError, match="positive"):
        bounded_pairwise_interactions(torch.ones(2, 2, 3), bound=0)
    empty = bounded_pairwise_interactions(torch.ones(2, 1, 3))
    assert empty.shape == (2, 0)


def test_selective_metrics_declares_zero_coverage_and_alignment() -> None:
    with pytest.raises(ValueError, match="non-empty aligned"):
        selective_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="non-empty aligned"):
        selective_metrics(np.array([1.0]), np.array([True, False]))
    zero = selective_metrics(np.array([0.1, 0.9]), np.array([True, True]))
    assert zero["coverage"] == 0.0
    assert np.isnan(float(zero["selective_risk"]))
    measured = selective_metrics(np.array([0.1, 0.9]), np.array([False, True]))
    assert measured["selective_risk_status"] == "MEASURED"
    assert measured["selective_risk"] == pytest.approx(0.1)


def test_abstention_objective_rejects_invalid_inputs_and_penalises_coverage() -> None:
    with pytest.raises(ValueError, match="non-empty aligned"):
        abstention_objective(torch.tensor([]), torch.tensor([]), abstention_cost=0.1)
    with pytest.raises(ValueError, match="invalid"):
        abstention_objective(torch.tensor([1.0]), torch.tensor([0.2]), abstention_cost=-1)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        abstention_objective(torch.tensor([1.0]), torch.tensor([1.2]), abstention_cost=0.1)
    value, terms = abstention_objective(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.8, 0.8]),
        abstention_cost=0.1,
        coverage_floor=0.5,
    )
    assert value.isfinite() and terms["coverage_penalty"] > 0


def test_cvar_ranking_and_task_facade_edges() -> None:
    with pytest.raises(ValueError, match="alpha"):
        empirical_cvar(torch.tensor([1.0]), alpha=1)
    with pytest.raises(ValueError, match="empty"):
        empirical_cvar(torch.tensor([]))
    with pytest.raises(ValueError, match="alpha"):
        variational_cvar(torch.tensor([1.0]), torch.tensor(0.0), alpha=-0.1)
    assert empirical_cvar(torch.tensor([1.0, 3.0]), alpha=0.5) == pytest.approx(3.0)
    assert variational_cvar(torch.tensor([1.0]), torch.tensor(0.0), alpha=0.5).isfinite()
    no_pairs = pairwise_logistic_ranking_loss(
        torch.tensor([0.2, 0.3]),
        torch.tensor([1, 1]),
        score_type=ScoreType.PROBABILITY,
    )
    assert no_pairs == 0
    sampled = pairwise_logistic_ranking_loss(
        torch.linspace(0, 1, 20),
        torch.tensor([0, 1] * 10),
        score_type=ScoreType.PROBABILITY,
        max_pairs=4,
    )
    assert sampled.isfinite()
    assert binary_cross_entropy(
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.0, 1.0]),
        score_type=ScoreType.PROBABILITY,
    ).isfinite()


def test_regret_validation_and_aggregation() -> None:
    risks = torch.tensor([[0.4, 0.2]])
    mask = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="align"):
        feasible_oracle_risk(risks, torch.tensor([[True]]))
    with pytest.raises(ValueError, match="at least one"):
        feasible_oracle_risk(risks, torch.tensor([[False, False]]))
    assert feasible_oracle_risk(risks, mask) == pytest.approx(torch.tensor([0.4]))
    with pytest.raises(ValueError, match="one value"):
        contract_regret(torch.tensor([0.5, 0.6]), risks, mask)
    regrets = contract_regret(torch.tensor([0.5]), risks, mask)
    summary = aggregate_regret(regrets)
    assert summary["maximum"] == pytest.approx(torch.tensor(0.1))
    with pytest.raises(ValueError, match="empty"):
        aggregate_regret(torch.tensor([]))


def test_resource_profiles_and_contract_router_validation() -> None:
    profiles = standard_profiles()
    assert len(profiles) == 8 and profiles[-1].dynamic
    with pytest.raises(ValueError, match="memory"):
        ResourceProfile("bad", memory_cap_gb=0)
    with pytest.raises(ValueError, match="latency"):
        ResourceProfile("bad", latency_cap_ms=-1)
    with pytest.raises(ValueError, match="review"):
        ResourceProfile("bad", review_budget_fraction=1.1)
    with pytest.raises(ValueError, match="dimensions"):
        ContractRouter(0, 1, 2)
    router = ContractRouter(2, 1, 2)
    with pytest.raises(ValueError, match="matrices"):
        router(torch.ones(2), torch.ones(1, 1), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="align"):
        router(torch.ones(2, 2), torch.ones(1, 1), torch.ones(2, 2, dtype=torch.bool))


def test_selective_transfer_bound_rejects_invalid_assumptions() -> None:
    with pytest.raises(ValueError, match="risk"):
        selective_risk_transfer_bound(
            source_selective_risk=2,
            source_coverage=0.5,
            target_coverage_lower_bound=0.5,
            density_ratio_bound=1,
        )
    with pytest.raises(ValueError, match="coverage"):
        selective_risk_transfer_bound(
            source_selective_risk=0.2,
            source_coverage=0.5,
            target_coverage_lower_bound=0,
            density_ratio_bound=1,
        )
    with pytest.raises(ValueError, match="positive"):
        selective_risk_transfer_bound(
            source_selective_risk=0.2,
            source_coverage=0.5,
            target_coverage_lower_bound=0.5,
            density_ratio_bound=-1,
        )
    assert selective_risk_transfer_bound(
        source_selective_risk=0.2,
        source_coverage=0.5,
        target_coverage_lower_bound=0.25,
        density_ratio_bound=2,
    ) == pytest.approx(0.8)
