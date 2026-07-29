from __future__ import annotations

import numpy as np
import pytest
import torch

from coregraph.contracts.axes import ContractRole, VisibilityAxis
from coregraph.experiments.pilot import SavedSourceGroup, fit_saved_output_corerouter
from coregraph.objectives.cvar import empirical_cvar, variational_cvar
from coregraph.objectives.scores import ScoreType
from coregraph.routing.diagnostics import validate_target_diagnostics
from coregraph.routing.router import CoReRouter
from coregraph.routing.stability import consistency_penalty, routing_flip_rate


def test_resource_mask_is_exact_and_all_missing_abstains() -> None:
    router = CoReRouter(num_experts=3, contract_dim=4, diagnostic_dim=2)
    output = router(
        expert_scores=torch.tensor([[0.1, 0.8, 0.3], [0.2, 0.4, 0.9]]),
        score_type=ScoreType.PROBABILITY,
        contract_embedding=torch.zeros((2, 4)),
        shared_diagnostics=torch.zeros((2, 2)),
        per_expert_diagnostics=torch.zeros((2, 3, 0)),
        availability_mask=torch.tensor([[True, False, True], [False, False, False]]),
        expert_costs=torch.tensor([1.0, 2.0, 3.0]),
    )
    assert output.expert_weights[0, 1].item() == 0
    assert output.expert_weights[0].sum().item() == pytest.approx(1.0)
    assert output.expert_weights[1].sum().item() == 0
    assert output.selected_expert.tolist()[1] == -1
    assert output.abstention_probability[1].item() == 1


def test_cvar_is_worst_tail_and_differentiable() -> None:
    losses = torch.tensor([1.0, 2.0, 10.0, 12.0], requires_grad=True)
    value = empirical_cvar(losses, alpha=0.5)
    assert value.item() == 11
    value.backward()
    assert losses.grad is not None
    eta = torch.tensor(2.0, requires_grad=True)
    variational_cvar(losses.detach(), eta, alpha=0.5).backward()
    assert eta.grad is not None


def test_stability_metrics() -> None:
    left = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    right = torch.tensor([[0.7, 0.3], [0.7, 0.3]])
    assert consistency_penalty(left, right).item() > 0
    assert routing_flip_rate(left, right) == 0.5


def test_saved_output_corerouter_never_receives_target_labels(contract_factory) -> None:
    labels = np.asarray([1, 2, 1, 2, 1, 2])
    splits = np.asarray(["train", "train", "train", "validation", "validation", "validation"])
    scores_a = {
        "feature": np.asarray([0.9, 0.2, 0.8, 0.3, 0.7, 0.1]),
        "graph": np.asarray([0.6, 0.4, 0.7, 0.2, 0.8, 0.3]),
    }
    scores_b = {
        "feature": np.asarray([0.7, 0.3, 0.8, 0.2, 0.6, 0.4]),
        "graph": np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
    }
    source = [
        SavedSourceGroup(
            contract=contract_factory("source_one"),
            scores=scores_a,
            labels=labels,
            splits=splits,
            availability={
                name: np.ones(len(labels), dtype=bool) for name in scores_a
            },
            expert_costs={"feature": 1.0, "graph": 2.0},
        ),
        SavedSourceGroup(
            contract=contract_factory(
                "source_two",
                visibility=VisibilityAxis.HISTORICAL_ONLY,
            ),
            scores=scores_b,
            labels=labels,
            splits=splits,
            availability={
                name: np.ones(len(labels), dtype=bool) for name in scores_b
            },
            expert_costs={"feature": 1.0, "graph": 2.0},
        ),
    ]
    target = contract_factory("target", role=ContractRole.TARGET)
    predicted = fit_saved_output_corerouter(
        source,
        target_contract=target,
        target_scores=scores_a,
        target_availability={
            name: np.ones(len(labels), dtype=bool) for name in scores_a
        },
        target_expert_costs={"feature": 1.0, "graph": 2.0},
        expert_prediction_seed=1,
        steps=2,
    )
    assert predicted.scores.shape == labels.shape
    assert np.all((predicted.scores >= 0) & (predicted.scores <= 1))
    assert predicted.routing_weights.shape == (len(labels), 2)


def test_label_requiring_diagnostic_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires labels"):
        validate_target_diagnostics(
            ("observed_target_error",),
            target_access_allowed=True,
        )
