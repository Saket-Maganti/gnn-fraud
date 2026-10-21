from __future__ import annotations

import pytest
import torch

from coregraph.objectives.scores import ScoreType
from coregraph.routing.router import CoReRouter


def _inputs(batch: int = 2, experts: int = 3):
    return {
        "expert_scores": torch.full((batch, experts), 0.5),
        "score_type": ScoreType.PROBABILITY,
        "contract_embedding": torch.zeros((batch, 2)),
        "shared_diagnostics": torch.zeros((batch, 1)),
        "per_expert_diagnostics": torch.zeros((batch, experts, 1)),
        "availability_mask": torch.ones((batch, experts), dtype=torch.bool),
        "expert_costs": torch.arange(1, experts + 1, dtype=torch.float32),
    }


def test_equal_scores_remain_identity_distinguishable() -> None:
    router = CoReRouter(
        num_experts=2,
        contract_dim=2,
        diagnostic_dim=1,
        per_expert_diagnostic_dim=1,
        expert_identity_dim=2,
        mode="linear",
    )
    with torch.no_grad():
        router.expert_identity.weight.copy_(
            torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
        )
        assert isinstance(router.scorer, torch.nn.Linear)
        router.scorer.weight.zero_()
        router.scorer.bias.zero_()
        router.scorer.weight[0, router.identity_token_offset] = 1.0
    args = _inputs(experts=2)
    output = router(**args)
    assert torch.all(output.expert_weights[:, 0] > output.expert_weights[:, 1])


def test_router_is_permutation_equivariant_when_identity_moves_with_expert() -> None:
    torch.manual_seed(7)
    router = CoReRouter(
        num_experts=3,
        contract_dim=2,
        diagnostic_dim=1,
        per_expert_diagnostic_dim=1,
        expert_identity_dim=3,
        expert_family_dim=2,
        num_expert_families=2,
        mode="mlp",
    )
    args = _inputs(experts=3)
    args["expert_scores"] = torch.tensor([[0.2, 0.7, 0.4], [0.5, 0.1, 0.9]])
    args["per_expert_diagnostics"] = torch.arange(6, dtype=torch.float32).reshape(
        2, 3, 1
    )
    args["expert_identity_indices"] = torch.tensor([0, 1, 2])
    args["expert_family_indices"] = torch.tensor([0, 1, 0])
    direct = router(**args)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    permuted_args = {
        **args,
        "expert_scores": args["expert_scores"][:, permutation],
        "per_expert_diagnostics": args["per_expert_diagnostics"][:, permutation],
        "availability_mask": args["availability_mask"][:, permutation],
        "expert_costs": args["expert_costs"][permutation],
        "expert_identity_indices": args["expert_identity_indices"][permutation],
        "expert_family_indices": args["expert_family_indices"][permutation],
    }
    permuted = router(**permuted_args)
    assert torch.allclose(
        direct.expert_weights,
        permuted.expert_weights[:, inverse],
        atol=1e-6,
    )
    assert torch.allclose(direct.blended_score, permuted.blended_score, atol=1e-6)


def test_per_expert_cost_is_an_input_to_routing() -> None:
    router = CoReRouter(
        num_experts=2,
        contract_dim=2,
        diagnostic_dim=1,
        per_expert_diagnostic_dim=1,
        expert_identity_dim=0,
        mode="linear",
    )
    with torch.no_grad():
        assert isinstance(router.scorer, torch.nn.Linear)
        router.scorer.weight.zero_()
        router.scorer.bias.zero_()
        router.scorer.weight[0, router.cost_token_offset] = -1.0
    args = _inputs(experts=2)
    output = router(**args)
    assert torch.all(output.expert_weights[:, 0] > output.expert_weights[:, 1])


@pytest.mark.parametrize("mode", ["linear", "mlp", "attention"])
def test_all_unavailable_rows_are_safe_and_differentiable(mode: str) -> None:
    torch.manual_seed(11)
    router = CoReRouter(
        num_experts=3,
        contract_dim=2,
        diagnostic_dim=1,
        per_expert_diagnostic_dim=1,
        mode=mode,
        unavailable_score_sentinel=-7.0,
    )
    args = _inputs(experts=3)
    args["expert_scores"] = torch.tensor(
        [[0.1, 0.7, 0.3], [0.4, 0.5, 0.6]],
        requires_grad=True,
    )
    args["contract_embedding"] = torch.zeros((2, 2), requires_grad=True)
    args["shared_diagnostics"] = torch.zeros((2, 1), requires_grad=True)
    args["per_expert_diagnostics"] = torch.zeros(
        (2, 3, 1),
        requires_grad=True,
    )
    args["availability_mask"] = torch.tensor(
        [[True, False, True], [False, False, False]]
    )
    output = router(**args)
    assert torch.equal(output.expert_weights[1], torch.zeros(3))
    assert output.selected_expert[1].item() == -1
    assert output.abstention_probability[1].item() == 1.0
    assert output.blended_score[1].item() == -7.0
    assert output.routing_entropy[1].item() == 0.0
    assert output.expected_compute[1].item() == 0.0
    for value in (
        output.expert_weights,
        output.blended_score,
        output.abstention_probability,
        output.routing_entropy,
        output.expected_compute,
    ):
        assert torch.isfinite(value).all()
    (
        output.blended_score.sum()
        + output.routing_entropy.sum()
        + output.expected_compute.sum()
        + output.abstention_probability.sum()
    ).backward()
    for tensor in (
        args["expert_scores"],
        args["contract_embedding"],
        args["shared_diagnostics"],
        args["per_expert_diagnostics"],
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
