from __future__ import annotations

import numpy as np
import pytest
import torch

from coregraph.evaluation.calibration import (
    IsotonicCalibrationAdapter,
    bootstrap_calibration_interval,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    logistic_calibration_slope_intercept,
)
from coregraph.evaluation.metrics import select_budget_cutoff, select_threshold_on_validation
from coregraph.evaluation.regret import contract_regrets, regret_summary
from coregraph.experts.official_adapters.robust import group_dro_loss, irm_penalty, vrex_loss
from coregraph.objectives.budget import soft_precision_at_k_loss, soft_topk_weights
from coregraph.objectives.classification import (
    binary_cross_entropy,
    class_balanced_loss,
    focal_loss,
)
from coregraph.objectives.compute import CostProvenance, ExpertCost, expected_compute_cost
from coregraph.objectives.ranking import pairwise_logistic_ranking_loss
from coregraph.objectives.regret import contract_regret, feasible_oracle_risk
from coregraph.routing.abstention import apply_abstention_capacity, selective_risk
from coregraph.theory.compositional_bound import compositional_error_bound
from coregraph.theory.fixed_mixture import fixed_mixture_lower_bound
from coregraph.theory.numerical_checks import run_numerical_checks
from coregraph.theory.resource_mask import resource_mask_monotonicity


def test_calibration_suite_and_degenerate_guard() -> None:
    labels = np.asarray([1, 2, 1, 2, 1, 2, 1, 2])
    probabilities = np.asarray([0.9, 0.1, 0.8, 0.2, 0.75, 0.25, 0.7, 0.3])
    calibration = logistic_calibration_slope_intercept(labels, probabilities)
    assert calibration.status == "FIT"
    assert calibration.slope > 0
    assert 0 <= brier_score(labels, probabilities) <= 1
    assert 0 <= expected_calibration_error(labels, probabilities, adaptive=True) <= 1
    logits = np.log(probabilities / (1 - probabilities))
    assert fit_temperature(logits, labels).temperature > 0
    isotonic = IsotonicCalibrationAdapter().fit(labels, probabilities)
    transformed = isotonic.transform([0.1, 0.9])
    assert np.all((transformed >= 0) & (transformed <= 1))
    low, high = bootstrap_calibration_interval(
        labels,
        probabilities,
        n_bootstrap=100,
        seed=3,
    )
    assert low <= high


def test_threshold_and_budget_selection_are_separate() -> None:
    labels = [1, 2, 1, 2]
    scores = [0.9, 0.8, 0.7, 0.1]
    selected = select_threshold_on_validation(labels, scores, objective="f1")
    assert 0 <= selected.threshold <= 1
    with pytest.raises(ValueError, match="threshold-free"):
        select_threshold_on_validation(labels, scores, objective="auprc")
    cutoff, k = select_budget_cutoff(scores, budget=0.5)
    assert k == 2 and cutoff == 0.8


def test_predictive_and_robust_objectives_have_gradients() -> None:
    logits = torch.tensor([1.0, -1.0, 0.5, -0.5], requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0])
    losses = (
        binary_cross_entropy(logits, targets)
        + focal_loss(logits, targets)
        + class_balanced_loss(logits, targets)
        + pairwise_logistic_ranking_loss(logits, targets)
        + soft_precision_at_k_loss(logits, targets, 2)
    )
    losses.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(soft_topk_weights(logits.detach(), 2).sum(), torch.tensor(2.0), atol=1e-4)


def test_regret_compute_and_abstention_semantics() -> None:
    expert_risks = torch.tensor([[0.2, 0.1], [0.3, 0.4]])
    available = torch.tensor([[True, False], [True, True]])
    oracle = feasible_oracle_risk(expert_risks, available)
    assert oracle.tolist() == pytest.approx([0.2, 0.3])
    regrets = contract_regret(torch.tensor([0.25, 0.35]), expert_risks, available)
    assert regrets.tolist() == pytest.approx([0.05, 0.05])
    cost = expected_compute_cost(
        torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        torch.tensor([1.0, 3.0]),
    )
    assert cost.item() == 1.5
    declared = ExpertCost("feature", 1.0, 0.1, None, CostProvenance.MEASURED)
    assert declared.provenance is CostProvenance.MEASURED
    abstain = apply_abstention_capacity(torch.tensor([0.1, 0.9, 0.8]), 1 / 3)
    assert abstain.tolist() == [False, True, False]
    assert selective_risk(torch.tensor([1.0, 2.0, 3.0]), abstain).item() == 2.0


def test_numpy_regret_and_robust_source_losses() -> None:
    regrets = contract_regrets(
        {"a": 0.3, "b": 0.4},
        {"a": {"x": 0.2, "y": 0.1}, "b": {"x": 0.5, "y": 0.3}},
        {"a": {"x": True, "y": False}, "b": {"x": True, "y": True}},
    )
    assert regrets == pytest.approx({"a": 0.1, "b": 0.1})
    assert regret_summary(regrets)["maximum_contract_regret"] == pytest.approx(0.1)
    group_losses = torch.tensor([0.2, 0.5], requires_grad=True)
    dro, updated = group_dro_loss(group_losses, torch.zeros(2))
    assert updated.shape == (2,)
    assert vrex_loss(group_losses) >= group_losses.mean()
    scale = torch.tensor(1.0, requires_grad=True)
    assert irm_penalty(group_losses * scale, scale) >= 0
    dro.backward()
    assert group_losses.grad is not None


def test_theory_statuses_and_failure_guards() -> None:
    assert run_numerical_checks()["all_pass"]
    fixed = fixed_mixture_lower_bound(0.3, 0.5)
    assert fixed.proof_status == "PROVED"
    bound = compositional_error_bound(
        [0.1, 0.2],
        interaction_residual=0.05,
        router_approximation_error=0.02,
    )
    assert bound.total_bound == pytest.approx(0.37)
    assert resource_mask_monotonicity(
        np.asarray([0.1, 0.2]),
        np.asarray([True, True]),
        np.asarray([False, True]),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        fixed_mixture_lower_bound(0.0, 0.5)
