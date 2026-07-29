from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from coregraph.contracts.axes import (
    BudgetSpec,
    ContractRole,
    ReviewMode,
)
from coregraph.experiments import pilot
from coregraph.objectives.composite import CompositeObjective, ObjectiveWeights
from coregraph.objectives.scores import ScoreType


def _prediction(
    scores: np.ndarray,
    *,
    abstain: np.ndarray | None = None,
    status: str = "EXECUTABLE",
    cost: float = 0.2,
    offline_oracle: bool = False,
    diagnostic_only: bool = False,
) -> pilot.BaselinePrediction:
    decision = (
        np.zeros(len(scores), dtype=bool)
        if abstain is None
        else np.asarray(abstain, dtype=bool)
    )
    return pilot.BaselinePrediction(
        scores=np.asarray(scores, dtype=float),
        abstention_probability=np.asarray([0.9, 0.8, 0.7, 0.6])[: len(scores)],
        abstain=decision,
        forced_abstention=np.zeros(len(scores), dtype=bool),
        expected_compute=np.ones(len(scores)),
        abstention_threshold=0.95,
        abstention_threshold_provenance="source_validation_balanced_contracts",
        abstention_capacity=1.0,
        abstention_cost=cost,
        execution_status=status,
        offline_oracle=offline_oracle or diagnostic_only,
        diagnostic_only=diagnostic_only,
    )


def _metric(result: dict[str, object], method: str, metric: str) -> float:
    rows = result["rows"]
    assert isinstance(rows, list)
    return float(
        next(
            row["value"]
            for row in rows
            if row["method"] == method and row["metric"] == metric
        )
    )


def _row(
    result: dict[str, object],
    method: str,
    metric: str,
) -> dict[str, object]:
    rows = result["rows"]
    assert isinstance(rows, list)
    return next(
        row
        for row in rows
        if row["method"] == method and row["metric"] == metric
    )


def test_evaluation_uses_frozen_abstention_decision_not_probability_cutoff() -> None:
    labels = np.asarray([1, 2, 1, 2])
    candidates = {
        "method": _prediction(np.asarray([0.9, 0.1, 0.8, 0.2])),
        "contract_feasible_oracle": _prediction(
            np.asarray([0.9, 0.1, 0.8, 0.2]),
            offline_oracle=True,
        ),
    }
    result = pilot.evaluate_saved_output_pilot(
        labels,
        candidates,
        dataset="elliptic",
        target_contract="strict_inductive",
        expert_prediction_seed=1,
        router_training_seeds={"method": 101},
        fold="fold0",
    )
    assert _metric(result, "method", "coverage") == 1.0
    assert _metric(result, "method", "selective_risk") == 0.0
    assert _row(result, "method", "coverage")["accepted_count"] == 4
    assert len(
        str(
            _row(result, "method", "coverage")[
                "abstention_decision_sha256"
            ]
        )
    ) == 64


def test_zero_coverage_is_undefined_and_pays_abstention_cost() -> None:
    labels = np.asarray([1, 2, 1, 2])
    candidates = {
        "zero_coverage": _prediction(
            np.asarray([0.9, 0.1, 0.8, 0.2]),
            abstain=np.ones(4, dtype=bool),
            status="ABSTAIN_ONLY",
            cost=0.3,
        ),
        "contract_feasible_oracle": _prediction(
            np.asarray([0.9, 0.1, 0.8, 0.2]),
            offline_oracle=True,
        ),
    }
    result = pilot.evaluate_saved_output_pilot(
        labels,
        candidates,
        dataset="elliptic",
        target_contract="strict_inductive",
        expert_prediction_seed=1,
        router_training_seeds={"zero_coverage": 101},
        fold="fold0",
    )
    assert np.isnan(_metric(result, "zero_coverage", "selective_risk"))
    assert _metric(result, "zero_coverage", "abstention_cost") == pytest.approx(0.3)


def test_contract_oracle_is_one_expert_and_instance_oracle_is_diagnostic() -> None:
    objective = CompositeObjective(
        ObjectiveWeights(robust_regret=1.0),
        include_instance_oracle_diagnostic=True,
    )
    _, terms = objective(
        router_scores=torch.tensor([0.9, 0.1]),
        score_type=ScoreType.PROBABILITY,
        targets=torch.tensor([1.0, 0.0]),
        group_indices=torch.tensor([0, 0]),
        expert_scores=torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        availability_mask=torch.ones((2, 2), dtype=torch.bool),
        expert_weights=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        expert_costs=torch.ones(2),
    )
    expected_contract_risk = float((-np.log(0.9) - np.log(0.1)) / 2)
    assert terms["contract_feasible_oracle_risk"].item() == pytest.approx(
        expected_contract_risk,
    )
    assert terms["instance_clairvoyant_oracle_risk"].item() == pytest.approx(
        -np.log(0.9),
    )
    assert (
        terms["contract_feasible_oracle_risk"]
        > terms["instance_clairvoyant_oracle_risk"]
    )

    scores = {
        "left": np.asarray([0.9, 0.9]),
        "right": np.asarray([0.1, 0.1]),
    }
    availability = {
        "left": np.ones(2, dtype=bool),
        "right": np.ones(2, dtype=bool),
    }
    contract = pilot.contract_feasible_oracle(
        target_scores=scores,
        target_availability=availability,
        target_expert_costs={"left": 1.0, "right": 1.0},
        target_labels=np.asarray([1, 2]),
    )
    instance = pilot.instance_clairvoyant_oracle_ceiling(
        target_scores=scores,
        target_availability=availability,
        target_expert_costs={"left": 1.0, "right": 1.0},
        target_labels=np.asarray([1, 2]),
    )
    assert len(np.unique(contract.scores)) == 1
    assert contract.offline_oracle
    assert not contract.diagnostic_only
    assert instance.scores.tolist() == [0.9, 0.1]
    assert instance.diagnostic_only


def test_review_budgets_are_derived_inside_each_contract(contract_factory) -> None:
    fraction_small = replace(
        contract_factory("small"),
        budget=BudgetSpec(
            review_mode=ReviewMode.FRACTION,
            review_fraction=0.5,
        ),
    )
    fraction_large = replace(
        contract_factory("large"),
        budget=BudgetSpec(
            review_mode=ReviewMode.FRACTION,
            review_fraction=0.5,
        ),
    )
    assert pilot.source_review_k_by_group(
        (fraction_small, fraction_large),
        np.asarray([0, 0, 1, 1, 1, 1]),
    ) == {0: 1, 1: 2}

    unconstrained = replace(
        contract_factory("unconstrained"),
        budget=BudgetSpec(review_mode=ReviewMode.UNCONSTRAINED_RANKING),
    )
    assert pilot.source_review_k_by_group(
        (fraction_small, unconstrained),
        np.asarray([0, 0, 1, 1]),
    ) == {0: 1, 1: None}

    fixed = replace(
        contract_factory("fixed"),
        budget=BudgetSpec(review_mode=ReviewMode.FIXED_K, fixed_k=1),
    )
    with pytest.raises(ValueError, match="mixed constrained review modes"):
        pilot.source_review_k_by_group(
            (fraction_small, fixed),
            np.asarray([0, 0, 1, 1]),
        )


def test_target_capacity_only_constrains_frozen_target_decision() -> None:
    probability = torch.tensor([0.9, 0.8, 0.7, 0.1])
    forced = torch.tensor([False, False, False, False])
    unconstrained = pilot.apply_frozen_abstention_decision(
        probability,
        threshold=0.6,
        capacity=None,
        forced_abstention=forced,
    )
    constrained = pilot.apply_frozen_abstention_decision(
        probability,
        threshold=0.6,
        capacity=0.25,
        forced_abstention=forced,
    )
    assert unconstrained.tolist() == [True, True, True, False]
    assert constrained.tolist() == [True, False, False, False]


def test_target_capacity_cannot_change_source_fit_or_threshold(
    contract_factory,
) -> None:
    labels = np.asarray([1, 2, 1, 2, 1, 2, 1, 2])
    splits = np.asarray(["train"] * 4 + ["validation"] * 4)
    source_groups = [
        pilot.SavedSourceGroup(
            contract=contract_factory(f"source_{index}"),
            scores={
                "feature": np.asarray(
                    [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3]
                ),
                "graph": np.asarray(
                    [0.7, 0.3, 0.6, 0.4, 0.6, 0.4, 0.8, 0.2]
                ),
            },
            labels=labels,
            splits=splits,
            availability={
                "feature": np.ones(8, dtype=bool),
                "graph": np.ones(8, dtype=bool),
            },
            expert_costs={"feature": 1.0, "graph": 2.0},
        )
        for index in range(2)
    ]
    target_scores = {
        "feature": np.asarray([0.8, 0.2, 0.7, 0.3]),
        "graph": np.asarray([0.6, 0.4, 0.9, 0.1]),
    }
    target_availability = {
        "feature": np.ones(4, dtype=bool),
        "graph": np.ones(4, dtype=bool),
    }

    def fit(capacity: float) -> pilot.SavedRouterPrediction:
        target = replace(
            contract_factory("target", role=ContractRole.TARGET),
            budget=BudgetSpec(
                review_mode=ReviewMode.FRACTION,
                review_fraction=0.01,
                abstention_capacity=capacity,
            ),
        )
        return pilot.fit_saved_output_corerouter(
            source_groups,
            target_contract=target,
            target_scores=target_scores,
            target_availability=target_availability,
            target_expert_costs={"feature": 1.0, "graph": 2.0},
            expert_prediction_seed=4,
            steps=1,
        )

    zero_capacity = fit(0.0)
    full_capacity = fit(1.0)
    assert zero_capacity.source_fit_hash == full_capacity.source_fit_hash
    assert zero_capacity.abstention_threshold == (
        full_capacity.abstention_threshold
    )
    assert zero_capacity.source_abstention_capacities == (
        full_capacity.source_abstention_capacities
    )
    assert zero_capacity.target_abstention_capacity == 0.0
    assert full_capacity.target_abstention_capacity == 1.0


def test_unavailable_single_expert_never_enters_ranking_metrics() -> None:
    labels = np.asarray([1, 2, 1, 2])
    blocked = _prediction(
        np.asarray([0.99, 0.01, 0.99, 0.01]),
        abstain=np.ones(4, dtype=bool),
        status="RESOURCE_BLOCKED",
    )
    result = pilot.evaluate_saved_output_pilot(
        labels,
        {
            "expert:blocked": blocked,
            "contract_feasible_oracle": _prediction(
                np.asarray([0.9, 0.1, 0.8, 0.2]),
                offline_oracle=True,
            ),
        },
        dataset="elliptic",
        target_contract="strict_inductive",
        expert_prediction_seed=1,
        router_training_seeds={"expert:blocked": 101},
        fold="fold0",
    )
    for metric in (
        "auprc",
        "recall_at_0.5pct",
        "recall_at_1pct",
        "recall_at_2pct",
        "budget_curve_area",
    ):
        assert np.isnan(_metric(result, "expert:blocked", metric))


def test_graphsafe_name_is_honest_and_mowst_threshold_is_source_fitted(
    contract_factory,
) -> None:
    labels = np.asarray([1, 2, 1, 2, 1, 2, 1, 2])
    splits = np.asarray(
        ["train"] * 4 + ["validation"] * 4,
    )
    sources = [
        pilot.SavedSourceGroup(
            contract=contract_factory(f"source_{index}"),
            scores={
                "feature_mlp": np.asarray(
                    [0.9, 0.1, 0.7, 0.3, 0.95, 0.05, 0.55, 0.45]
                ),
                "gcn": np.asarray(
                    [0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.9, 0.1]
                ),
            },
            labels=labels,
            splits=splits,
            availability={
                "feature_mlp": np.ones(8, dtype=bool),
                "gcn": np.ones(8, dtype=bool),
            },
            expert_costs={"feature_mlp": 1.0, "gcn": 2.0},
        )
        for index in range(2)
    ]
    target_contract = replace(
        contract_factory("target", role=ContractRole.TARGET),
        budget=BudgetSpec(
            review_mode=ReviewMode.FRACTION,
            review_fraction=0.01,
            abstention_capacity=0.5,
        ),
    )
    baselines = pilot.baseline_scores(
        sources,
        target_contract=target_contract,
        target_scores={
            "feature_mlp": np.asarray([0.9, 0.1, 0.6, 0.4]),
            "gcn": np.asarray([0.6, 0.4, 0.9, 0.1]),
        },
        target_availability={
            "feature_mlp": np.ones(4, dtype=bool),
            "gcn": np.ones(4, dtype=bool),
        },
        target_expert_costs={"feature_mlp": 1.0, "gcn": 2.0},
        expert_prediction_seed=3,
    )
    assert "graphsafe_v2_adapter" not in baselines
    assert "graphsafe_confidence_abstention_component" in baselines
    mowst = baselines["MOWST_INSPIRED_REIMPLEMENTATION"]
    assert mowst.details["routing_threshold_fitted_on"] == (
        "source_validation_balanced_contracts"
    )
    assert np.isfinite(mowst.details["routing_threshold"])
    assert mowst.details["routing_threshold"] != 0.25


def test_router_seed_is_method_specific_and_not_an_inferential_replication() -> None:
    first = pilot.derive_router_seed(7, "full_corerouter")
    repeated = pilot.derive_router_seed(7, "full_corerouter")
    ablation = pilot.derive_router_seed(7, "ablation:no_budget")
    assert first == repeated
    assert first != 7
    assert first != ablation
