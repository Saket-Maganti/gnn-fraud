from __future__ import annotations

import numpy as np
import pytest

from coregraph.evaluation.corrections import holm
from coregraph.evaluation.statistics import (
    aggregate_contexts_within_seed,
    bootstrap_seed_blocks,
    exact_wilcoxon,
    paired_effect_size,
    paired_permutation,
)
from scripts.coregraph.evaluate_pilot_gate import (
    REQUIRED_ABLATIONS,
    _derive_regret_rows,
    _holm_family,
    _validate_gate_schema,
    _worst_contract_seed_deltas,
)
from scripts.coregraph.run_statistical_analysis import _seed_block_values


def _paired_rows() -> list[dict[str, str]]:
    return [
        {
            "dataset": "fixture",
            "target_contract": contract,
            "seed": str(seed),
            "fold": "fold0",
            "method_value": str(0.8 + 0.01 * seed),
            "baseline_value": str(
                0.6 + 0.01 * seed + (0.05 if contract == "b" else 0.0)
            ),
        }
        for seed in (1, 2, 3)
        for contract in ("a", "b")
    ]


def test_seed_blocks_aggregate_contracts_within_seed() -> None:
    seeds, method, baseline, contexts = _seed_block_values(_paired_rows())
    assert seeds == [1, 2, 3]
    assert contexts == [2, 2, 2]
    assert np.asarray(method) - np.asarray(baseline) == pytest.approx(
        [0.175, 0.175, 0.175]
    )
    with pytest.raises(ValueError, match="duplicate"):
        _seed_block_values([*_paired_rows(), _paired_rows()[0]])
    with pytest.raises(ValueError, match="unequal"):
        _seed_block_values(_paired_rows()[:-1])


def test_declared_inference_stack_operates_on_seed_blocks() -> None:
    improvement = np.asarray([0.1, 0.2, 0.15, 0.12])
    zeros = np.zeros_like(improvement)
    assert exact_wilcoxon(improvement, zeros, alternative="greater").n_blocks == 4
    assert paired_permutation(
        improvement,
        zeros,
        alternative="greater",
    ).p_value <= 0.125
    low, high = bootstrap_seed_blocks(improvement, n_bootstrap=200)
    assert 0 < low <= high
    correction = holm([0.01, 0.02, 0.5], alpha=0.05)
    assert correction.reject == (True, True, False)
    assert _holm_family("auprc") == "ranking_and_budget"
    assert _holm_family("maximum_regret") == "robust_risk"
    assert _holm_family("compute") == "deployment"
    assert len(REQUIRED_ABLATIONS) == 8
    _validate_gate_schema(
        {
            "required_strong_baselines": [
                "average_all_feasible",
                "best_source_validation",
                "source_validation_convex_mixture",
                "graphsafe_confidence_abstention_component",
                "current_graph_feature_gate_adapter",
                "learned_no_contract_router",
                "learned_atomic_contract_router",
                "MOWST_INSPIRED_REIMPLEMENTATION",
            ],
            "schema_version": "coregraph_pilot_gate_v3",
            "required_datasets": ["elliptic", "dgraphfin"],
            "required_target_contracts": [
                "strict_inductive",
                "isolated_inductive",
                "transductive_structure",
            ],
            "required_expert_prediction_seeds": list(range(1, 11)),
            "required_folds": ["fold0"],
            "required_experts": ["feature_mlp", "gcn", "graphsage"],
            "required_ablations": [
                name.removeprefix("ablation:")
                for name in sorted(REQUIRED_ABLATIONS)
            ],
            "required_contract_metrics": [
                "auprc",
                "recall_at_0.5pct",
                "recall_at_1pct",
                "recall_at_2pct",
                "budget_curve_area",
                "contract_regret",
                "selective_risk",
                "coverage",
                "aurc",
                "abstention_cost",
                "compute",
            ],
            "holm_families": {
                "ranking_and_budget": [
                    "auprc",
                    "recall_at_0.5pct",
                    "recall_at_1pct",
                    "recall_at_2pct",
                    "budget_curve_area",
                ],
                "robust_risk": [
                    "mean_regret",
                    "maximum_regret",
                    "cvar_regret",
                ],
                "deployment": [
                    "selective_risk",
                    "abstention_cost",
                    "compute",
                ],
            },
            "ablation_tests": {
                name.removeprefix("ablation:"): {
                    "metric": "auprc",
                    "direction": "higher",
                    "minimum_effect": 0.0,
                    "required_contribution": False,
                }
                for name in REQUIRED_ABLATIONS
            },
            "required_meaningful_ablations": [
                "no_contract",
                "atomic_contract",
                "no_regret",
                "no_budget",
                "no_resource_mask",
            ],
            "effect_thresholds": {
                "worst_contract_regret_improvement": 0.001,
                "robust_primary_improvement": 0.001,
            },
        }
    )


def test_inference_guards_and_seed_aggregation_fail_closed() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        exact_wilcoxon([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="non-finite"):
        exact_wilcoxon([np.nan], [0.0])
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation([0.1, 0.2], [0.0, 0.0], alternative="invalid")
    with pytest.raises(ValueError, match="finite vector"):
        bootstrap_seed_blocks([0.1, np.inf])
    assert bootstrap_seed_blocks([0.25]) == (0.25, 0.25)
    assert paired_effect_size([0.25]) == 0.0
    assert aggregate_contexts_within_seed(
        [
            {"expert_prediction_seed": 2, "value": 0.3},
            {"expert_prediction_seed": 2, "value": 0.5},
            {"expert_prediction_seed": 1, "value": 0.1},
        ],
        seed_key="expert_prediction_seed",
    ) == {1: pytest.approx(0.1), 2: pytest.approx(0.4)}


def test_regret_and_worst_contract_use_paired_target_outcomes() -> None:
    rows: list[dict[str, object]] = []
    for seed in (1, 2):
        for contract, full, baseline, regret in (
            ("a", 0.8, 0.7, 0.1),
            ("b", 0.65, 0.6, 0.3),
        ):
            rows.extend(
                [
                    {
                        "dataset": "fixture",
                        "target_contract": contract,
                        "seed": seed,
                        "fold": "fold0",
                        "method": "full_corerouter",
                        "metric": "auprc",
                        "value": full,
                    },
                    {
                        "dataset": "fixture",
                        "target_contract": contract,
                        "seed": seed,
                        "fold": "fold0",
                        "method": "baseline",
                        "metric": "auprc",
                        "value": baseline,
                    },
                    {
                        "dataset": "fixture",
                        "target_contract": contract,
                        "seed": seed,
                        "fold": "fold0",
                        "method": "full_corerouter",
                        "metric": "contract_regret",
                        "value": regret,
                    },
                ]
            )
    deltas = _worst_contract_seed_deltas(rows, "baseline")
    assert deltas == pytest.approx((0.05, 0.05))
    derived = _derive_regret_rows(rows)
    assert {row["metric"] for row in derived} == {
        "mean_regret",
        "maximum_regret",
        "cvar_regret",
    }
    assert max(
        float(row["value"])
        for row in derived
        if row["metric"] == "maximum_regret"
    ) == pytest.approx(0.3)
