from __future__ import annotations

import csv
import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coregraph.contracts.axes import ContractRole
from coregraph.evaluation.statistics import build_paired_seed_blocks
from coregraph.experiments.pilot import (
    PilotAblation,
    PredictionArtifact,
    SavedSourceGroup,
    align_artifact_group,
    baseline_scores,
    contract_feasible_oracle,
    fit_saved_output_corerouter,
    instance_clairvoyant_oracle_ceiling,
    validate_artifact_groups,
)


def _artifact(
    contract,
    *,
    expert_id: str,
    seed: int,
    path: Path,
    alias_of: str = "",
) -> PredictionArtifact:
    return PredictionArtifact(
        expert_id=expert_id,
        expert_alias_of=alias_of,
        dataset="fixture",
        task="node_classification",
        prediction_unit="node",
        contract_coordinate_hash=contract.coordinate_hash,
        environment_id=contract.environment_id,
        seed=seed,
        fold="fold0",
        path=path,
        checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        config_hash="a" * 64,
        code_hash="b" * 40,
        contract_role=contract.role.value,
        deployment_contract=contract,
    )


def _write_predictions(
    path: Path,
    *,
    expert: str,
    ids: tuple[str, ...] = ("node:1", "node:2"),
    labels: tuple[int, ...] = (1, 2),
    splits: tuple[str, ...] = ("validation", "test"),
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "node_id",
                "score",
                "y_true",
                "split",
                "expert_id",
            ),
        )
        writer.writeheader()
        for index, identifier in enumerate(ids):
            writer.writerow(
                {
                    "node_id": identifier,
                    "score": 0.8 - 0.5 * index,
                    "y_true": labels[index],
                    "split": splits[index],
                    "expert_id": expert,
                }
            )


def test_manifest_grouping_is_seed_fold_and_expert_exact(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory()
    paths = []
    for seed in (1, 2):
        for expert in ("feature", "graph"):
            path = tmp_path / f"{seed}_{expert}.csv"
            _write_predictions(path, expert=expert)
            paths.append(
                _artifact(
                    contract,
                    expert_id=expert,
                    seed=seed,
                    path=path,
                )
            )
    groups = validate_artifact_groups(
        paths,
        expected_experts=("feature", "graph"),
        expected_seeds=(1, 2),
    )
    assert len(groups) == 2
    other_contract = replace(contract, environment_id="other_environment")
    other_path = tmp_path / "other_feature.csv"
    _write_predictions(other_path, expert="feature")
    other_artifact = _artifact(
        other_contract,
        expert_id="feature",
        seed=1,
        path=other_path,
    )
    assert other_artifact.group_key != paths[0].group_key
    with pytest.raises(ValueError, match="environment ID"):
        replace(paths[0], environment_id="wrong_environment")
    with pytest.raises(ValueError, match="config hash"):
        replace(paths[0], config_hash="not-a-hash")
    with pytest.raises(ValueError, match="available reason"):
        replace(
            paths[0],
            expert_available=False,
            availability_reason_codes=("available",),
        )
    with pytest.raises(ValueError, match="missing seeds"):
        validate_artifact_groups(
            [artifact for artifact in paths if artifact.seed == 1],
            expected_experts=("feature", "graph"),
            expected_seeds=(1, 2),
        )
    with pytest.raises(ValueError, match="duplicate expert-seed"):
        validate_artifact_groups(
            [*paths, paths[0]],
            expected_experts=("feature", "graph"),
            expected_seeds=(1, 2),
        )
    aliased = replace(
        paths[1],
        expert_id="graph_alias",
        expert_alias_of="feature",
    )
    with pytest.raises(ValueError, match="alias"):
        validate_artifact_groups(
            [paths[0], aliased],
            expected_experts=("feature",),
            expected_seeds=(1,),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"ids": ("node:1", "node:3")}, "identifier"),
        ({"labels": (2, 2)}, "label mismatch"),
        ({"splits": ("train", "test")}, "split mismatch"),
    ],
)
def test_prediction_alignment_rejects_id_label_and_split_mismatch(
    tmp_path: Path,
    contract_factory,
    mutation,
    message,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_predictions(first, expert="feature")
    _write_predictions(second, expert="graph", **mutation)
    artifacts = [
        _artifact(
            contract_factory(),
            expert_id="feature",
            seed=1,
            path=first,
        ),
        _artifact(
            contract_factory(),
            expert_id="graph",
            seed=1,
            path=second,
        ),
    ]
    with pytest.raises(ValueError, match=message):
        align_artifact_group(artifacts)


def _source_group(contract, feature, graph) -> SavedSourceGroup:
    labels = np.asarray([1, 2, 1, 2, 1, 2, 1, 2])
    splits = np.asarray(
        ["train", "train", "train", "train", "validation", "validation", "validation", "validation"]
    )
    return SavedSourceGroup(
        contract=contract,
        scores={"feature": np.asarray(feature), "graph": np.asarray(graph)},
        labels=labels,
        splits=splits,
        availability={
            "feature": np.ones(8, dtype=bool),
            "graph": np.ones(8, dtype=bool),
        },
        expert_costs={"feature": 1.0, "graph": 3.0},
    )


def test_baseline_registry_contains_honest_distinct_methods(
    contract_factory,
) -> None:
    assert not any(
        "label" in name
        for name in inspect.signature(baseline_scores).parameters
    )
    sources = [
        _source_group(
            contract_factory("source_a"),
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
            [0.7, 0.3, 0.6, 0.4, 0.6, 0.4, 0.55, 0.45],
        ),
        _source_group(
            contract_factory("source_b"),
            [0.6, 0.4, 0.55, 0.45, 0.6, 0.4, 0.55, 0.45],
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
        ),
    ]
    target = {
        "feature": np.asarray([0.8, 0.2, 0.6, 0.4]),
        "graph": np.asarray([0.6, 0.4, 0.9, 0.1]),
    }
    baselines = baseline_scores(
        sources,
        target_contract=contract_factory(
            "target",
            role=ContractRole.TARGET,
        ),
        target_scores=target,
        target_availability={
            "feature": np.ones(4, dtype=bool),
            "graph": np.ones(4, dtype=bool),
        },
        target_expert_costs={"feature": 1.0, "graph": 3.0},
        expert_prediction_seed=1,
    )
    baselines["contract_feasible_oracle"] = contract_feasible_oracle(
        target_scores=target,
        target_availability={
            "feature": np.ones(4, dtype=bool),
            "graph": np.ones(4, dtype=bool),
        },
        target_expert_costs={"feature": 1.0, "graph": 3.0},
        target_labels=np.asarray([1, 2, 1, 2]),
    )
    baselines["instance_clairvoyant_oracle_ceiling"] = (
        instance_clairvoyant_oracle_ceiling(
            target_scores=target,
            target_availability={
                "feature": np.ones(4, dtype=bool),
                "graph": np.ones(4, dtype=bool),
            },
            target_expert_costs={"feature": 1.0, "graph": 3.0},
            target_labels=np.asarray([1, 2, 1, 2]),
        )
    )
    expected = {
        "expert:feature",
        "expert:graph",
        "average_all_feasible",
        "best_source_validation",
        "source_validation_convex_mixture",
        "graphsafe_confidence_abstention_component",
        "current_graph_feature_gate_adapter",
        "learned_no_contract_router",
        "learned_atomic_contract_router",
        "MOWST_INSPIRED_REIMPLEMENTATION",
        "contract_feasible_oracle",
        "instance_clairvoyant_oracle_ceiling",
    }
    assert expected <= set(baselines)
    component = baselines["graphsafe_confidence_abstention_component"]
    assert (
        component.details["parity_status"]
        == "PARTIAL_CONFIDENCE_COMPONENT_NOT_FULL_GRAPHSAFE"
    )
    assert (
        baselines["current_graph_feature_gate_adapter"].adapter
        == "models.graph_feature_gating.GraphFeatureGate"
    )
    assert baselines["learned_no_contract_router"].learned
    assert baselines["learned_atomic_contract_router"].learned
    assert baselines["contract_feasible_oracle"].offline_oracle
    assert baselines["instance_clairvoyant_oracle_ceiling"].diagnostic_only


def test_every_pilot_baseline_respects_all_unavailable_rows(
    contract_factory,
) -> None:
    sources = [
        _source_group(
            contract_factory("source_a"),
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
            [0.7, 0.3, 0.6, 0.4, 0.6, 0.4, 0.55, 0.45],
        ),
        _source_group(
            contract_factory("source_b"),
            [0.6, 0.4, 0.55, 0.45, 0.6, 0.4, 0.55, 0.45],
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
        ),
    ]
    target = {
        "feature": np.asarray([0.8, 0.2, 0.6, 0.4]),
        "graph": np.asarray([0.6, 0.4, 0.9, 0.1]),
    }
    availability = {
        "feature": np.asarray([False, True, True, True]),
        "graph": np.asarray([False, True, True, True]),
    }
    baselines = baseline_scores(
        sources,
        target_contract=contract_factory(
            "target",
            role=ContractRole.TARGET,
        ),
        target_scores=target,
        target_availability=availability,
        target_expert_costs={"feature": 1.0, "graph": 3.0},
        expert_prediction_seed=1,
    )
    for prediction in baselines.values():
        assert prediction.abstention_probability[0] >= 0.5
        assert prediction.expected_compute[0] == 0


def test_source_only_pilot_uses_real_masks_and_declared_ablation(
    contract_factory,
) -> None:
    sources = [
        _source_group(
            contract_factory("source_a"),
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
            [0.6, 0.4, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3],
        ),
        _source_group(
            contract_factory("source_b"),
            [0.6, 0.4, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3],
            [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
        ),
    ]
    target_scores = {
        "feature": np.asarray([0.8, 0.2, 0.7, 0.3]),
        "graph": np.asarray([0.6, 0.4, 0.9, 0.1]),
    }
    availability = {
        "feature": np.ones(4, dtype=bool),
        "graph": np.asarray([True, False, True, False]),
    }
    prediction = fit_saved_output_corerouter(
        sources,
        target_contract=contract_factory("target", role=ContractRole.TARGET),
        target_scores=target_scores,
        target_availability=availability,
        target_expert_costs={"feature": 1.0, "graph": 3.0},
        expert_prediction_seed=1,
        steps=3,
        ablation=PilotAblation.NO_BUDGET,
    )
    assert prediction.ablation is PilotAblation.NO_BUDGET
    assert prediction.source_train_examples == 8
    assert prediction.source_validation_examples == 8
    assert prediction.abstention_threshold_fitted_on == (
        "source_validation_balanced_contracts"
    )
    assert np.all(prediction.routing_weights[~availability["graph"], 1] == 0)
    with pytest.raises(ValueError, match="target availability"):
        fit_saved_output_corerouter(
            sources,
            target_contract=contract_factory(
                "target2",
                role=ContractRole.TARGET,
            ),
            target_scores=target_scores,
            target_availability={},
            target_expert_costs={"feature": 1.0, "graph": 3.0},
            expert_prediction_seed=1,
            steps=1,
        )


def test_statistics_use_paired_seed_blocks_not_contract_pseudoreplication() -> None:
    rows = []
    for seed in (1, 2, 3):
        for contract in ("a", "b"):
            rows.extend(
                [
                    {
                        "dataset": "fixture",
                        "target_contract": contract,
                        "seed": seed,
                        "method": "coregraph",
                        "metric": "auprc",
                        "value": 0.7 + 0.01 * seed,
                    },
                    {
                        "dataset": "fixture",
                        "target_contract": contract,
                        "seed": seed,
                        "method": "baseline",
                        "metric": "auprc",
                        "value": 0.6 + 0.01 * seed,
                    },
                ]
            )
    blocks = build_paired_seed_blocks(
        rows,
        method="coregraph",
        baseline="baseline",
        metric="auprc",
    )
    assert blocks.seeds == (1, 2, 3)
    assert len(blocks.method_values) == 3
    assert np.asarray(blocks.method_values) - np.asarray(
        blocks.baseline_values
    ) == pytest.approx([0.1, 0.1, 0.1])
