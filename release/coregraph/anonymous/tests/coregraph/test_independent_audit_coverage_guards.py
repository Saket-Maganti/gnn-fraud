from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coregraph.contracts.axes import (
    BudgetAxis,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    DeviceClass,
    HistoryPolicy,
    ResourceAxis,
    ResourceSpec,
    ReviewMode,
    TimeAxis,
    TimeSpec,
    TopologyTransform,
    VisibilityAxis,
    VisibilitySpec,
    validate_identifier,
)
from coregraph.contracts.compatibility import from_protocol_contract
from coregraph.contracts.contract import DeploymentContract, migrate_v2_contract_payload
from coregraph.contracts.serialization import to_primitive, write_json, write_yaml
from coregraph.data.graph_views import GraphView, ViewRole
from coregraph.experiments.contract_splits import (
    ContractSplit,
    SplitFamily,
    leave_one_axis_value_out,
    leave_one_contract_out,
    observed_axes_unseen_combination_split,
)
from coregraph.experiments.pilot import (
    BaselinePrediction,
    MethodExecutionStatus,
    SavedSourceGroup,
    discover_prediction_manifests,
    evaluate_saved_output_pilot,
    load_prediction_artifacts,
)
from coregraph.objectives.scores import ScoreType
from coregraph.routing.abstention import (
    abstention_capacity_penalty,
    abstention_cost,
    apply_abstention_capacity,
    area_under_risk_coverage_curve,
    coverage,
    select_abstention_threshold,
    selective_risk,
)
from coregraph.routing.contract_encoder import (
    AtomicContractEncoder,
    FactorisedContractEncoder,
    NoContractEncoder,
)
from coregraph.routing.diagnostics import (
    graph_diagnostics,
    score_diagnostics,
    validate_target_diagnostics,
)
from coregraph.routing.fallback import FallbackStrategy, fallback_weights
from coregraph.routing.stability import (
    availability_mask_consistency,
    consistency_penalty,
    empirical_lipschitz_penalty,
    routing_flip_rate,
)


def test_structured_axes_cover_legacy_adapters_and_validation() -> None:
    assert validate_identifier("valid.id-1", "fixture") == "valid.id-1"
    with pytest.raises(ValueError, match="must match"):
        validate_identifier("Not valid", "fixture")

    for mode in VisibilityAxis:
        assert isinstance(VisibilitySpec.from_v2(mode), VisibilitySpec)

    for mode in ConstructionAxis:
        kwargs: dict[str, object] = {}
        if mode is ConstructionAxis.RECENT_WINDOW:
            kwargs["recent_window"] = 2
        if mode is ConstructionAxis.DEGREE_CAPPED:
            kwargs["degree_cap"] = 3
        if mode is ConstructionAxis.TASK_SPECIFIC_CUSTOM_TRANSFORM:
            kwargs["custom_transform_identifier"] = "curated_transform"
        spec = ConstructionSpec(mode, **kwargs)
        assert isinstance(spec.mode, ConstructionAxis)

    assert (
        ConstructionSpec(
            history_policy=HistoryPolicy.FULL_HISTORY,
            topology_transform=TopologyTransform.DEGREE_ONLY,
        ).mode
        is ConstructionAxis.DEGREE_ONLY
    )
    invalid_constructions = (
        {"recent_window": 0},
        {"degree_cap": 0},
        {"history_policy": HistoryPolicy.RECENT_WINDOW},
        {"topology_transform": TopologyTransform.DEGREE_CAPPED},
        {"topology_transform": TopologyTransform.CUSTOM},
        {
            "topology_transform": TopologyTransform.NONE,
            "custom_transform_identifier": "unexpected",
        },
        {
            "topology_transform": TopologyTransform.NO_GRAPH,
            "history_policy": HistoryPolicy.FULL_HISTORY,
        },
    )
    for kwargs in invalid_constructions:
        with pytest.raises(ValueError):
            ConstructionSpec(**kwargs)

    budget_values = {
        BudgetAxis.FRACTIONAL_REVIEW_CAPACITY: 0.2,
        BudgetAxis.FIXED_K: 2,
        BudgetAxis.ABSTENTION_CAPACITY: 0.1,
        BudgetAxis.LATENCY_BUDGET: 4.0,
    }
    for mode in BudgetAxis:
        spec = BudgetSpec(mode, value=budget_values.get(mode))
        assert isinstance(spec.mode, BudgetAxis)
        assert spec.value is None or spec.value >= 0
    for kwargs in (
        {"review_mode": ReviewMode.FRACTION},
        {"review_mode": ReviewMode.UNCONSTRAINED_RANKING, "review_fraction": 0.1},
        {"review_mode": ReviewMode.FIXED_K, "fixed_k": -1},
        {"review_mode": ReviewMode.UNCONSTRAINED_RANKING, "fixed_k": 1},
        {"cost_matrix": ((0.0, 1.0),)},
        {"cost_matrix": ((0.0, -1.0), (1.0, 0.0))},
        {"abstention_capacity": 1.1},
        {"latency_allowance_ms": -1.0},
    ):
        with pytest.raises(ValueError):
            BudgetSpec(**kwargs)

    for mode in ResourceAxis:
        kwargs = (
            {"custom_envelope_id": "custom_cpu"}
            if mode is ResourceAxis.CUSTOM_DEVICE_ENVELOPE
            else {}
        )
        spec = ResourceSpec(mode, **kwargs)
        assert isinstance(spec.mode, ResourceAxis)
        assert spec.memory_gb == spec.memory_cap_gb
        assert spec.latency_ms == spec.latency_cap_ms
    for kwargs in (
        {"memory_cap_gb": 0.0},
        {"latency_cap_ms": -1.0},
        {"unavailable_experts": ("bad expert",)},
        {"device_class": DeviceClass.CUSTOM},
        {"custom_envelope_id": "unexpected"},
    ):
        with pytest.raises(ValueError):
            ResourceSpec(**kwargs)


def test_contract_migration_serialization_and_legacy_protocol_adapter(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory()
    assert "Coordinate hash:" in contract.human_card()
    assert contract.compatible_with(replace(contract, environment_id="other"))
    assert contract.claim_projection(("time", "resource"))
    with pytest.raises(ValueError, match="unknown claim"):
        contract.claim_projection(("invalid",))

    json_path = tmp_path / "contract.json"
    yaml_path = tmp_path / "contract.yaml"
    json_path.write_text(contract.to_json(pretty=True), encoding="utf-8")
    yaml_path.write_text(contract.to_yaml(), encoding="utf-8")
    assert DeploymentContract.from_json(json_path) == contract
    assert DeploymentContract.from_json(contract.to_json()) == contract
    assert DeploymentContract.from_yaml(yaml_path) == contract
    assert DeploymentContract.from_yaml(contract.to_yaml()) == contract
    with pytest.raises(ValueError, match="mapping"):
        DeploymentContract.from_yaml("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="unsupported"):
        DeploymentContract.from_dict({"schema_version": 99})
    with pytest.raises(ValueError, match="V2 only"):
        migrate_v2_contract_payload({"schema_version": 3})

    assert to_primitive([Path("a"), TimeAxis.ROLLING]) == ["a", "rolling"]
    with pytest.raises(TypeError, match="serialize"):
        to_primitive(object())
    write_json(tmp_path / "nested" / "value.json", {"contract": contract})
    write_yaml(tmp_path / "nested" / "value.yaml", {"contract": contract})

    for index, name in enumerate(
        (
            "transductive_static",
            "strict_inductive_temporal",
            "rolling_deployment",
            "isolated_feature_control",
        )
    ):
        adapted = from_protocol_contract(
            SimpleNamespace(name=name),
            environment_id=f"legacy_{index}",
            role=ContractRole.SOURCE,
        )
        assert adapted.schema_version == 3
    with pytest.raises(ValueError, match="no curated"):
        from_protocol_contract(
            SimpleNamespace(name="unknown_protocol"),
            environment_id="legacy_unknown",
            role=ContractRole.SOURCE,
        )


def test_contract_split_adversarial_and_axis_holdout_paths(contract_factory) -> None:
    left = contract_factory("left")
    right = replace(
        contract_factory("right"),
        time=TimeSpec(TimeAxis.ROLLING, window=1),
    )
    contracts = (left, right)
    with pytest.raises(IndexError):
        leave_one_contract_out(contracts, 10)
    with pytest.raises(ValueError, match="unknown contract axis"):
        leave_one_axis_value_out(contracts, axis="bad", value="bad")
    with pytest.raises(ValueError, match="non-empty"):
        leave_one_axis_value_out(contracts, axis="time", value=TimeSpec(TimeAxis.EVENT_STREAM))
    held_out = leave_one_axis_value_out(contracts, axis="time", value=right.time)
    assert held_out.family is SplitFamily.TEMPORAL_HORIZON_HOLDOUT

    target = left.as_role(ContractRole.TARGET, environment_id="left_target")
    with pytest.raises(ValueError, match="non-empty"):
        ContractSplit(
            "empty",
            SplitFamily.LEAVE_ONE_CONTRACT_OUT,
            (),
            (target,),
            left.access_regime,
        )
    with pytest.raises(ValueError, match="leaked"):
        ContractSplit(
            "overlap",
            SplitFamily.LEAVE_ONE_CONTRACT_OUT,
            (target,),
            (target,),
            target.access_regime,
        )
    with pytest.raises(ValueError, match="atomic"):
        ContractSplit(
            "atomic",
            SplitFamily.LEAVE_ONE_CONTRACT_OUT,
            (right,),
            (target,),
            target.access_regime,
            atomic_target_id_seen=True,
        )
    with pytest.raises(ValueError, match="target role"):
        ContractSplit(
            "role",
            SplitFamily.LEAVE_ONE_CONTRACT_OUT,
            (right,),
            (left,),
            left.access_regime,
        )
    with pytest.raises(ValueError, match="unseen axis"):
        observed_axes_unseen_combination_split(contracts, 1)


def test_router_helpers_cover_all_fallback_and_diagnostic_paths(
    contract_factory,
) -> None:
    availability = torch.tensor([[True, True, False], [False, False, False]])
    for strategy in FallbackStrategy:
        kwargs = (
            {"source_validation_order": torch.tensor([1, 0, 2])}
            if strategy is FallbackStrategy.BEST_SOURCE_VALIDATION
            else {}
        )
        weights, abstain = fallback_weights(
            availability,
            strategy=strategy,
            feature_expert_index=2,
            **kwargs,
        )
        assert weights.shape == availability.shape
        assert abstain[1]
    with pytest.raises(ValueError, match="fixed expert order"):
        fallback_weights(
            availability[:1],
            strategy=FallbackStrategy.BEST_SOURCE_VALIDATION,
        )

    scores = score_diagnostics(
        np.asarray([[0.2, 0.8], [0.7, 0.3]]),
        score_type=ScoreType.PROBABILITY,
    )
    assert scores["score_disagreement"].shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        score_diagnostics(np.asarray([0.2, 0.8]), score_type=ScoreType.PROBABILITY)
    validate_target_diagnostics(("confidence",), target_access_allowed=False)
    with pytest.raises(ValueError, match="unknown"):
        validate_target_diagnostics(("not_registered",), target_access_allowed=False)
    with pytest.raises(ValueError, match="requires labels"):
        validate_target_diagnostics(("observed_target_error",), target_access_allowed=True)
    with pytest.raises(ValueError, match="declared target access"):
        validate_target_diagnostics(("predicted_prevalence",), target_access_allowed=False)

    contract = contract_factory()
    view = GraphView(
        visible_node_ids=np.asarray([0, 1, 2]),
        edge_index=np.asarray([[0], [1]]),
        directed=True,
        edge_attributes=None,
        edge_timestamps=np.asarray([1.0]),
        time_cutoff=1.0,
        source_mask=np.asarray([True, False, False]),
        target_mask=np.asarray([False, True, True]),
        construction=contract.construction,
        contract=contract,
        provenance=(),
        role=ViewRole.TARGET,
    )
    graph = graph_diagnostics(view)
    assert graph["graph_density"] == pytest.approx(1 / 6)


def test_encoder_stability_and_abstention_guard_paths(contract_factory) -> None:
    contracts = [contract_factory("source"), contract_factory("target")]
    encoder = FactorisedContractEncoder(
        embedding_dim=2,
        output_dim=4,
        axis_dropout=0.5,
        contract_noise_std=0.1,
    )
    encoder.train()
    assert encoder(contracts).shape == (2, 4)
    with pytest.raises(ValueError, match="at least one"):
        encoder([])
    for kwargs in (
        {"embedding_dim": 0},
        {"axis_dropout": 1.0},
        {"contract_noise_std": -1.0},
    ):
        with pytest.raises(ValueError):
            FactorisedContractEncoder(**kwargs)
    atomic = AtomicContractEncoder((contracts[0].contract_id,), output_dim=4)
    assert atomic(contracts).shape == (2, 4)
    assert NoContractEncoder(output_dim=4)(contracts).shape == (2, 4)

    weights = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    perturbed = torch.tensor([[0.7, 0.3], [0.8, 0.2]])
    assert consistency_penalty(weights, perturbed) > 0
    assert routing_flip_rate(weights, perturbed) == 0.5
    assert empirical_lipschitz_penalty(
        weights,
        perturbed,
        torch.tensor([0.1, 0.2]),
        limit=0.1,
    ) > 0
    assert availability_mask_consistency(
        weights,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[True, False], [True, False]]),
    ) == 0
    with pytest.raises(ValueError, match="align"):
        consistency_penalty(weights, perturbed[:1])
    with pytest.raises(ValueError, match="align"):
        routing_flip_rate(weights, perturbed[:1])

    probability = torch.tensor([0.1, 0.9, 0.8])
    forced = torch.tensor([True, False, False])
    assert apply_abstention_capacity(
        probability,
        2 / 3,
        forced_abstention=forced,
    ).sum() == 2
    with pytest.raises(ValueError):
        apply_abstention_capacity(probability, -0.1)
    with pytest.raises(ValueError, match="align"):
        apply_abstention_capacity(probability, 0.5, forced_abstention=forced[:1])
    with pytest.raises(ValueError):
        coverage(torch.tensor([]))
    assert torch.isnan(selective_risk(torch.ones(2), torch.ones(2, dtype=torch.bool)))
    with pytest.raises(ValueError, match="align"):
        selective_risk(torch.ones(2), torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError, match="non-empty"):
        area_under_risk_coverage_curve(torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError):
        abstention_cost(forced, cost=-1)
    with pytest.raises(ValueError):
        abstention_capacity_penalty(probability, capacity=2)
    with pytest.raises(ValueError, match="align"):
        select_abstention_threshold(torch.ones(2), torch.ones(1), capacity=0.5)
    with pytest.raises(ValueError, match="source-validation"):
        select_abstention_threshold(torch.tensor([]), torch.tensor([]), capacity=0.5)
    with pytest.raises(ValueError, match="configuration"):
        select_abstention_threshold(torch.ones(2), torch.ones(2), capacity=1.1)


def test_manifest_loader_and_pilot_long_form_metrics(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory()
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "node_id,score,y_true,split,expert_id\n"
        "node:1,0.9,1,test,feature\n"
        "node:2,0.1,2,test,feature\n",
        encoding="utf-8",
    )
    checksum = hashlib.sha256(predictions.read_bytes()).hexdigest()
    payload = {
        "expert_id": "feature",
        "dataset": "fixture",
        "task": "node_classification",
        "prediction_unit": "node",
        "contract_coordinate_hash": contract.coordinate_hash,
        "environment_id": contract.environment_id,
        "expert_prediction_seed": 1,
        "fold": "fold0",
        "prediction_path": predictions.name,
        "prediction_checksum": checksum,
        "config_hash": "a" * 64,
        "code_hash": "b" * 40,
        "contract_role": contract.role.value,
        "deployment_contract": contract.to_dict(),
        "expert_available": True,
        "availability_reason_codes": ["available"],
        "compute_cost": 1.0,
        "score_type": "PROBABILITY",
    }
    manifest = tmp_path / "prediction_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert discover_prediction_manifests((tmp_path, manifest)) == [manifest.resolve()]
    artifacts = load_prediction_artifacts((manifest,))
    assert artifacts[0].score_type is ScoreType.PROBABILITY

    missing = tmp_path / "missing.json"
    missing.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing prediction manifest keys"):
        load_prediction_artifacts((missing,))
    bad_checksum = tmp_path / "bad_checksum.json"
    bad_checksum.write_text(
        json.dumps({**payload, "prediction_checksum": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prediction_artifacts((bad_checksum,))
    bad_hash = tmp_path / "bad_hash.json"
    bad_hash.write_text(
        json.dumps({**payload, "contract_coordinate_hash": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coordinate hash mismatch"):
        load_prediction_artifacts((bad_hash,))

    candidates = {
        "method": BaselinePrediction(
            scores=np.asarray([0.9, 0.1, 0.8, 0.2]),
            abstention_probability=np.asarray([0.1, 0.2, 0.3, 0.4]),
            abstain=np.zeros(4, dtype=bool),
            forced_abstention=np.zeros(4, dtype=bool),
            expected_compute=np.asarray([1.0, 1.0, 1.0, 1.0]),
            abstention_threshold=0.8,
            abstention_threshold_provenance="source_validation",
            abstention_capacity=0.5,
            abstention_cost=0.2,
            execution_status=MethodExecutionStatus.EXECUTABLE,
        ),
        "contract_feasible_oracle": BaselinePrediction(
            scores=np.asarray([1.0, 0.0, 1.0, 0.0]),
            abstention_probability=np.zeros(4),
            abstain=np.zeros(4, dtype=bool),
            forced_abstention=np.zeros(4, dtype=bool),
            expected_compute=np.ones(4),
            abstention_threshold=None,
            abstention_threshold_provenance="offline_contract_oracle",
            abstention_capacity=None,
            abstention_cost=0.0,
            execution_status=MethodExecutionStatus.EXECUTABLE,
            offline_oracle=True,
            diagnostic_only=False,
        ),
    }
    result = evaluate_saved_output_pilot(
        np.asarray([1, 2, 1, 2]),
        candidates,
        dataset="fixture",
        target_contract="target",
        expert_prediction_seed=1,
        router_training_seeds={"method": 101},
        fold="fold0",
    )
    assert result["status"] == "MEASURED_FROM_SAVED_PREDICTIONS"
    assert {row["metric"] for row in result["rows"]} == {
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
    }
    with pytest.raises(ValueError, match="oracle"):
        evaluate_saved_output_pilot(
            np.asarray([1, 2]),
            {"method": candidates["method"]},
            dataset="fixture",
            target_contract="target",
            expert_prediction_seed=1,
            router_training_seeds={"method": 101},
            fold="fold0",
        )


def test_saved_source_group_validation_paths(contract_factory) -> None:
    base = {
        "contract": contract_factory(),
        "scores": {"a": np.ones(2)},
        "labels": np.asarray([1, 2]),
        "splits": np.asarray(["train", "validation"]),
        "availability": {"a": np.ones(2, dtype=bool)},
        "expert_costs": {"a": 1.0},
    }
    SavedSourceGroup(**base)
    for mutation, message in (
        ({"availability": {}}, "identical experts"),
        ({"splits": np.asarray(["train"])}, "split rows"),
        ({"scores": {"a": np.ones(1)}}, "rows do not align"),
        ({"expert_costs": {"a": -1.0}}, "cannot be negative"),
        ({"splits": np.asarray(["train", "test"])}, "train/validation"),
    ):
        with pytest.raises(ValueError, match=message):
            SavedSourceGroup(**{**base, **mutation})
