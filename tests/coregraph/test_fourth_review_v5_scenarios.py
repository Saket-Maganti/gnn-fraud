from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from coregraph.contracts.axes import (
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    VisibilityAxis,
)
from coregraph.data.leakage import (
    ScenarioPredictionScope,
    audit_evaluation_scenario_scopes,
)
from coregraph.experiments.canonical_recovery import (
    REQUIRED_DATASETS,
    REQUIRED_EXPERTS,
    REQUIRED_PROTOCOLS,
    REQUIRED_SEEDS,
    discover_evidence_locks,
    discover_package_import_manifests,
    discover_prediction_index_records,
    discover_raw_filesystem_predictions,
    discover_result_index_records,
    discover_validation_reports,
    base_completeness_matrix,
    recover_rb09v3,
    scenario_completeness_surfaces,
)
from coregraph.experiments.protocol_registry import load_protocol_registry
from coregraph.experiments.scenario_manifests import (
    BasePredictionArtifact,
    CodeProvenanceType,
    EvaluationScenarioBinding,
    ScenarioArtifactBinding,
    make_scenario_id,
    role_neutral_contract_coordinates,
    validate_no_training_scenarios,
)
from scripts.coregraph import (
    evaluate_pilot_gate as gate_cli,
    run_saved_output_pilot as runner_cli,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results/coregraph_build/CONTRACT_PROTOCOL_REGISTRY_V4.json"
if not REGISTRY.is_file():
    REGISTRY = ROOT / "specifications/CONTRACT_PROTOCOL_REGISTRY_V4.json"

PROTOCOLS = (
    "strict_inductive",
    "isolated_inductive",
    "transductive_structure",
)
EXPERTS = ("feature_mlp", "gcn", "graphsage")
DATASETS = ("tiny_elliptic", "tiny_dgraphfin")
SEEDS = (1, 2)


def _protocol_contract(contract_factory, dataset: str, protocol: str):
    visibility = {
        "strict_inductive": VisibilityAxis.STRICT_INDUCTIVE,
        "isolated_inductive": VisibilityAxis.ISOLATED_INDUCTIVE,
        "transductive_structure": VisibilityAxis.TRANSDUCTIVE_STRUCTURE,
    }[protocol]
    construction = (
        ConstructionSpec(ConstructionAxis.NO_GRAPH)
        if protocol == "isolated_inductive"
        else ConstructionSpec(ConstructionAxis.FULL_GRAPH)
    )
    contract = contract_factory(
        f"{dataset[:12]}.{protocol[:20]}",
        role=ContractRole.TARGET,
        visibility=visibility,
        construction=construction,
    )
    return replace(contract, dataset_id=dataset, task_id="node_classification")


def _write_base_csv(path: Path, *, dataset: str, protocol: str, expert: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "node_id",
                "score",
                "y_true",
                "split",
                "label_known",
                "timestamp",
                "expert_id",
            ),
        )
        writer.writeheader()
        for identifier, score, label, split, known, timestamp in (
            (f"{dataset}:train", 0.2, 2, "train", True, 1),
            (f"{dataset}:validation", 0.8, 1, "validation", True, 2),
            (f"{dataset}:unknown", 0.3, 0, "test", False, 10),
            (f"{dataset}:test", 0.9, 1, "test", True, 11),
        ):
            writer.writerow(
                {
                    "node_id": identifier,
                    "score": score,
                    "y_true": label,
                    "split": split,
                    "label_known": str(known).lower(),
                    "timestamp": timestamp,
                    "expert_id": expert,
                }
            )


def _base_artifact(path: Path, *, contract, dataset: str, protocol: str, expert: str, seed: int):
    coordinates = role_neutral_contract_coordinates(contract)
    config = {"dataset": dataset, "protocol": protocol, "expert": expert, "seed": seed}
    return BasePredictionArtifact(
        dataset=dataset,
        task="node_classification",
        prediction_unit="node",
        protocol_id=protocol,
        contract_coordinate_hash=contract.coordinate_hash,
        role_neutral_contract_coordinates=coordinates,
        expert_id=expert,
        expert_prediction_seed=seed,
        fold="fold0",
        path=path,
        checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_schema=(
            "node_id",
            "score",
            "y_true",
            "split",
            "label_known",
            "timestamp",
            "expert_id",
        ),
        provider_split_mapping={
            "train": "train",
            "validation": "validation",
            "test": "test",
            "unscored": "unscored",
        },
        label_mapping={"0": "unknown", "1": "fraud", "2": "normal"},
        positive_label_id=1,
        config_payload=config,
        config_sha256=hashlib.sha256(
            (
                '{"dataset":"'
                + dataset
                + '","expert":"'
                + expert
                + '","protocol":"'
                + protocol
                + '","seed":'
                + str(seed)
                + "}"
            ).encode()
        ).hexdigest(),
        config_provenance_type="FIXTURE_CONFIG",
        config_provenance_path=str(path),
        code_provenance_type=CodeProvenanceType.UNRESOLVED_LEGACY_CODE,
        code_provenance_value="fixture-code",
        code_provenance_path=str(path),
        routing_cost_value=1.0,
        routing_cost_unit="relative_parameter_count",
        routing_cost_provenance="FIXTURE_DECLARED_PROXY",
        measured_compute_available=True,
        measured_compute_record={
            "runtime_seconds": 1.0,
            "measurement_scope": "per_run",
        },
        validation_evidence=({"status": "FIXTURE_VALIDATED"},),
        artifact_family="fixture_v5",
        source_package="fixture",
        source_archive_path=str(path),
        source_archive_sha256="c" * 64,
    )


def _fixture_surface(tmp_path: Path, contract_factory):
    artifacts = []
    contracts = {}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            contract = _protocol_contract(contract_factory, dataset, protocol)
            contracts[(dataset, protocol)] = contract
            for seed in SEEDS:
                for expert in EXPERTS:
                    path = tmp_path / (
                        f"{dataset}__{protocol}__{expert}__seed{seed}.csv"
                    )
                    _write_base_csv(
                        path,
                        dataset=dataset,
                        protocol=protocol,
                        expert=expert,
                    )
                    artifacts.append(
                        _base_artifact(
                            path,
                            contract=contract,
                            dataset=dataset,
                            protocol=protocol,
                            expert=expert,
                            seed=seed,
                        )
                    )
    by_key = {artifact.logical_key: artifact for artifact in artifacts}
    scenarios = []
    for dataset in DATASETS:
        for target_protocol in PROTOCOLS:
            sources = tuple(
                protocol for protocol in PROTOCOLS if protocol != target_protocol
            )
            for seed in SEEDS:
                scenario_id = make_scenario_id(
                    dataset=dataset,
                    target_protocol_id=target_protocol,
                    expert_prediction_seed=seed,
                    fold="fold0",
                    access_regime="DG_NO_TARGET",
                )
                bindings = []
                for protocol in PROTOCOLS:
                    role = "target" if protocol == target_protocol else "source"
                    for expert in EXPERTS:
                        artifact = by_key[
                            (
                                dataset,
                                "node_classification",
                                protocol,
                                expert,
                                seed,
                                "fold0",
                            )
                        ]
                        bindings.append(
                            ScenarioArtifactBinding(
                                scenario_id=scenario_id,
                                base_artifact_hash=artifact.base_artifact_hash,
                                base_protocol_id=protocol,
                                bound_protocol_id=protocol,
                                expert_id=expert,
                                role=role,
                                permitted_splits=(
                                    ("test",)
                                    if role == "target"
                                    else ("train", "validation")
                                ),
                                evaluation_split=(
                                    "test" if role == "target" else "validation"
                                ),
                            )
                        )
                scenarios.append(
                    EvaluationScenarioBinding(
                        scenario_id=scenario_id,
                        dataset=dataset,
                        target_protocol_id=target_protocol,
                        source_protocol_ids=sources,
                        expert_prediction_seed=seed,
                        fold="fold0",
                        access_regime=contracts[
                            (dataset, target_protocol)
                        ].access_regime,
                        target_operational_contract=contracts[
                            (dataset, target_protocol)
                        ],
                        bindings=tuple(bindings),
                    )
                )
    return artifacts, scenarios


def test_v5_end_to_end_no_training_scenario_materialization(
    tmp_path: Path,
    contract_factory,
) -> None:
    artifacts, scenarios = _fixture_surface(tmp_path, contract_factory)
    result = validate_no_training_scenarios(
        artifacts,
        scenarios,
        registry=load_protocol_registry(REGISTRY),
        expected_datasets=DATASETS,
        expected_protocols=PROTOCOLS,
        expected_experts=EXPERTS,
        expected_seeds=SEEDS,
    )
    assert result["base_artifact_count"] == 36
    assert result["scenario_count"] == 12
    assert result["scenario_binding_count"] == 108
    assert result["training_performed"] is False
    assert result["fitting_path_reachable"] is False
    assert result["metric_computation_performed"] is False
    assert result["oracle_computation_performed"] is False
    assert result["target_labels_accessed_before_scoring"] is False
    assert all(
        item["excluded_unknown_target_identifiers"]
        == [f"{item['dataset']}:unknown"]
        for item in result["materializations"]
    )
    strict = next(
        artifact
        for artifact in artifacts
        if artifact.dataset == DATASETS[0]
        and artifact.protocol_id == "strict_inductive"
        and artifact.expert_id == "feature_mlp"
        and artifact.expert_prediction_seed == 1
    )
    roles = {
        binding.role
        for scenario in scenarios
        for binding in scenario.bindings
        if binding.base_artifact_hash == strict.base_artifact_hash
    }
    assert roles == {"source", "target"}
    assert all(
        item["target_protocol_id"]
        not in item["source_protocol_ids"]
        for item in result["materializations"]
    )
    assert all(
        ":" in item["target_contract_id"]
        and item["target_protocol_id"] in PROTOCOLS
        for item in result["materializations"]
    )


def _scope(*, scenario: str, role: str, base_hash: str, protocol: str):
    return ScenarioPredictionScope(
        scenario_id=scenario,
        dataset="fixture",
        base_artifact_hash=base_hash,
        expert_id="feature_mlp",
        base_protocol_id=protocol,
        bound_protocol_id=protocol,
        expert_prediction_seed=1,
        fold="fold0",
        role=role,
        contract_coordinate_hash=("a" if protocol == "strict_inductive" else "b")
        * 64,
        path="/tmp/shared.csv",
        checksum="c" * 64,
        selected_identifiers=("source",) if role == "source" else ("target",),
        selected_splits=("validation",) if role == "source" else ("test",),
        selected_label_known=(True,),
        selected_timestamps=(1.0,) if role == "source" else (2.0,),
    )


def test_cross_scenario_role_reuse_passes_but_same_scenario_reuse_fails() -> None:
    target_a = _scope(
        scenario="scenario-a",
        role="target",
        base_hash="same",
        protocol="strict_inductive",
    )
    source_b = replace(
        target_a,
        scenario_id="scenario-b",
        role="source",
        selected_identifiers=("source",),
        selected_splits=("validation",),
        selected_timestamps=(1.0,),
    )
    report_a = audit_evaluation_scenario_scopes(
        (target_a,),
        scenario_id="scenario-a",
        dataset="fixture",
        target_protocol_id="strict_inductive",
        source_protocol_ids=("isolated_inductive", "transductive_structure"),
        expert_prediction_seed=1,
        fold="fold0",
    )
    report_b = audit_evaluation_scenario_scopes(
        (source_b,),
        scenario_id="scenario-b",
        dataset="fixture",
        target_protocol_id="transductive_structure",
        source_protocol_ids=("isolated_inductive", "strict_inductive"),
        expert_prediction_seed=1,
        fold="fold0",
    )
    assert report_a.passed
    assert report_b.passed
    source_a = replace(target_a, role="source")
    conflict = audit_evaluation_scenario_scopes(
        (target_a, source_a),
        scenario_id="scenario-a",
        dataset="fixture",
        target_protocol_id="strict_inductive",
        source_protocol_ids=("isolated_inductive", "transductive_structure"),
        expert_prediction_seed=1,
        fold="fold0",
    )
    assert not conflict.passed
    assert "SAME_ARTIFACT_BOUND_TO_BOTH_ROLES" in {
        finding.code for finding in conflict.findings
    }


def test_scenario_scope_rejects_overlap_unknown_target_and_source_test() -> None:
    source = _scope(
        scenario="scenario-a",
        role="source",
        base_hash="source",
        protocol="isolated_inductive",
    )
    target = _scope(
        scenario="scenario-a",
        role="target",
        base_hash="target",
        protocol="strict_inductive",
    )
    report = audit_evaluation_scenario_scopes(
        (
            replace(
                source,
                selected_identifiers=("shared",),
                selected_splits=("test",),
            ),
            replace(
                target,
                selected_identifiers=("shared",),
                selected_label_known=(False,),
            ),
        ),
        scenario_id="scenario-a",
        dataset="fixture",
        target_protocol_id="strict_inductive",
        source_protocol_ids=("isolated_inductive", "transductive_structure"),
        expert_prediction_seed=1,
        fold="fold0",
    )
    codes = {finding.code for finding in report.findings}
    assert {
        "SOURCE_TARGET_SPLIT_ID_OVERLAP",
        "TEST_ROWS_ENTER_SOURCE_SCOPE",
        "UNKNOWN_PROVIDER_LABEL_ENTERS_SCORING",
    } <= codes


def test_missing_base_artifact_blocks_scenario_reference(
    tmp_path: Path,
    contract_factory,
) -> None:
    artifacts, scenarios = _fixture_surface(tmp_path, contract_factory)
    by_hash = {artifact.base_artifact_hash: artifact for artifact in artifacts[1:]}
    with pytest.raises(ValueError, match="missing base artifact"):
        scenarios[0].validate_artifact_references(by_hash)


def test_canonical_index_discovery_does_not_depend_on_filename(
    tmp_path: Path,
) -> None:
    opaque_prediction = tmp_path / "opaque_member_0007.csv"
    opaque_prediction.write_text("node_id,score\n1,0.5\n", encoding="utf-8")
    index = tmp_path / "PREDICTIONS_FULL10_INDEX_V22.json"
    index.write_text(
        json.dumps(
            {
                "lane_id": "canonical_lane",
                "predictions": [
                    {
                        "dataset": "elliptic",
                        "protocol": "strict_inductive",
                        "model": "sage",
                        "seed": 7,
                        "logical_path": opaque_prediction.name,
                        "sha256": hashlib.sha256(
                            opaque_prediction.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records = discover_prediction_index_records((tmp_path,))
    assert len(records) == 1
    assert records[0].logical_key == (
        "elliptic",
        "strict_inductive",
        "graphsage",
        7,
    )
    assert records[0].prediction_path == "opaque_member_0007.csv"


def test_all_canonical_discovery_adapters_use_structured_evidence(
    tmp_path: Path,
) -> None:
    result_record = {
        "artifact_family": "fixture_family",
        "dataset": "elliptic",
        "protocol": "strict_inductive",
        "model": "sage",
        "seed": 7,
        "logical_path": "opaque-result.json",
        "prediction_reference": "opaque-prediction.csv",
    }
    (tmp_path / "RESULT_INDEX_V22.json").write_text(
        json.dumps({"results": [result_record]}),
        encoding="utf-8",
    )
    (tmp_path / "RESULTS_FULL10.jsonl").write_text(
        json.dumps({**result_record, "seed": 8}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "V22_FINAL_EVIDENCE_LOCK.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )
    (tmp_path / "import_manifest.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )
    (tmp_path / "prediction_validation_report.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )
    raw = tmp_path / "elliptic__strict_inductive__sage__seed7.csv"
    raw.write_text("node_id,score\n1,0.5\n", encoding="utf-8")

    results = discover_result_index_records((tmp_path,))
    assert {record.seed for record in results} == {7, 8}
    assert all(
        record.logical_key[:3]
        == ("elliptic", "strict_inductive", "graphsage")
        for record in results
    )
    assert len(discover_evidence_locks((tmp_path,))) == 1
    assert len(discover_package_import_manifests((tmp_path,))) == 1
    assert len(discover_validation_reports((tmp_path,))) == 1
    raw_candidates = discover_raw_filesystem_predictions((tmp_path,))
    assert raw_candidates[raw.name] == (str(raw.resolve()),)


def test_rb09v3_recovery_uses_canonical_index_then_archive(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    run_dir = historical / "results/runs_rb09v3"
    import_dir = historical / "results/runs_rb15_graphsafe_tta"
    run_dir.mkdir(parents=True)
    import_dir.mkdir(parents=True)
    archive = historical / "canonical_rb09v3.zip"
    prediction_bytes = (
        b"node_id,score,y_true,split,label_known\n"
        b"1,0.9,1,test,true\n"
    )
    source_protocol = {
        "strict_inductive": "strict_inductive",
        "isolated_inductive": "inductive_isolated",
        "transductive_structure": "transductive",
    }
    source_model = {
        "feature_mlp": "mlp",
        "gcn": "gcn",
        "graphsage": "sage",
    }
    inventory = []
    result_rows = []
    locations = []
    with zipfile.ZipFile(archive, "w") as handle:
        for dataset in REQUIRED_DATASETS:
            for protocol in REQUIRED_PROTOCOLS:
                for expert in REQUIRED_EXPERTS:
                    for seed in REQUIRED_SEEDS:
                        name = (
                            f"{dataset}__{source_protocol[protocol]}__"
                            f"{source_model[expert]}__seed{seed}.csv"
                        )
                        member = f"predictions/{name}"
                        handle.writestr(member, prediction_bytes)
                        inventory.append(
                            {
                                "dataset": dataset,
                                "protocol": source_protocol[protocol],
                                "model": source_model[expert],
                                "seed": seed,
                                "path": f"kaggleoutputs/canonical/{member}",
                                "size_bytes": len(prediction_bytes),
                            }
                        )
                        result_rows.append(
                            {
                                "dataset": dataset,
                                "protocol": source_protocol[protocol],
                                "model": source_model[expert],
                                "seed": seed,
                                "command": "runner --device cuda",
                                "early_stopping_metric": "val_f1",
                                "early_stopping_split": "validation",
                                "scaler_mode": "train_only",
                                "graph_mode": source_protocol[protocol],
                                "split_name": "frozen",
                                "runtime_seconds": "1.25",
                                "git_commit": "a" * 40,
                            }
                        )
                        locations.append(f"{archive}::{member}")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (run_dir / "ARTIFACT_FAMILY.json").write_text(
        json.dumps(
            {
                "artifact_family": "RB09v3_structure_decay",
                "n_run_rows": 180,
                "n_prediction_files": 180,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions_manifest.json").write_text(
        json.dumps({"n_prediction_files": 180, "files": inventory}),
        encoding="utf-8",
    )
    with (run_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (import_dir / "import_manifest.json").write_text(
        json.dumps(
            {
                "artifact_family": "RB15_graphsafe_tta",
                "source_files": [
                    {
                        "path": str(archive),
                        "sha256": archive_sha,
                        "size_bytes": archive.stat().st_size,
                    }
                ],
                "prediction_locations": locations,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    records, _, summary = recover_rb09v3(
        historical_root=historical,
        search_roots=(historical,),
    )
    assert summary["usable_artifact_count"] == 180
    assert {record.status for record in records} == {"RECOVERED_CANONICAL"}
    assert summary["unresolved_routing_cost_count"] == 180
    assert summary["unresolved_code_provenance_count"] == 0
    assert {
        row["status"] for row in base_completeness_matrix(records)
    } == {"BLOCKED_PROVENANCE"}

    archive.unlink()
    records, _, summary = recover_rb09v3(
        historical_root=historical,
        search_roots=(historical,),
    )
    assert summary["missing_index_reference_count"] == 180
    assert summary["true_missing_artifact_count"] == 0
    assert {record.status for record in records} == {
        "INDEX_REFERENCED_FILE_MISSING"
    }


def test_production_completeness_surface_is_exactly_180_60_540() -> None:
    base_rows = []
    for dataset in REQUIRED_DATASETS:
        for protocol in REQUIRED_PROTOCOLS:
            for expert in REQUIRED_EXPERTS:
                for seed in REQUIRED_SEEDS:
                    coordinate = hashlib.sha256(
                        f"{dataset}:{protocol}:{expert}:{seed}".encode()
                    ).hexdigest()
                    base_rows.append(
                        {
                            "dataset": dataset,
                            "protocol_id": protocol,
                            "expert_id": expert,
                            "expert_prediction_seed": seed,
                            "fold": "fold0",
                            "status": "COMPLETE",
                            "recovery_status": "RECOVERED_CANONICAL",
                            "base_coordinate_id": coordinate,
                            "base_artifact_hash": hashlib.sha256(
                                f"artifact:{coordinate}".encode()
                            ).hexdigest(),
                        }
                    )
    scenario_rows, index = scenario_completeness_surfaces(base_rows)
    assert len(base_rows) == 180
    assert len(scenario_rows) == 60
    assert index["scenario_count"] == 60
    assert index["binding_count"] == 540
    assert sum(
        len(scenario["bindings"]) for scenario in index["scenarios"]
    ) == 540


def test_v5_runner_and_gate_end_to_end_without_execution(
    tmp_path: Path,
    contract_factory,
    monkeypatch,
) -> None:
    artifacts, scenarios = _fixture_surface(tmp_path, contract_factory)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for index, artifact in enumerate(artifacts):
        manifest = manifest_dir / f"{index:03d}_base_prediction_manifest_v5.json"
        manifest.write_text(
            json.dumps(artifact.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    scenario_index = tmp_path / "scenario_index.json"
    scenario_index.write_text(
        json.dumps(
            {
                "schema_version": "coregraph_scenario_binding_index_v5",
                "scenario_count": 12,
                "binding_count": 108,
                "source_binding_count": 72,
                "target_binding_count": 36,
                "scenarios": [scenario.to_dict() for scenario in scenarios],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_output = tmp_path / "validation_v5.json"
    forbidden_output = tmp_path / "forbidden_execution.json"
    config = tmp_path / "runner_v5.json"
    config.write_text(
        json.dumps(
            {
                "manifest_schema_version": "v5",
                "base_prediction_manifest_roots": [str(manifest_dir)],
                "scenario_binding_index": str(scenario_index),
                "required_datasets": list(DATASETS),
                "required_target_protocols": list(PROTOCOLS),
                "required_experts": list(EXPERTS),
                "required_seeds": list(SEEDS),
                "required_folds": ["fold0"],
                "output": str(forbidden_output),
                "validation_output": str(validation_output),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_saved_output_pilot.py",
            "--config",
            str(config),
            "--validate-only",
            "--protocol-registry",
            str(REGISTRY),
        ],
    )
    assert runner_cli.main() == 0
    assert not forbidden_output.exists()
    materialized = json.loads(validation_output.read_text(encoding="utf-8"))
    assert materialized["status"] == "VALIDATED_NO_TRAINING_V5"
    assert materialized["fitting_path_reachable"] is False

    readiness_spec = {
        "schema_version": "coregraph_pilot_manifest_readiness_spec_v5",
        "required_surface": {
            "base_artifacts": 36,
            "evaluation_scenarios": 12,
            "scenario_bindings": 108,
            "source_bindings": 72,
            "target_bindings": 36,
            "datasets": list(DATASETS),
            "protocols": list(PROTOCOLS),
            "expert_prediction_seeds": list(SEEDS),
            "folds": ["fold0"],
        },
    }
    gate = gate_cli.validate_v5_manifest_readiness(
        materialized,
        readiness_spec,
    )
    assert gate["status"] == "V5_MANIFEST_READINESS_VALIDATED"
    assert gate["passed"]
    assert gate["pilot_authorized"] is False
    assert gate["training_performed"] is False
    assert gate["metric_computation_performed"] is False
    assert gate["oracle_computation_performed"] is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_saved_output_pilot.py",
            "--config",
            str(config),
            "--execute",
            "--protocol-registry",
            str(REGISTRY),
        ],
    )
    with pytest.raises(RuntimeError, match="readiness-only"):
        runner_cli.main()


def test_v5_schemas_separate_artifact_and_scenario_roles() -> None:
    base_schema = json.loads(
        (
            ROOT
            / "configs/coregraph/schemas/base_prediction_manifest_v5.schema.json"
        ).read_text(encoding="utf-8")
    )
    scenario_schema = json.loads(
        (
            ROOT
            / "configs/coregraph/schemas/scenario_binding_index_v5.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert base_schema["properties"]["schema_version"]["const"].endswith("_v5")
    assert "contract_role" not in base_schema["properties"]
    assert base_schema["not"] == {"required": ["contract_role"]}
    binding = scenario_schema["$defs"]["binding"]
    assert binding["properties"]["role"]["enum"] == ["source", "target"]
    assert binding["properties"]["role_binding_id"]["pattern"].endswith("{64}$")
