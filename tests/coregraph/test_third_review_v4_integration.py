from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coregraph.contracts.axes import (
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    VisibilityAxis,
)
from coregraph.data.leakage import (
    PredictionArtifactScope,
    audit_cross_role_prediction_scopes,
)
from coregraph.evaluation.statistics import build_paired_seed_blocks
from coregraph.experiments.manifest_conversion import (
    audit_candidates,
    build_completeness_matrix,
    build_conversion_records,
    discover_historical_predictions,
    discover_validation_evidence,
)
from coregraph.experiments.pilot import (
    MethodExecutionStatus,
    PredictionArtifact,
    align_artifact_group,
    derive_router_seed,
    load_prediction_artifacts,
    validate_artifact_groups,
)
from coregraph.experiments.protocol_registry import (
    load_protocol_registry,
    validate_protocol_bindings,
)
from scripts.coregraph.evaluate_pilot_gate import validate_no_training_completeness
from scripts.coregraph import (
    convert_prediction_manifests_v4 as converter_cli,
    evaluate_pilot_gate as gate,
    run_saved_output_pilot as runner_cli,
)
from scripts.coregraph.run_saved_output_pilot import build_no_training_materialization


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results/coregraph_build/CONTRACT_PROTOCOL_REGISTRY_V4.json"
if not REGISTRY.is_file():
    REGISTRY = ROOT / "specifications/CONTRACT_PROTOCOL_REGISTRY_V4.json"
V4_GATE = ROOT / "results/coregraph_build/PILOT_GATE_FROZEN_SPEC_V4.json"
if not V4_GATE.is_file():
    V4_GATE = ROOT / "specifications/PILOT_GATE_FROZEN_SPEC_V4.json"


def _write_predictions(
    path: Path,
    *,
    expert: str,
    rows: tuple[tuple[str, float, int, str, bool, float], ...],
) -> None:
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
        for identifier, score, label, split, known, timestamp in rows:
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


def _write_historical_predictions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "protocol",
                "model",
                "seed",
                "split",
                "node_id",
                "timestep",
                "y_true",
                "score",
                "label_known",
                "artifact_source",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "dataset": "elliptic",
                    "protocol": "strict_inductive",
                    "model": "mlp",
                    "seed": 1,
                    "split": "train",
                    "node_id": 1,
                    "timestep": 1,
                    "y_true": 2,
                    "score": 0.1,
                    "label_known": True,
                    "artifact_source": "fixture",
                },
                {
                    "dataset": "elliptic",
                    "protocol": "strict_inductive",
                    "model": "mlp",
                    "seed": 1,
                    "split": "test",
                    "node_id": 2,
                    "timestep": 2,
                    "y_true": 0,
                    "score": 0.2,
                    "label_known": False,
                    "artifact_source": "fixture",
                },
                {
                    "dataset": "elliptic",
                    "protocol": "strict_inductive",
                    "model": "mlp",
                    "seed": 1,
                    "split": "test",
                    "node_id": 3,
                    "timestep": 3,
                    "y_true": 1,
                    "score": 0.9,
                    "label_known": True,
                    "artifact_source": "fixture",
                },
            )
        )


def _artifact(
    contract,
    path: Path,
    *,
    expert: str,
    protocol_id: str,
    seed: int = 1,
    permitted_splits: tuple[str, ...] | None = None,
    evaluation_split: str | None = None,
) -> PredictionArtifact:
    role = contract.role.value
    return PredictionArtifact(
        expert_id=expert,
        dataset="fixture",
        task="node_classification",
        prediction_unit="node",
        protocol_id=protocol_id,
        contract_coordinate_hash=contract.coordinate_hash,
        contract_id=contract.contract_id,
        environment_id=contract.environment_id,
        seed=seed,
        fold="fold0",
        path=path,
        checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        config_hash="a" * 64,
        code_hash="b" * 40,
        contract_role=role,
        deployment_contract=contract,
        permitted_splits=permitted_splits
        or (("train", "validation") if role == "source" else ("test",)),
        evaluation_split=evaluation_split
        or ("validation" if role == "source" else "test"),
        row_scope_policy="filter_and_audit",
        label_mapping={"0": "unknown", "1": "fraud", "2": "normal"},
        positive_label_id=1,
        provider_split_mapping={
            "train": "train",
            "val": "validation",
            "validation": "validation",
            "test": "test",
            "unscored": "unscored",
        },
        compute_cost_provenance="fixture_declared",
        original_prediction_path=str(path),
        original_prediction_checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_protocol_alias_contract_and_coordinate_identities_are_distinct(
    tmp_path: Path,
    contract_factory,
) -> None:
    path = tmp_path / "target.csv"
    _write_predictions(
        path,
        expert="feature_mlp",
        rows=(("target:1", 0.9, 1, "test", True, 10.0),),
    )
    contract = contract_factory(
        "target_hashed",
        role=ContractRole.TARGET,
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
    )
    artifact = _artifact(
        contract,
        path,
        expert="feature_mlp",
        protocol_id="strict_inductive",
    )
    registry = load_protocol_registry(REGISTRY)
    bindings = validate_protocol_bindings((artifact,), registry)
    assert artifact.protocol_id == "strict_inductive"
    assert artifact.contract_coordinate_hash == contract.coordinate_hash
    assert artifact.contract_id == contract.contract_id
    assert ":" in artifact.contract_id
    assert bindings[0]["target_protocol_id"] == "strict_inductive"
    assert bindings[0]["target_contract_id"] == contract.contract_id

    collision = replace(artifact, protocol_id="isolated_inductive")
    with pytest.raises(ValueError, match="protocol|alias|visibility"):
        validate_protocol_bindings((artifact, collision), registry)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema_version="v3"), "V4"),
        (
            lambda value: value.update(
                frozen_before_manifest_conversion=False
            ),
            "frozen",
        ),
        (lambda value: value.update(protocols=[]), "protocol"),
        (
            lambda value: value["protocols"][0].update(
                allowed_contract_roles=[]
            ),
            "roles",
        ),
        (
            lambda value: value["protocols"][0].update(
                visibility_profile="unknown"
            ),
            "visibility",
        ),
    ),
)
def test_protocol_registry_fails_closed_on_invalid_frozen_metadata(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_protocol_registry(path)


def test_target_scope_filters_mixed_rows_and_never_coerces_unknown_labels(
    tmp_path: Path,
    contract_factory,
) -> None:
    rows = (
        ("source:train", 0.1, 2, "train", True, 1.0),
        ("target:unknown", 0.01, 0, "test", False, 10.0),
        ("target:fraud", 0.9, 1, "test", True, 11.0),
        ("unscored:1", 0.5, 0, "unscored", False, 12.0),
    )
    artifacts = []
    contract = contract_factory(
        "target_scope",
        role=ContractRole.TARGET,
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
    )
    for expert in ("feature_mlp", "gcn"):
        path = tmp_path / f"{expert}.csv"
        _write_predictions(path, expert=expert, rows=rows)
        artifacts.append(
            _artifact(
                contract,
                path,
                expert=expert,
                protocol_id="strict_inductive",
            )
        )
    aligned = align_artifact_group(artifacts)
    assert aligned.identifiers.tolist() == ["target:fraud"]
    assert aligned.labels.tolist() == [1]
    assert aligned.label_known.tolist() == [True]
    assert aligned.splits.tolist() == ["test"]
    assert aligned.audit["raw_rows"] == 4
    assert aligned.audit["excluded_unknown_label_rows"] == 1
    assert aligned.audit["excluded_split_rows"] == 2


def test_unknown_source_supervision_fails_closed(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory(
        "source_unknown",
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
    )
    artifacts = []
    for expert in ("feature_mlp", "gcn"):
        path = tmp_path / f"{expert}.csv"
        _write_predictions(
            path,
            expert=expert,
            rows=(
                ("source:unknown", 0.2, 0, "train", False, 1.0),
                ("source:known", 0.8, 1, "validation", True, 2.0),
            ),
        )
        artifacts.append(
            _artifact(
                contract,
                path,
                expert=expert,
                protocol_id="strict_inductive",
            )
        )
    with pytest.raises(ValueError, match="known|unknown"):
        align_artifact_group(artifacts)


def test_evidenced_legacy_val_token_normalizes_to_validation(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory(
        "source_val_alias",
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
    )
    artifacts = []
    for expert in ("feature_mlp", "gcn", "graphsage"):
        path = tmp_path / f"{expert}.csv"
        _write_predictions(
            path,
            expert=expert,
            rows=(
                ("train:1", 0.1, 2, "train", True, 1.0),
                ("val:1", 0.9, 1, "val", True, 2.0),
            ),
        )
        artifacts.append(
            _artifact(
                contract,
                path,
                expert=expert,
                protocol_id="strict_inductive",
            )
        )
    aligned = align_artifact_group(artifacts)
    assert aligned.splits.tolist() == ["train", "validation"]
    assert aligned.audit["provider_split_mapping"]["val"] == "validation"


def test_provider_unknown_cannot_be_declared_known(
    tmp_path: Path,
    contract_factory,
) -> None:
    contract = contract_factory(
        "target_bad_known",
        role=ContractRole.TARGET,
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
    )
    artifacts = []
    for expert in ("feature_mlp", "gcn"):
        path = tmp_path / f"{expert}.csv"
        _write_predictions(
            path,
            expert=expert,
            rows=(("target:unknown", 0.1, 0, "test", True, 1.0),),
        )
        artifacts.append(
            _artifact(
                contract,
                path,
                expert=expert,
                protocol_id="strict_inductive",
            )
        )
    with pytest.raises(ValueError, match="provider-unknown"):
        align_artifact_group(artifacts)


def test_cross_role_overlap_and_coordinate_equivalence_are_atomic() -> None:
    source = PredictionArtifactScope(
        dataset="fixture",
        expert_id="feature_mlp",
        protocol_id="strict_inductive",
        expert_prediction_seed=1,
        fold="fold0",
        role="source",
        contract_coordinate_hash="a" * 64,
        contract_id="source:" + "1" * 16,
        path="/tmp/source.csv",
        checksum="b" * 64,
        original_checksum="b" * 64,
        identifiers=("shared",),
        splits=("validation",),
        label_known=(True,),
        timestamps=(5.0,),
    )
    target = replace(
        source,
        role="target",
        contract_id="target:" + "2" * 16,
        path="/tmp/target.csv",
        checksum="c" * 64,
        splits=("test",),
        timestamps=(6.0,),
    )
    reports = audit_cross_role_prediction_scopes((source, target))
    assert len(reports) == 1
    assert not reports[0].passed
    assert {
        finding.code for finding in reports[0].findings
    } >= {"ATOMIC_ID_OVERLAP", "HELD_OUT_COORDINATE_EQUIVALENCE"}


def test_seed_numbers_are_never_pooled_across_datasets() -> None:
    rows = []
    for dataset in ("elliptic", "dgraphfin"):
        for method, value in (("coregraph", 0.8), ("baseline", 0.6)):
            rows.append(
                {
                    "dataset": dataset,
                    "target_protocol_id": "strict_inductive",
                    "expert_prediction_seed": 1,
                    "fold": "fold0",
                    "method": method,
                    "metric": "auprc",
                    "value": value,
                }
            )
    with pytest.raises(ValueError, match="dataset-stratified"):
        build_paired_seed_blocks(
            rows,
            method="coregraph",
            baseline="baseline",
            metric="auprc",
        )


def test_end_to_end_v4_no_training_runner_to_gate(
    tmp_path: Path,
    contract_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_paths: list[Path] = []
    contracts = (
        (
            "strict_inductive",
            contract_factory(
                "source_strict",
                visibility=VisibilityAxis.STRICT_INDUCTIVE,
            ),
            (
                ("source:s1", 0.8, 1, "train", True, 1.0),
                ("source:s2", 0.2, 2, "validation", True, 2.0),
            ),
        ),
        (
            "isolated_inductive",
                contract_factory(
                    "source_isolated",
                    visibility=VisibilityAxis.ISOLATED_INDUCTIVE,
                    construction=ConstructionSpec(
                        ConstructionAxis.NO_GRAPH
                    ),
            ),
            (
                ("source:i1", 0.7, 1, "train", True, 1.0),
                ("source:i2", 0.3, 2, "validation", True, 2.0),
            ),
        ),
        (
            "transductive_structure",
            contract_factory(
                "target_transductive",
                role=ContractRole.TARGET,
                visibility=VisibilityAxis.TRANSDUCTIVE_STRUCTURE,
            ),
            (
                ("target:t0", 0.01, 0, "test", False, 9.0),
                ("target:t1", 0.9, 1, "test", True, 10.0),
                ("source:excluded", 0.2, 2, "train", True, 2.0),
            ),
        ),
    )
    for protocol_id, contract, rows in contracts:
        for expert in ("feature_mlp", "gcn", "graphsage"):
            csv_path = tmp_path / f"{contract.environment_id}_{expert}.csv"
            _write_predictions(csv_path, expert=expert, rows=rows)
            artifact = _artifact(
                contract,
                csv_path,
                expert=expert,
                protocol_id=protocol_id,
            )
            payload = {
                "schema_version": "coregraph_prediction_manifest_v4",
                "expert_id": artifact.expert_id,
                "dataset": artifact.dataset,
                "task": artifact.task,
                "prediction_unit": artifact.prediction_unit,
                "protocol_id": artifact.protocol_id,
                "contract_coordinate_hash": artifact.contract_coordinate_hash,
                "contract_id": artifact.contract_id,
                "environment_id": artifact.environment_id,
                "expert_prediction_seed": artifact.seed,
                "fold": artifact.fold,
                "prediction_path": str(csv_path),
                "prediction_checksum": artifact.checksum,
                "config_hash": artifact.config_hash,
                "code_hash": artifact.code_hash,
                "contract_role": artifact.contract_role,
                "deployment_contract": contract.to_dict(),
                "expert_available": True,
                "availability_reason_codes": ["available"],
                "compute_cost": 1.0,
                "compute_cost_provenance": artifact.compute_cost_provenance,
                "score_type": "PROBABILITY",
                "permitted_splits": list(artifact.permitted_splits),
                "evaluation_split": artifact.evaluation_split,
                "row_scope_policy": artifact.row_scope_policy,
                "label_mapping": dict(artifact.label_mapping),
                "positive_label_id": artifact.positive_label_id,
                "provider_split_mapping": dict(
                    artifact.provider_split_mapping
                ),
                "original_prediction_path": artifact.original_prediction_path,
                "original_prediction_checksum": artifact.original_prediction_checksum,
            }
            manifest = tmp_path / (
                f"{contract.environment_id}_{expert}_prediction_manifest.json"
            )
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_paths.append(manifest)

    loaded = load_prediction_artifacts(manifest_paths)
    groups = validate_artifact_groups(
        loaded,
        expected_experts=("feature_mlp", "gcn", "graphsage"),
        expected_seeds=(1,),
        expected_datasets=("fixture",),
        expected_target_protocols=("transductive_structure",),
    )
    gate_schema = {
        "schema_version": "coregraph_pilot_gate_v4",
        "required_datasets": ["fixture"],
        "required_target_protocols": ["transductive_structure"],
        "required_expert_prediction_seeds": [1],
        "required_folds": ["fold0"],
        "required_experts": ["feature_mlp", "gcn", "graphsage"],
        "strong_baselines": ["average_all_feasible"],
        "compatibility_components": [],
        "diagnostic_comparators": [],
        "offline_oracles": [
            "contract_feasible_oracle",
            "instance_clairvoyant_oracle_ceiling",
        ],
        "required_ablations": [],
        "required_contract_metrics": ["auprc", "brier_contract_regret"],
        "headline_risk": "brier_contract_regret",
    }
    materialized = build_no_training_materialization(
        loaded,
        groups,
        gate_schema=gate_schema,
        registry=load_protocol_registry(REGISTRY),
    )
    assert materialized["status"] == "VALIDATED_NO_TRAINING"
    assert materialized["training_performed"] is False
    assert materialized["metric_computation_performed"] is False
    target_reports = [
        report
        for report in materialized["row_scope_reports"]
        if report["contract_role"] == "target"
    ]
    assert target_reports
    assert all(
        report["excluded_unknown_label_rows"] == 1
        for report in target_reports
    )
    target_contract = contracts[2][1]
    assert {
        row["target_protocol_id"] for row in materialized["planned_rows"]
    } == {"transductive_structure"}
    assert {
        row["target_contract_id"] for row in materialized["planned_rows"]
    } == {target_contract.contract_id}
    assert {
        row["target_contract_coordinate_hash"]
        for row in materialized["planned_rows"]
    } == {target_contract.coordinate_hash}
    gate = validate_no_training_completeness(materialized, gate_schema)
    assert gate["status"] == "NO_TRAINING_COMPLETENESS_VALIDATED"
    assert gate["complete"]

    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate_schema), encoding="utf-8")
    validation_output = tmp_path / "validation_output.json"
    config = tmp_path / "pilot_config.json"
    config.write_text(
        json.dumps(
            {
                "prediction_manifest_roots": [str(tmp_path)],
                "required_datasets": ["fixture"],
                "required_target_protocols": [
                    "transductive_structure"
                ],
                "required_seeds": [1],
                "required_experts": [
                    "feature_mlp",
                    "gcn",
                    "graphsage",
                ],
                "output": str(tmp_path / "forbidden_execution.json"),
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
            "--gate-schema",
            str(gate_path),
        ],
    )
    assert runner_cli.main() == 0
    cli_materialization = json.loads(
        validation_output.read_text(encoding="utf-8")
    )
    assert cli_materialization["status"] == "VALIDATED_NO_TRAINING"
    assert not (tmp_path / "forbidden_execution.json").exists()


def test_converter_blocks_unresolved_metadata_without_inventing_it(
    tmp_path: Path,
) -> None:
    prediction = (
        tmp_path / "elliptic__strict_inductive__mlp__seed1.csv"
    )
    _write_historical_predictions(prediction)
    checksum = hashlib.sha256(prediction.read_bytes()).hexdigest()
    report_dir = tmp_path / "validation"
    report_dir.mkdir()
    (report_dir / "prediction_validation_report.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": prediction.name,
                        "sha256": checksum,
                        "ok": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidates = discover_historical_predictions((tmp_path,))
    audits = audit_candidates(
        candidates,
        (
            (
                report_dir / "prediction_validation_report.json",
                json.loads(
                    (
                        report_dir / "prediction_validation_report.json"
                    ).read_text(encoding="utf-8")
                ),
            ),
        ),
    )
    assert len(audits) == 1
    assert audits[0].split_counts == {"test": 2, "train": 1}
    assert audits[0].excluded_unknown_label_counts == {"test": 1}
    records = build_conversion_records(audits, None)
    assert records[0]["conversion_status"] == "BLOCKED_METADATA_UNRESOLVED"
    assert records[0]["manifest"] is None
    assert "deployment_contract" in records[0]["unresolved_fields"]
    matrix = build_completeness_matrix(records)
    cell = next(
        row
        for row in matrix
        if row["dataset"] == "elliptic"
        and row["protocol_id"] == "strict_inductive"
        and row["expert_id"] == "feature_mlp"
        and row["expert_prediction_seed"] == 1
        and row["contract_role"] == "target"
    )
    assert cell["status"] == "BLOCKED_METADATA_UNRESOLVED"


def test_converter_discovers_input_style_validation_and_ignores_bad_json(
    tmp_path: Path,
) -> None:
    prediction = (
        tmp_path / "elliptic__strict_inductive__mlp__seed1.csv"
    )
    _write_historical_predictions(prediction)
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "prediction_validation_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "issues": [],
                "inputs": [prediction.name],
            }
        ),
        encoding="utf-8",
    )
    (bad / "prediction_validation_report.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    reports = discover_validation_evidence((tmp_path,))
    assert len(reports) == 1
    audit = audit_candidates(
        discover_historical_predictions((tmp_path,)),
        reports,
    )[0]
    assert audit.validated_export
    assert audit.structurally_usable
    assert audit.validation_evidence[0].status == (
        "VALIDATED_REPORT_PATH_MATCH_CHECKSUM_RECOMPUTED"
    )


def test_converter_emits_v4_only_with_complete_explicit_evidence(
    tmp_path: Path,
    contract_factory,
) -> None:
    prediction = (
        tmp_path / "elliptic__strict_inductive__mlp__seed1.csv"
    )
    _write_historical_predictions(prediction)
    checksum = hashlib.sha256(prediction.read_bytes()).hexdigest()
    candidate = discover_historical_predictions((tmp_path,))[0]
    audit = audit_candidates(
        (candidate,),
        (
            (
                tmp_path / "prediction_validation_report.json",
                {
                    "files": [
                        {
                            "path": prediction.name,
                            "sha256": checksum,
                            "ok": True,
                        }
                    ]
                },
            ),
        ),
    )[0]
    contract = replace(
        contract_factory(
            "elliptic_target_strict",
            role=ContractRole.TARGET,
            visibility=VisibilityAxis.STRICT_INDUCTIVE,
        ),
        dataset_id="elliptic",
    )
    evidence = {
        "schema_version": "coregraph_manifest_conversion_evidence_v4",
        "artifacts": [
            {
                "original_prediction_path": str(prediction),
                "original_prediction_checksum": checksum,
                "contract_role": "target",
                "deployment_contract": contract.to_dict(),
                "config_hash": "a" * 64,
                "code_hash": "b" * 40,
                "compute_cost": 1.0,
                "compute_cost_provenance": "fixture_measured",
            }
        ],
    }
    record = build_conversion_records((audit,), evidence)[0]
    assert record["conversion_status"] == "CONVERTED_V4"
    manifest = record["manifest"]
    assert manifest["protocol_id"] == "strict_inductive"
    assert manifest["contract_coordinate_hash"] == contract.coordinate_hash
    assert manifest["contract_id"] == contract.contract_id
    assert manifest["original_prediction_checksum"] == checksum
    assert manifest["provider_split_mapping"]["val"] == "validation"


def test_converter_cli_writes_blocked_no_training_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = (
        tmp_path / "elliptic__strict_inductive__mlp__seed1.csv"
    )
    _write_historical_predictions(prediction)
    checksum = hashlib.sha256(prediction.read_bytes()).hexdigest()
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "prediction_validation_report.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": prediction.name,
                        "sha256": checksum,
                        "ok": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "converted"
    build = tmp_path / "build"
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert_prediction_manifests_v4.py",
            "--root",
            str(tmp_path),
            "--output-root",
            str(output),
            "--build-root",
            str(build),
            "--protocol-registry",
            str(REGISTRY),
        ],
    )
    assert converter_cli.main() == 0
    status = json.loads(
        (output / "no_training_audit_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["training_performed"] is False
    assert status["metric_computation_performed"] is False
    assert status["contract_registry_audit"][
        "registry_schema_status"
    ] == "PASS_FROZEN_V4_REGISTRY"
    assert status["no_training_runner_status"].startswith("BLOCKED_")
    assert (build / "MANIFEST_COMPLETENESS_MATRIX.csv").is_file()
    assert (build / "MANIFEST_LEAKAGE_AUDIT.json").is_file()


def test_v4_inference_and_taxonomy_are_dataset_stratified() -> None:
    schema = json.loads(
        V4_GATE.read_text(encoding="utf-8")
    )
    gate._validate_gate_schema(schema)
    assert (
        "graphsafe_confidence_abstention_component"
        not in schema["strong_baselines"]
    )
    assert schema["compatibility_components"] == [
        "graphsafe_confidence_abstention_component"
    ]
    rows = []
    for dataset, improvement in (("elliptic", 0.2), ("dgraphfin", 0.1)):
        for seed in range(1, 11):
            for method, value in (
                ("full_corerouter", 0.5 + improvement),
                ("average_all_feasible", 0.5),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "target_protocol_id": "strict_inductive",
                        "expert_prediction_seed": seed,
                        "fold": "fold0",
                        "method": method,
                        "metric": "auprc",
                        "value": value,
                    }
                )
    elliptic, _ = gate._dataset_effect_record_v4(
        rows,
        method="full_corerouter",
        baseline="average_all_feasible",
        metric="auprc",
        direction="higher",
        minimum_effect=0.0,
        dataset="elliptic",
    )
    dgraphfin, _ = gate._dataset_effect_record_v4(
        rows,
        method="full_corerouter",
        baseline="average_all_feasible",
        metric="auprc",
        direction="higher",
        minimum_effect=0.0,
        dataset="dgraphfin",
    )
    assert elliptic["mean_improvement"] == pytest.approx(0.2)
    assert dgraphfin["mean_improvement"] == pytest.approx(0.1)
    assert elliptic["seeds"] == list(range(1, 11))
    assert dgraphfin["seeds"] == list(range(1, 11))


def test_full_v4_gate_path_keeps_dataset_strata_separate() -> None:
    schema = json.loads(
        V4_GATE.read_text(encoding="utf-8")
    )
    methods = {
        "full_corerouter",
        *(f"expert:{name}" for name in schema["required_experts"]),
        *schema["strong_baselines"],
        *(
            f"ablation:{name}"
            for name in schema["required_ablations"]
        ),
    }
    higher = {
        "auprc",
        "recall_at_0.5pct",
        "recall_at_1pct",
        "recall_at_2pct",
        "budget_curve_area",
    }
    rows = []
    for dataset in schema["required_datasets"]:
        for protocol_id in schema["required_target_protocols"]:
            for seed in schema["required_expert_prediction_seeds"]:
                for method in methods:
                    for metric in schema["required_contract_metrics"]:
                        if metric in higher:
                            value = 0.8 if method == "full_corerouter" else 0.6
                        elif metric in {
                            "brier_contract_regret",
                            "selective_zero_one_risk",
                            "abstention_cost",
                            "compute",
                            "aurc",
                        }:
                            value = 0.1 if method == "full_corerouter" else 0.3
                        else:
                            value = 0.8
                        rows.append(
                            {
                                "dataset": dataset,
                                "target_protocol_id": protocol_id,
                                "seed": seed,
                                "expert_prediction_seed": seed,
                                "router_training_seed": derive_router_seed(
                                    seed,
                                    method,
                                ),
                                "fold": "fold0",
                                "method": method,
                                "metric": metric,
                                "value": value,
                                "execution_status": (
                                    MethodExecutionStatus.EXECUTABLE.value
                                ),
                            }
                        )
    result = gate.evaluate_pilot_gate(
        {
            "rows": rows,
            "routing": [],
            "target_label_selection": False,
            "oracle_target_selection": False,
            "headline_oracle": "contract_feasible_oracle",
            "diagnostic_oracle": "instance_clairvoyant_oracle_ceiling",
        },
        schema,
    )
    assert result["status"] == "GATE_EVALUATED_V4"
    assert result["coverage"]["complete"]
    assert {
        record["dataset"]
        for record in result["dataset_stratified_comparisons"]
    } == {"elliptic", "dgraphfin"}
    assert result["inferential_block"] == (
        "dataset_stratified_expert_prediction_seed"
    )
    blocked = gate.evaluate_pilot_gate({"rows": []}, schema)
    assert blocked["status"] == "BLOCKED_INCOMPLETE_PILOT_COVERAGE"
