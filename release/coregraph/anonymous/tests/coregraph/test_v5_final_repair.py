from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from coregraph.evaluation.regret import (
    contract_regrets,
    feasible_row_oracle_brier_with_abstention,
    regret_summary,
    v5_matched_action_brier_metrics,
)
from coregraph.experiments.v5_package_validator import (
    PackageValidationError,
    validate_package_root,
    write_package_validation_artifacts,
)
from coregraph.experiments.v5_pilot_outputs import (
    METHOD_RESULT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    POLICY_FREEZE_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    atomic_write_csv,
    atomic_write_npz,
    build_effective_execution_config,
    coordinate_identity_hash,
    mark_complete,
    reusable_complete,
)
from coregraph.experiments.v5_numerics import (
    NUMERICAL_IMPLEMENTATION_VERSION,
    SCIENTIFIC_COMPUTE_DTYPE,
)
from coregraph.experiments.v5_pilot_types import (
    METHOD_REGISTRY_VERSION,
    METRIC_SCHEMA_VERSION,
    PilotCoordinate,
)
from coregraph.utils.io import atomic_write_json, sha256_path


ROOT = Path(__file__).resolve().parents[2]
CODE_SHA = "a" * 40
BASE_CONFIG_SHA = "b" * 64
PREREGISTRATION_SHA = "c" * 64
DEPENDENCY_SHA = "d" * 64


def _effective(**overrides):
    values = {
        "base_config_sha256": BASE_CONFIG_SHA,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "configured_chunk_rows": 50_000,
        "effective_chunk_rows": 50_000,
        "max_workers": 1,
        "execution_mode": "synthetic",
        "synthetic_fixture": True,
        "dependency_lock_sha256": DEPENDENCY_SHA,
        "code_sha": CODE_SHA,
    }
    values.update(overrides)
    return build_effective_execution_config(**values)


def test_effective_execution_config_binds_every_effective_setting() -> None:
    base = _effective()
    assert base == _effective()
    variants = (
        _effective(effective_chunk_rows=7),
        _effective(execution_mode="real", synthetic_fixture=False),
        _effective(max_workers=2),
        _effective(output_schema_version="changed-output-schema"),
        _effective(metric_schema_version="changed-metric-schema"),
    )
    hashes = {base["effective_execution_config_sha256"]}
    hashes.update(item["effective_execution_config_sha256"] for item in variants)
    assert len(hashes) == 6
    assert base["effective_execution_config_sha256"] != BASE_CONFIG_SHA


def test_resume_rejects_changed_effective_execution_identity(tmp_path: Path) -> None:
    output, coordinate, _ = _build_valid_package(tmp_path)
    method_root = output / "scenarios/scenario-1/methods/coregraph"
    changed = PilotCoordinate(
        **{
            **asdict(coordinate),
            "effective_execution_config_sha256": "f" * 64,
        }
    )
    changed_identity = coordinate_identity_hash(
        changed,
        code_sha=CODE_SHA,
        config_sha256=BASE_CONFIG_SHA,
        preregistration_sha256=PREREGISTRATION_SHA,
        dependency_lock_sha256=DEPENDENCY_SHA,
        effective_execution_config_sha256="f" * 64,
    )
    reusable, reasons = reusable_complete(
        method_root,
        coordinate=changed,
        identity_hash=changed_identity,
    )
    assert not reusable
    assert "coordinate_key_mismatch" in reasons
    assert "identity_hash_mismatch" in reasons
    assert "effective_execution_config_mismatch" in reasons


def test_matched_action_regret_semantics_and_fixed_diagnostic() -> None:
    labels = np.asarray([0, 1, 1], dtype=np.int8)
    experts = np.asarray([[0.0, 0.9], [0.1, 1.0], [0.2, 0.3]])
    available = np.asarray([[True, True], [True, True], [False, False]])
    oracle = feasible_row_oracle_brier_with_abstention(
        labels=labels,
        expert_scores=experts,
        availability=available,
        abstention_cost=0.2,
    )
    np.testing.assert_allclose(oracle, [0.0, 0.0, 0.2])
    metrics = v5_matched_action_brier_metrics(
        labels=labels,
        method_scores=np.asarray([0.0, 1.0, 0.5]),
        method_abstains=np.asarray([False, False, True]),
        expert_scores=experts,
        availability=available,
        abstention_cost=0.2,
    )
    assert metrics["contract_regret_vs_feasible_row_oracle"] == 0.0
    assert metrics["feasible_row_oracle_loss_with_abstention"] == pytest.approx(
        0.2 / 3
    )
    assert metrics["best_fixed_nonabstaining_expert_brier"] is None
    assert metrics["excess_cost_vs_best_fixed_nonabstaining_expert"] is None


def test_oracle_uses_abstention_and_excludes_unavailable_experts() -> None:
    oracle = feasible_row_oracle_brier_with_abstention(
        labels=np.asarray([0, 1]),
        expert_scores=np.asarray([[0.9, 0.0], [0.0, 0.1]]),
        availability=np.asarray([[True, False], [False, True]]),
        abstention_cost=0.2,
    )
    np.testing.assert_allclose(oracle, [0.2, 0.2])


def test_negative_regret_below_tolerance_fails_closed() -> None:
    with pytest.raises(ValueError, match="below the frozen numeric tolerance"):
        v5_matched_action_brier_metrics(
            labels=np.asarray([0]),
            method_scores=np.asarray([0.0]),
            method_abstains=np.asarray([False]),
            expert_scores=np.asarray([[0.5, 0.8]]),
            availability=np.asarray([[True, True]]),
            abstention_cost=0.2,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("oracle_shape", "aligned matrices"),
        ("oracle_rows", "align with labels"),
        ("bad_labels", "finite binary"),
        ("bad_experts", "finite probabilities"),
        ("bad_cost", "finite and non-negative"),
        ("empty", "empty target"),
        ("method_shape", "outputs do not align"),
        ("bad_method", "method scores must be finite"),
    ),
)
def test_regret_inputs_fail_closed(mutation: str, match: str) -> None:
    labels = np.asarray([0, 1])
    experts = np.asarray([[0.1, 0.9], [0.2, 0.8]])
    availability = np.ones_like(experts, dtype=bool)
    if mutation.startswith("oracle") or mutation in {"bad_labels", "bad_experts", "bad_cost"}:
        oracle_labels = np.asarray([0, 2]) if mutation == "bad_labels" else labels
        oracle_experts = np.asarray([[np.nan, 0.9], [0.2, 0.8]]) if mutation == "bad_experts" else experts
        oracle_availability = availability
        if mutation == "oracle_shape":
            oracle_availability = np.ones((2, 1), dtype=bool)
        elif mutation == "oracle_rows":
            oracle_experts = experts[:1]
            oracle_availability = availability[:1]
        with pytest.raises(ValueError, match=match):
            feasible_row_oracle_brier_with_abstention(
                labels=oracle_labels,
                expert_scores=oracle_experts,
                availability=oracle_availability,
                abstention_cost=-1.0 if mutation == "bad_cost" else 0.2,
            )
        return
    method_labels = np.asarray([]) if mutation == "empty" else labels
    method_scores = np.asarray([0.1]) if mutation == "method_shape" else np.asarray([0.1, 0.9])
    if mutation == "bad_method":
        method_scores[0] = np.nan
    with pytest.raises(ValueError, match=match):
        v5_matched_action_brier_metrics(
            labels=method_labels,
            method_scores=method_scores,
            method_abstains=np.zeros(len(method_scores), dtype=bool),
            expert_scores=experts,
            availability=availability,
            abstention_cost=0.2,
        )


def test_legacy_loss_parametric_regret_helpers() -> None:
    regrets = contract_regrets(
        {"a": 0.3, "b": 0.4},
        {"a": {"x": 0.1, "y": 0.2}, "b": {"x": 0.5, "y": 0.2}},
        {"a": {"x": True, "y": False}, "b": {"x": False, "y": True}},
    )
    assert regrets == pytest.approx({"a": 0.2, "b": 0.2})
    assert regret_summary(regrets)["maximum_contract_regret"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no feasible oracle"):
        contract_regrets({"a": 0.3}, {"a": {"x": 0.1}}, {"a": {"x": False}})
    with pytest.raises(ValueError, match="empty regret"):
        regret_summary({})


def _artifact(index: int) -> dict[str, str]:
    return {
        "archive_name": f"archive-{index}.zip",
        "archive_sha256": f"{index + 10:064x}",
        "member_name": f"predictions/member-{index}.csv",
        "member_sha256": f"{index + 100:064x}",
    }


def _build_valid_package(root: Path) -> tuple[Path, PilotCoordinate, dict[str, object]]:
    output = root / "run"
    effective = _effective()
    effective_hash = str(effective["effective_execution_config_sha256"])
    coordinate = PilotCoordinate(
        dataset="elliptic",
        target_protocol="strict_inductive",
        provider_seed=1,
        method="coregraph",
        pilot_specification_version="coregraph_saved_output_pilot_v5.2",
        scenario_id="scenario-1",
        scenario_fingerprint="e" * 64,
        effective_execution_config_sha256=effective_hash,
    )
    row = {
        "coordinate_key": coordinate.key,
        "dataset": coordinate.dataset,
        "target_protocol": coordinate.target_protocol,
        "provider_seed": coordinate.provider_seed,
        "method": coordinate.method,
        "pilot_specification_version": coordinate.pilot_specification_version,
        "scenario_id": coordinate.scenario_id,
        "scenario_fingerprint": coordinate.scenario_fingerprint,
        "effective_execution_config_sha256": effective_hash,
        "status": "PLANNED",
    }
    atomic_write_csv(output / "PILOT_PLAN.csv", [row])
    archive_hashes = {
        item["archive_name"]: item["archive_sha256"]
        for item in (_artifact(index) for index in range(9))
    }
    run_manifest: dict[str, object] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "repository_sha": CODE_SHA,
        "base_config_sha256": BASE_CONFIG_SHA,
        "effective_execution_config": effective,
        "effective_execution_config_sha256": effective_hash,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "dependency_lock_sha256": DEPENDENCY_SHA,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "coordinate_keys": [coordinate.key],
        "coordinate_count": 1,
        "archive_hashes": archive_hashes,
    }
    atomic_write_json(output / "RUN_MANIFEST.json", run_manifest)
    method_root = output / "scenarios/scenario-1/methods/coregraph"
    atomic_write_json(method_root / "fit_report.json", {"schema": "fixture"})
    atomic_write_npz(
        method_root / "target_scores.npz",
        scores=np.asarray([0.2], dtype=np.float64),
        routing_weights=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        expected_compute=np.asarray([1.0], dtype=np.float64),
    )
    atomic_write_json(method_root / "route_summary.json", {"coverage": 1.0})
    artifacts = [_artifact(index) for index in range(9)]
    freeze = {
        "schema": POLICY_FREEZE_SCHEMA,
        "code_sha": CODE_SHA,
        "base_config_sha256": BASE_CONFIG_SHA,
        "effective_execution_config_sha256": effective_hash,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "dependency_lock_sha256": DEPENDENCY_SHA,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "source_artifacts": artifacts[:6],
        "target_artifacts": artifacts[6:],
    }
    atomic_write_json(method_root / "POLICY_FREEZE_MANIFEST.json", freeze)
    identity_hash = coordinate_identity_hash(
        coordinate,
        code_sha=CODE_SHA,
        config_sha256=BASE_CONFIG_SHA,
        preregistration_sha256=PREREGISTRATION_SHA,
        dependency_lock_sha256=DEPENDENCY_SHA,
        effective_execution_config_sha256=effective_hash,
    )
    evaluation = {
        "schema": METHOD_RESULT_SCHEMA,
        "coordinate": asdict(coordinate),
        "coordinate_key": coordinate.key,
        "identity_hash": identity_hash,
        "code_sha": CODE_SHA,
        "base_config_sha256": BASE_CONFIG_SHA,
        "effective_execution_config_sha256": effective_hash,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "dependency_lock_sha256": DEPENDENCY_SHA,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "metrics": {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "global_target_auprc": 0.5,
            "contract_regret_vs_feasible_row_oracle": 0.0,
            "scientific_compute_dtype": SCIENTIFIC_COMPUTE_DTYPE,
            "rows_with_raw_regret_below_tolerance": 0,
            "rows_with_unavailable_nonzero_weight": 0,
        },
        "policy_freeze_sha256": sha256_path(
            method_root / "POLICY_FREEZE_MANIFEST.json"
        ),
        "target_score_sha256": sha256_path(method_root / "target_scores.npz"),
        "route_summary_sha256": sha256_path(method_root / "route_summary.json"),
    }
    atomic_write_json(method_root / "evaluation.json", evaluation)
    mark_complete(
        method_root,
        coordinate=coordinate,
        identity_hash=identity_hash,
        outputs=(
            "fit_report.json",
            "POLICY_FREEZE_MANIFEST.json",
            "target_scores.npz",
            "route_summary.json",
            "evaluation.json",
        ),
        retry_count=0,
    )
    return output, coordinate, run_manifest


def _rewrite_json(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    atomic_write_json(path, payload)


def test_exact_package_validator_and_post_extraction_package(tmp_path: Path) -> None:
    output, coordinate, _ = _build_valid_package(tmp_path)
    report, rows = validate_package_root(output)
    assert report["status"] == "PASS"
    assert rows[0]["coordinate_key"] == coordinate.key
    write_package_validation_artifacts(output, report, rows)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/coregraph/run_saved_output_pilot_v5.py"),
            "--package",
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    package_report = json.loads(
        (output / "PACKAGE_VALIDATION_REPORT.json").read_text(encoding="utf-8")
    )
    assert package_report["post_extraction_validation"] == "PASS"
    assert package_report["zip_crc_validation"] == "PASS"
    assert (tmp_path / "run.zip").is_file()
    assert validate_package_root(output)[0]["status"] == "PASS"


@pytest.mark.parametrize(
    "field,expected_failure",
    [
        ("scores", "stored_score_dtype_invalid"),
        ("routing_weights", "stored_weight_dtype_invalid"),
        ("expected_compute", "stored_compute_dtype_invalid"),
    ],
)
def test_package_rejects_each_float32_scientific_array(
    tmp_path: Path, field: str, expected_failure: str
) -> None:
    output, _, _ = _build_valid_package(tmp_path)
    path = output / "scenarios/scenario-1/methods/coregraph/target_scores.npz"
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays[field] = arrays[field].astype(np.float32)
    atomic_write_npz(path, **arrays)
    with pytest.raises(PackageValidationError) as raised:
        validate_package_root(output)
    assert expected_failure in str(raised.value)


def test_package_artifact_writer_rejects_empty_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty package coordinate manifest"):
        write_package_validation_artifacts(tmp_path, {"status": "PASS"}, [])


def test_output_checksum_failures_are_detected_in_process(tmp_path: Path) -> None:
    output, _, _ = _build_valid_package(tmp_path)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/coregraph/run_saved_output_pilot_v5.py"),
            "--package",
            "--output-root",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    checksum_path = output / "OUTPUT_CHECKSUMS.sha256"
    original = checksum_path.read_text(encoding="utf-8")
    headers = [line for line in original.splitlines() if line.startswith("# ")]
    records = [line for line in original.splitlines() if line and not line.startswith("# ")]
    mutations = (
        ("\n".join([*headers, *records, records[0]]) + "\n", "duplicate_output_checksum"),
        ("\n".join([*headers, *records[1:]]) + "\n", "output_checksum_path_set_mismatch"),
        ("\n".join([*headers, "0" * 64 + records[0][64:], *records[1:]]) + "\n", "output_checksum_mismatch"),
        (original.replace("# effective_execution_config_sha256=", "# effective_execution_config_sha256=wrong-"), "checksum_manifest_effective_hash_mismatch"),
    )
    for content, expected in mutations:
        checksum_path.write_text(content, encoding="utf-8")
        with pytest.raises(PackageValidationError, match=expected):
            validate_package_root(output)
    checksum_path.write_text(original, encoding="utf-8")


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("missing", "missing_coordinate_directory"),
        ("extra", "extra_coordinate_directory"),
        ("duplicate", "duplicate_coordinate_in_plan"),
        ("stale_code", "evaluation_code_sha_mismatch"),
        ("stale_config", "evaluation_base_config_sha256_mismatch"),
        ("stale_effective", "evaluation_effective_execution_config_sha256_mismatch"),
        ("stale_prereg", "evaluation_preregistration_sha256_mismatch"),
        ("wrong_method", "coordinate_payload_mismatch"),
        ("wrong_scenario", "coordinate_payload_mismatch"),
        ("wrong_fingerprint", "coordinate_payload_mismatch"),
        ("missing_evaluation", "missing_method_files"),
        ("wrong_checksum", "coordinate_checksum_mismatch"),
        ("failure", "unresolved_failure"),
        ("fake_complete", "extra_coordinate_directory"),
        ("partial", "temporary_partial_files_present"),
        ("mixed_mode", "effective_execution_config_payload_hash_mismatch"),
        ("invalid_json_manifest", "invalid_json:RUN_MANIFEST.json"),
        ("nonmapping_manifest", "invalid_mapping:RUN_MANIFEST.json"),
        ("missing_plan", "missing:PILOT_PLAN.csv"),
        ("invalid_plan", "plan_schema_invalid"),
        ("duplicate_manifest", "duplicate_coordinate_in_run_manifest"),
        ("manifest_set", "run_manifest_coordinate_set_mismatch"),
        ("manifest_count", "run_manifest_coordinate_count_mismatch"),
        ("manifest_schema", "run_manifest_schema_invalid"),
        ("output_schema", "run_manifest_output_schema_invalid"),
        ("metric_schema", "run_manifest_metric_schema_invalid"),
        ("method_registry", "run_manifest_method_registry_invalid"),
        ("effective_missing", "effective_execution_config_missing"),
        ("global_effective", "effective_execution_config_manifest_hash_mismatch"),
        ("coordinate_nonmapping", "coordinate_payload_invalid"),
        ("coordinate_invalid", "coordinate_invalid"),
        ("superseded_metric", "superseded_metric_present"),
        ("checksums_nonmapping", "checkpoint_checksums_invalid"),
        ("artifact_nonmapping", "artifact_identity_invalid"),
        ("archive_mismatch", "archive_identity_mismatch"),
        ("gate_effective", "gate_effective_hash_mismatch"),
        ("gate_metric", "gate_metric_schema_mismatch"),
    ),
)
def test_package_validator_rejects_all_mutations(
    tmp_path: Path, case: str, expected: str
) -> None:
    output, coordinate, _ = _build_valid_package(tmp_path)
    method_root = output / "scenarios/scenario-1/methods/coregraph"
    evaluation_path = method_root / "evaluation.json"
    if case == "missing":
        shutil.rmtree(method_root)
    elif case in {"extra", "fake_complete"}:
        extra = output / "scenarios/scenario-extra/methods/coregraph"
        extra.mkdir(parents=True)
        if case == "fake_complete":
            (extra / "COMPLETE").write_text("fake\n", encoding="utf-8")
    elif case == "duplicate":
        with (output / "PILOT_PLAN.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        atomic_write_csv(output / "PILOT_PLAN.csv", [row, row])
    elif case == "stale_code":
        _rewrite_json(evaluation_path, code_sha="f" * 40)
    elif case == "stale_config":
        _rewrite_json(evaluation_path, base_config_sha256="f" * 64)
    elif case == "stale_effective":
        _rewrite_json(evaluation_path, effective_execution_config_sha256="f" * 64)
    elif case == "stale_prereg":
        _rewrite_json(evaluation_path, preregistration_sha256="f" * 64)
    elif case in {"wrong_method", "wrong_scenario", "wrong_fingerprint"}:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        field = {
            "wrong_method": "method",
            "wrong_scenario": "scenario_id",
            "wrong_fingerprint": "scenario_fingerprint",
        }[case]
        evaluation["coordinate"][field] = "wrong"
        atomic_write_json(evaluation_path, evaluation)
    elif case == "missing_evaluation":
        evaluation_path.unlink()
    elif case == "wrong_checksum":
        (method_root / "target_scores.npz").write_bytes(b"changed")
    elif case == "failure":
        atomic_write_json(
            output / "scenarios/scenario-1/failures/coregraph.json",
            {"coordinate_key": coordinate.key},
        )
    elif case == "partial":
        (method_root / ".tmp-partial").write_text("partial", encoding="utf-8")
    elif case == "mixed_mode":
        manifest_path = output / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["effective_execution_config"]["execution_mode"] = "real"
        atomic_write_json(manifest_path, manifest)
    elif case == "invalid_json_manifest":
        (output / "RUN_MANIFEST.json").write_text("{bad", encoding="utf-8")
    elif case == "nonmapping_manifest":
        (output / "RUN_MANIFEST.json").write_text("[]\n", encoding="utf-8")
    elif case == "missing_plan":
        (output / "PILOT_PLAN.csv").unlink()
    elif case == "invalid_plan":
        (output / "PILOT_PLAN.csv").write_text("bad\n", encoding="utf-8")
    elif case in {
        "duplicate_manifest", "manifest_set", "manifest_count", "manifest_schema",
        "output_schema", "metric_schema", "method_registry", "effective_missing",
        "global_effective",
    }:
        manifest_path = output / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if case == "duplicate_manifest":
            manifest["coordinate_keys"] = [coordinate.key, coordinate.key]
        elif case == "manifest_set":
            manifest["coordinate_keys"] = ["wrong"]
        elif case == "manifest_count":
            manifest["coordinate_count"] = 2
        elif case == "manifest_schema":
            manifest["schema"] = "old"
        elif case == "output_schema":
            manifest["output_schema_version"] = "old"
        elif case == "metric_schema":
            manifest["metric_schema_version"] = "old"
        elif case == "method_registry":
            manifest["method_registry_version"] = "old"
        elif case == "effective_missing":
            manifest["effective_execution_config"] = []
        elif case == "global_effective":
            manifest["effective_execution_config_sha256"] = "f" * 64
        atomic_write_json(manifest_path, manifest)
    elif case in {"coordinate_nonmapping", "superseded_metric"}:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if case == "coordinate_nonmapping":
            evaluation["coordinate"] = []
        else:
            evaluation["metrics"]["contract_regret"] = 0.0
        atomic_write_json(evaluation_path, evaluation)
    elif case == "coordinate_invalid":
        with (output / "PILOT_PLAN.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        row["effective_execution_config_sha256"] = "short"
        atomic_write_csv(output / "PILOT_PLAN.csv", [row])
    elif case == "checksums_nonmapping":
        _rewrite_json(method_root / "checkpoint.json", checksums=[])
    elif case in {"artifact_nonmapping", "archive_mismatch"}:
        freeze = json.loads((method_root / "POLICY_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
        if case == "artifact_nonmapping":
            freeze["source_artifacts"][0] = "invalid"
        else:
            freeze["source_artifacts"][0]["archive_sha256"] = "f" * 64
        atomic_write_json(method_root / "POLICY_FREEZE_MANIFEST.json", freeze)
    elif case in {"gate_effective", "gate_metric"}:
        gate = {
            "effective_execution_config_sha256": "f" * 64 if case == "gate_effective" else coordinate.effective_execution_config_sha256,
            "metric_schema_version": "old" if case == "gate_metric" else METRIC_SCHEMA_VERSION,
        }
        atomic_write_json(output / "gates/PILOT_GATE_RESULT.json", gate)
    with pytest.raises(PackageValidationError, match=expected):
        validate_package_root(output)
