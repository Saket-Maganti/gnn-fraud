"""Exact-set, identity, and checksum validation for V5 pilot packages."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from coregraph.experiments.v5_pilot_outputs import (
    GATE_SCHEMA,
    METHOD_RESULT_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    POLICY_FREEZE_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    atomic_write_csv,
    canonical_hash,
    coordinate_identity_hash,
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


REQUIRED_METHOD_FILES = frozenset(
    {
        "fit_report.json",
        "POLICY_FREEZE_MANIFEST.json",
        "target_scores.npz",
        "route_summary.json",
        "evaluation.json",
        "checkpoint.json",
        "COMPLETE",
    }
)


@dataclass(frozen=True, slots=True)
class PackageValidationError(RuntimeError):
    failures: tuple[str, ...]

    def __str__(self) -> str:
        return "V5 package validation failed: " + "; ".join(self.failures)


def _json(path: Path, failures: list[str]) -> Mapping[str, Any]:
    if not path.is_file():
        failures.append(f"missing:{path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        failures.append(f"invalid_json:{path.name}:{type(exc).__name__}")
        return {}
    if not isinstance(value, Mapping):
        failures.append(f"invalid_mapping:{path.name}")
        return {}
    return value


def _plan_rows(root: Path, failures: list[str]) -> list[dict[str, str]]:
    path = root / "PILOT_PLAN.csv"
    if not path.is_file():
        failures.append("missing:PILOT_PLAN.csv")
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "coordinate_key",
        "dataset",
        "target_protocol",
        "provider_seed",
        "method",
        "pilot_specification_version",
        "scenario_id",
        "scenario_fingerprint",
        "effective_execution_config_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        failures.append("plan_schema_invalid")
    return rows


def _verify_output_checksums(root: Path, failures: list[str]) -> None:
    path = root / "OUTPUT_CHECKSUMS.sha256"
    if not path.is_file():
        return
    declared: dict[str, str] = {}
    headers: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# ") and "=" in raw:
            key, value = raw[2:].split("=", 1)
            headers[key] = value
        elif raw.strip():
            digest, relative = raw.split("  ", 1)
            if relative in declared:
                failures.append(f"duplicate_output_checksum:{relative}")
            declared[relative] = digest
    manifest = _json(root / "RUN_MANIFEST.json", failures)
    if headers.get("effective_execution_config_sha256") != manifest.get(
        "effective_execution_config_sha256"
    ):
        failures.append("checksum_manifest_effective_hash_mismatch")
    expected = {
        str(item.relative_to(root)): sha256_path(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item != path
    }
    if set(declared) != set(expected):
        failures.append("output_checksum_path_set_mismatch")
    for relative in sorted(set(declared) & set(expected)):
        if declared[relative] != expected[relative]:
            failures.append(f"output_checksum_mismatch:{relative}")


def validate_package_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an output root by exact coordinate set, identities, and bytes."""

    root = root.resolve()
    failures: list[str] = []
    rows = _plan_rows(root, failures)
    manifest = _json(root / "RUN_MANIFEST.json", failures)
    expected_keys = [row.get("coordinate_key", "") for row in rows]
    if len(expected_keys) != len(set(expected_keys)):
        failures.append("duplicate_coordinate_in_plan")
    expected_set = set(expected_keys)
    manifest_keys = [str(value) for value in manifest.get("coordinate_keys", ())]
    if len(manifest_keys) != len(set(manifest_keys)):
        failures.append("duplicate_coordinate_in_run_manifest")
    if set(manifest_keys) != expected_set:
        failures.append("run_manifest_coordinate_set_mismatch")
    if int(manifest.get("coordinate_count", -1)) != len(rows):
        failures.append("run_manifest_coordinate_count_mismatch")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        failures.append("run_manifest_schema_invalid")
    global_identity = {
        "code_sha": str(manifest.get("repository_sha", "")),
        "base_config_sha256": str(manifest.get("base_config_sha256", "")),
        "effective_execution_config_sha256": str(
            manifest.get("effective_execution_config_sha256", "")
        ),
        "preregistration_sha256": str(manifest.get("preregistration_sha256", "")),
        "dependency_lock_sha256": str(manifest.get("dependency_lock_sha256", "")),
        "output_schema_version": str(manifest.get("output_schema_version", "")),
        "metric_schema_version": str(manifest.get("metric_schema_version", "")),
        "method_registry_version": str(manifest.get("method_registry_version", "")),
        "numerical_implementation_version": str(
            manifest.get("numerical_implementation_version", "")
        ),
    }
    if global_identity["output_schema_version"] != OUTPUT_SCHEMA_VERSION:
        failures.append("run_manifest_output_schema_invalid")
    if global_identity["metric_schema_version"] != METRIC_SCHEMA_VERSION:
        failures.append("run_manifest_metric_schema_invalid")
    if global_identity["method_registry_version"] != METHOD_REGISTRY_VERSION:
        failures.append("run_manifest_method_registry_invalid")
    if (
        global_identity["numerical_implementation_version"]
        != NUMERICAL_IMPLEMENTATION_VERSION
    ):
        failures.append("run_manifest_numerical_implementation_invalid")
    effective_config = manifest.get("effective_execution_config", {})
    if not isinstance(effective_config, Mapping):
        failures.append("effective_execution_config_missing")
    else:
        unhashed = dict(effective_config)
        declared_hash = str(unhashed.pop("effective_execution_config_sha256", ""))
        if declared_hash != canonical_hash(unhashed):
            failures.append("effective_execution_config_payload_hash_mismatch")
        if declared_hash != global_identity["effective_execution_config_sha256"]:
            failures.append("effective_execution_config_manifest_hash_mismatch")

    expected_pairs = {(row.get("scenario_id", ""), row.get("method", "")) for row in rows}
    observed_pairs = {
        (path.parents[1].name, path.name)
        for path in root.glob("scenarios/*/methods/*")
        if path.is_dir()
    }
    if observed_pairs - expected_pairs:
        failures.append("extra_coordinate_directory")
    if expected_pairs - observed_pairs:
        failures.append("missing_coordinate_directory")
    partials = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.endswith((".tmp", ".partial", ".incomplete"))
            or path.name.startswith(".tmp")
        )
    ]
    if partials:
        failures.append("temporary_partial_files_present:" + ",".join(sorted(partials)))

    coordinate_manifest: list[dict[str, Any]] = []
    observed_keys: list[str] = []
    archive_hashes = {
        str(key): str(value)
        for key, value in dict(manifest.get("archive_hashes", {})).items()
    }
    for row in rows:
        scenario_id = row.get("scenario_id", "")
        method = row.get("method", "")
        coordinate_key = row.get("coordinate_key", "")
        method_root = root / "scenarios" / scenario_id / "methods" / method
        present = {item.name for item in method_root.iterdir()} if method_root.is_dir() else set()
        missing_files = REQUIRED_METHOD_FILES - present
        unexpected_files = present - REQUIRED_METHOD_FILES
        if missing_files:
            failures.append(f"missing_method_files:{coordinate_key}:{sorted(missing_files)}")
        if unexpected_files:
            failures.append(f"unexpected_method_files:{coordinate_key}:{sorted(unexpected_files)}")
        failure_root = method_root.parents[1] / "failures"
        if (failure_root / f"{method}.json").exists() or (
            failure_root / f"{method}.traceback.txt"
        ).exists():
            failures.append(f"unresolved_failure:{coordinate_key}")
        checkpoint = _json(method_root / "checkpoint.json", failures)
        evaluation = _json(method_root / "evaluation.json", failures)
        freeze = _json(method_root / "POLICY_FREEZE_MANIFEST.json", failures)
        observed_key = str(evaluation.get("coordinate_key", checkpoint.get("coordinate_key", "")))
        observed_keys.append(observed_key)
        if observed_key != coordinate_key:
            failures.append(f"coordinate_key_mismatch:{coordinate_key}")
        coordinate_payload = evaluation.get("coordinate", {})
        if not isinstance(coordinate_payload, Mapping):
            coordinate_payload = {}
            failures.append(f"coordinate_payload_invalid:{coordinate_key}")
        expected_fields: dict[str, Any] = {
            "dataset": row.get("dataset"),
            "target_protocol": row.get("target_protocol"),
            "provider_seed": int(row.get("provider_seed", "-1")),
            "method": method,
            "pilot_specification_version": row.get("pilot_specification_version"),
            "scenario_id": scenario_id,
            "scenario_fingerprint": row.get("scenario_fingerprint"),
            "effective_execution_config_sha256": row.get(
                "effective_execution_config_sha256"
            ),
        }
        if dict(coordinate_payload) != expected_fields:
            failures.append(f"coordinate_payload_mismatch:{coordinate_key}")
        try:
            coordinate = PilotCoordinate(**expected_fields)
        except (TypeError, ValueError) as exc:
            failures.append(f"coordinate_invalid:{coordinate_key}:{type(exc).__name__}")
            coordinate = None
        for field, expected_value in global_identity.items():
            if evaluation.get(field) != expected_value:
                failures.append(f"evaluation_{field}_mismatch:{coordinate_key}")
        if evaluation.get("schema") != METHOD_RESULT_SCHEMA:
            failures.append(f"evaluation_schema_invalid:{coordinate_key}")
        metrics = evaluation.get("metrics", {})
        if not isinstance(metrics, Mapping) or metrics.get(
            "metric_schema_version"
        ) != METRIC_SCHEMA_VERSION:
            failures.append(f"metric_schema_invalid:{coordinate_key}")
        if isinstance(metrics, Mapping) and (
            "contract_regret" in metrics or "auprc" in metrics
        ):
            failures.append(f"superseded_metric_present:{coordinate_key}")
        if isinstance(metrics, Mapping):
            if metrics.get("scientific_compute_dtype") != SCIENTIFIC_COMPUTE_DTYPE:
                failures.append(f"scientific_dtype_invalid:{coordinate_key}")
            if int(metrics.get("rows_with_unavailable_nonzero_weight", -1)) != 0:
                failures.append(f"unavailable_nonzero_weight:{coordinate_key}")
            if int(metrics.get("rows_with_raw_regret_below_tolerance", -1)) != 0:
                failures.append(f"regret_below_tolerance:{coordinate_key}")
        if checkpoint.get("stage") != "COMPLETE":
            failures.append(f"checkpoint_not_terminal:{coordinate_key}")
        if checkpoint.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
            failures.append(f"checkpoint_output_schema_mismatch:{coordinate_key}")
        if checkpoint.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            failures.append(f"checkpoint_metric_schema_mismatch:{coordinate_key}")
        if checkpoint.get("effective_execution_config_sha256") != global_identity[
            "effective_execution_config_sha256"
        ]:
            failures.append(f"checkpoint_effective_hash_mismatch:{coordinate_key}")
        identity_hash = ""
        if coordinate is not None:
            identity_hash = coordinate_identity_hash(
                coordinate,
                code_sha=global_identity["code_sha"],
                config_sha256=global_identity["base_config_sha256"],
                preregistration_sha256=global_identity["preregistration_sha256"],
                dependency_lock_sha256=global_identity["dependency_lock_sha256"],
                effective_execution_config_sha256=global_identity[
                    "effective_execution_config_sha256"
                ],
            )
            if checkpoint.get("identity_hash") != identity_hash or evaluation.get(
                "identity_hash"
            ) != identity_hash:
                failures.append(f"coordinate_identity_hash_mismatch:{coordinate_key}")
        checksums = checkpoint.get("checksums", {})
        if not isinstance(checksums, Mapping):
            checksums = {}
            failures.append(f"checkpoint_checksums_invalid:{coordinate_key}")
        if set(checksums) != REQUIRED_METHOD_FILES - {"checkpoint.json"}:
            failures.append(f"checkpoint_checksum_set_mismatch:{coordinate_key}")
        for relative, expected_digest in checksums.items():
            path = method_root / str(relative)
            if not path.is_file() or sha256_path(path) != expected_digest:
                failures.append(f"coordinate_checksum_mismatch:{coordinate_key}:{relative}")
        marker_checksums = {str(k): str(v) for k, v in checksums.items() if k != "COMPLETE"}
        expected_marker = canonical_hash(
            {
                "coordinate_key": coordinate_key,
                "identity_hash": identity_hash,
                "effective_execution_config_sha256": global_identity[
                    "effective_execution_config_sha256"
                ],
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "checksums": marker_checksums,
            }
        )
        complete_path = method_root / "COMPLETE"
        if not complete_path.is_file() or complete_path.read_text(encoding="utf-8").strip() != expected_marker:
            failures.append(f"complete_identity_mismatch:{coordinate_key}")
        for field in (
            "code_sha",
            "base_config_sha256",
            "effective_execution_config_sha256",
            "preregistration_sha256",
            "dependency_lock_sha256",
            "output_schema_version",
            "metric_schema_version",
            "method_registry_version",
        ):
            if freeze.get(field) != global_identity[field]:
                failures.append(f"freeze_{field}_mismatch:{coordinate_key}")
        if freeze.get("schema") != POLICY_FREEZE_SCHEMA:
            failures.append(f"freeze_schema_invalid:{coordinate_key}")
        if freeze.get("numerical_implementation_version") != NUMERICAL_IMPLEMENTATION_VERSION:
            failures.append(f"freeze_numerical_implementation_mismatch:{coordinate_key}")
        score_path = method_root / "target_scores.npz"
        if score_path.is_file():
            try:
                with np.load(score_path, allow_pickle=False) as arrays:
                    if arrays["scores"].dtype != np.dtype(np.float64):
                        failures.append(f"stored_score_dtype_invalid:{coordinate_key}")
                    if arrays["routing_weights"].dtype != np.dtype(np.float64):
                        failures.append(f"stored_weight_dtype_invalid:{coordinate_key}")
                    if arrays["expected_compute"].dtype != np.dtype(np.float64):
                        failures.append(f"stored_compute_dtype_invalid:{coordinate_key}")
            except (KeyError, OSError, ValueError):
                failures.append(f"stored_numerical_payload_invalid:{coordinate_key}")
        source_artifacts = freeze.get("source_artifacts", ())
        target_artifacts = freeze.get("target_artifacts", ())
        artifacts = [*source_artifacts, *target_artifacts] if isinstance(
            source_artifacts, list
        ) and isinstance(target_artifacts, list) else []
        if len(source_artifacts) != 6 or len(target_artifacts) != 3:
            failures.append(f"artifact_identity_count_mismatch:{coordinate_key}")
        members: set[tuple[str, str, str]] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                failures.append(f"artifact_identity_invalid:{coordinate_key}")
                continue
            archive_name = str(artifact.get("archive_name", ""))
            archive_sha = str(artifact.get("archive_sha256", ""))
            member = (
                archive_name,
                str(artifact.get("member_name", "")),
                str(artifact.get("member_sha256", "")),
            )
            members.add(member)
            if archive_hashes.get(archive_name) != archive_sha:
                failures.append(f"archive_identity_mismatch:{coordinate_key}")
        if len(members) != 9:
            failures.append(f"member_identity_set_mismatch:{coordinate_key}")
        if evaluation.get("policy_freeze_sha256") != (
            sha256_path(method_root / "POLICY_FREEZE_MANIFEST.json")
            if (method_root / "POLICY_FREEZE_MANIFEST.json").is_file()
            else ""
        ):
            failures.append(f"policy_freeze_checksum_mismatch:{coordinate_key}")
        if evaluation.get("target_score_sha256") != (
            sha256_path(method_root / "target_scores.npz")
            if (method_root / "target_scores.npz").is_file()
            else ""
        ):
            failures.append(f"target_score_checksum_mismatch:{coordinate_key}")
        if evaluation.get("route_summary_sha256") != (
            sha256_path(method_root / "route_summary.json")
            if (method_root / "route_summary.json").is_file()
            else ""
        ):
            failures.append(f"route_summary_checksum_mismatch:{coordinate_key}")
        coordinate_manifest.append(
            {
                "coordinate_key": coordinate_key,
                "dataset": row.get("dataset"),
                "target_protocol": row.get("target_protocol"),
                "provider_seed": row.get("provider_seed"),
                "method": method,
                "scenario_id": scenario_id,
                "scenario_fingerprint": row.get("scenario_fingerprint"),
                **global_identity,
                "identity_hash": identity_hash,
                "policy_freeze_sha256": evaluation.get("policy_freeze_sha256", ""),
                "target_score_sha256": evaluation.get("target_score_sha256", ""),
                "route_summary_sha256": evaluation.get("route_summary_sha256", ""),
                "evaluation_sha256": (
                    sha256_path(method_root / "evaluation.json")
                    if (method_root / "evaluation.json").is_file()
                    else ""
                ),
                "complete_sha256": (
                    sha256_path(complete_path) if complete_path.is_file() else ""
                ),
            }
        )
    if len(observed_keys) != len(set(observed_keys)):
        failures.append("duplicate_observed_coordinate_key")
    if set(observed_keys) != expected_set:
        failures.append("observed_coordinate_set_mismatch")
    gate_path = root / "gates" / "PILOT_GATE_RESULT.json"
    if gate_path.is_file():
        gate = _json(gate_path, failures)
        if gate.get("schema") != GATE_SCHEMA:
            failures.append("gate_schema_mismatch")
        if gate.get("effective_execution_config_sha256") != global_identity[
            "effective_execution_config_sha256"
        ]:
            failures.append("gate_effective_hash_mismatch")
        if gate.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            failures.append("gate_metric_schema_mismatch")
    _verify_output_checksums(root, failures)
    report = {
        "schema": "coregraph_v5_2_package_validation_report_v2",
        "status": "PASS" if not failures else "FAIL",
        "expected_coordinate_count": len(expected_set),
        "observed_coordinate_count": len(set(observed_keys)),
        "expected_coordinate_set_sha256": canonical_hash(sorted(expected_set)),
        "observed_coordinate_set_sha256": canonical_hash(sorted(set(observed_keys))),
        "effective_execution_config_sha256": global_identity[
            "effective_execution_config_sha256"
        ],
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "failures": sorted(set(failures)),
    }
    if failures:
        raise PackageValidationError(tuple(sorted(set(failures))))
    return report, coordinate_manifest


def write_package_validation_artifacts(
    root: Path,
    report: Mapping[str, Any],
    coordinate_manifest: list[dict[str, Any]],
) -> None:
    if not coordinate_manifest:
        raise ValueError("refusing to write an empty package coordinate manifest")
    atomic_write_csv(root / "PACKAGE_COORDINATE_MANIFEST.csv", coordinate_manifest)
    atomic_write_json(root / "PACKAGE_VALIDATION_REPORT.json", dict(report))
