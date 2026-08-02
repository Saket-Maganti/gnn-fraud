#!/usr/bin/env python3
"""Authoritative V5 saved-output pilot planner, validator, and guarded executor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evidence.archive_store import ArchiveStore  # noqa: E402
from coregraph.experiments.v5_pilot_executor import (  # noqa: E402
    assemble_source_environments,
    assemble_target_unlabeled,
    compute_gate,
    execute_coordinate,
)
from coregraph.experiments.v5_pilot_outputs import (  # noqa: E402
    AGGREGATE_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA,
    SCENARIO_MANIFEST_SCHEMA,
    atomic_write_csv,
    atomic_write_text,
    build_effective_execution_config,
    canonical_hash,
)
from coregraph.experiments.v5_package_validator import (  # noqa: E402
    validate_package_root,
    write_package_validation_artifacts,
)
from coregraph.experiments.v5_pilot_types import (  # noqa: E402
    METHOD_REGISTRY_VERSION,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METHODS,
    PilotCoordinate,
    V5ScenarioMaterialization,
)
from coregraph.experiments.v5_numerics import (  # noqa: E402
    NUMERICAL_IMPLEMENTATION_VERSION,
    SCIENTIFIC_COMPUTE_DTYPE,
)
from coregraph.experiments.v5_scenario_loader import (  # noqa: E402
    V5PilotConfig,
    build_pilot_coordinates,
    load_v5_config,
    load_v5_surface,
    stable_file_sha256,
    validate_archive_surface,
)
from coregraph.experiments.v5_synthetic import build_synthetic_fixture  # noqa: E402
from coregraph.utils.io import atomic_write_json, sha256_path  # noqa: E402


DEFAULT_CACHE = ROOT.parent / "gnn-fraud-local-evidence-cache"
DEFAULT_OUTPUT = ROOT / "results" / "coregraph_pilot" / "v5"


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def _git_state() -> tuple[str, bool, Mapping[str, Any]]:
    sha = _git("rev-parse", "HEAD")
    unstaged = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--binary"],
    )
    staged = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--cached", "--binary"],
    )
    submodules = subprocess.check_output(
        ["git", "-C", str(ROOT), "submodule", "status", "--recursive"],
    )
    untracked_raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    )
    untracked_paths = sorted(
        value.decode("utf-8", errors="surrogateescape")
        for value in untracked_raw.split(b"\0")
        if value
    )
    untracked_contents = bytearray()
    untracked_records: list[dict[str, Any]] = []
    for relative in untracked_paths:
        path = ROOT / relative
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif path.is_file():
            content = path.read_bytes()
        else:
            content = b""
        encoded_path = relative.encode("utf-8", errors="surrogateescape")
        untracked_contents.extend(len(encoded_path).to_bytes(8, "big"))
        untracked_contents.extend(encoded_path)
        untracked_contents.extend(len(content).to_bytes(8, "big"))
        untracked_contents.extend(content)
        untracked_records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    combined = b"\0".join((unstaged, staged, bytes(untracked_contents), submodules))
    diagnostics = {
        "tracked_unstaged_diff_bytes": len(unstaged),
        "tracked_unstaged_diff_sha256": hashlib.sha256(unstaged).hexdigest(),
        "staged_diff_bytes": len(staged),
        "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
        "untracked_paths": untracked_records,
        "untracked_content_bytes": sum(item["bytes"] for item in untracked_records),
        "untracked_contents_sha256": hashlib.sha256(bytes(untracked_contents)).hexdigest(),
        "submodule_state_bytes": len(submodules),
        "submodule_state_sha256": hashlib.sha256(submodules).hexdigest(),
        "dirty_state_sha256": hashlib.sha256(combined).hexdigest(),
    }
    dirty = bool(unstaged or staged or untracked_paths or submodules.strip())
    return sha, dirty, diagnostics


def _dependency_lock_hash() -> str:
    return stable_file_sha256(ROOT / "requirements-coregraph-lock.txt")


def _plan_rows(coordinates: Sequence[PilotCoordinate]) -> list[dict[str, Any]]:
    return [
        {
            "coordinate_key": item.key,
            "dataset": item.dataset,
            "target_protocol": item.target_protocol,
            "provider_seed": item.provider_seed,
            "method": item.method,
            "pilot_specification_version": item.pilot_specification_version,
            "scenario_id": item.scenario_id,
            "scenario_fingerprint": item.scenario_fingerprint,
            "effective_execution_config_sha256": (
                item.effective_execution_config_sha256
            ),
            "status": "PLANNED",
        }
        for item in coordinates
    ]


def _write_plan(
    output_root: Path,
    coordinates: Sequence[PilotCoordinate],
    *,
    code_sha: str,
    config: V5PilotConfig,
    evidence_cache: Path,
    dirty: bool,
    dirty_diagnostics: Mapping[str, Any],
    command: Sequence[str],
    execution_authorized: bool,
    effective_execution_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "PILOT_PLAN.csv"
    atomic_write_csv(plan_path, _plan_rows(coordinates))
    plan_hash = sha256_path(plan_path)
    atomic_write_text(output_root / "PILOT_PLAN.sha256", f"{plan_hash}  PILOT_PLAN.csv\n")
    archive_hashes = dict(config.payload["archive_hashes"])
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "repository_sha": code_sha,
        "dirty_tree": dirty,
        "dirty_state_diagnostics": dirty_diagnostics,
        "base_config_sha256": config.config_sha256,
        "preregistration_sha256": config.preregistration_sha256,
        "effective_execution_config": dict(effective_execution_config),
        "effective_execution_config_sha256": effective_execution_config[
            "effective_execution_config_sha256"
        ],
        "dependency_lock_sha256": _dependency_lock_hash(),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "scientific_compute_dtype": SCIENTIFIC_COMPUTE_DTYPE,
        "evidence_cache_manifest_sha256": (
            stable_file_sha256(evidence_cache / "manifests" / "EVIDENCE_CACHE_MANIFEST.csv")
            if (evidence_cache / "manifests" / "EVIDENCE_CACHE_MANIFEST.csv").is_file()
            else None
        ),
        "archive_hashes": archive_hashes,
        "scenario_fingerprints": {
            item.scenario_id: item.scenario_fingerprint
            for item in coordinates[:: len(PRIMARY_METHODS)]
        },
        "coordinate_keys": [item.key for item in coordinates],
        "coordinate_count": len(coordinates),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "max_workers": effective_execution_config["max_workers"],
            "execution_device": "cpu_first",
            "configured_chunk_rows": effective_execution_config[
                "configured_chunk_rows"
            ],
            "effective_chunk_rows": effective_execution_config[
                "effective_chunk_rows"
            ],
            "chunk_override_active": (
                effective_execution_config["configured_chunk_rows"]
                != effective_execution_config["effective_chunk_rows"]
            ),
        },
        "command": list(command),
        "output_root": str(output_root),
        "plan_sha256": plan_hash,
        "execution_authorized": execution_authorized,
        "real_pilot_executed_by_plan_stage": False,
        "target_labels_loaded_by_plan_stage": False,
    }
    atomic_write_json(output_root / "RUN_MANIFEST.json", manifest)
    return manifest


def _scenario_by_id(
    scenarios: Sequence[V5ScenarioMaterialization],
) -> dict[str, V5ScenarioMaterialization]:
    return {item.definition.scenario_id: item for item in scenarios}


def _resource_estimate(
    artifacts,
    config: V5PilotConfig,
    output_root: Path,
) -> Mapping[str, Any]:
    target_rows = sum(
        item.label_known_count
        for item in artifacts
        if item.protocol in config.payload["required_protocols"]
    ) // len(config.payload["required_protocols"])
    score_bytes = target_rows * len(PRIMARY_METHODS) * 8
    route_bytes = target_rows * len(PRIMARY_METHODS) * len(config.experts) * 8
    chunk_rows = int(config.payload["streaming"]["chunk_rows"])
    estimated_chunk_working_set = chunk_rows * len(config.experts) * (
        4 + 1 + 4 + 4 + 4
    )
    free = shutil.disk_usage(output_root.parent if output_root.parent.exists() else ROOT).free
    return {
        "schema": "coregraph_v5_2_operational_estimate_v2",
        "estimate_not_measurement": True,
        "source_rows_per_split_per_environment": int(
            config.payload["streaming"]["source_rows_per_split_per_environment"]
        ),
        "estimated_raw_target_score_bytes": score_bytes,
        "estimated_raw_route_weight_bytes": route_bytes,
        "target_inference_chunk_rows": chunk_rows,
        "estimated_peak_numeric_chunk_working_set_bytes": estimated_chunk_working_set,
        "working_set_estimate_excludes_python_and_model_overhead": True,
        "compression_ratio_assumed": None,
        "real_fit_runtime": "UNKNOWN_UNTIL_AUTHORISED_RUN",
        "available_output_filesystem_bytes": free,
        "safest_workers": 1,
        "moderate_workers": 2,
        "high_memory_workers": "UNVALIDATED_NOT_RECOMMENDED",
    }


def _validate_real_output_root(
    output_root: Path,
    *,
    resume: bool,
    effective_execution_config: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise RuntimeError("real V5 output root exists and is not a directory")
    try:
        relative = output_root.relative_to(ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        ignored = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", "--", str(relative)],
            check=False,
        ).returncode == 0
        if not ignored:
            raise RuntimeError("real V5 output root inside Git must be ignored")
    existing = list(output_root.iterdir()) if output_root.is_dir() else []
    if existing:
        if not resume:
            raise RuntimeError("real V5 execution refuses a non-empty output root without --resume")
        manifest_path = output_root / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise RuntimeError("resume root is non-empty but has no RUN_MANIFEST.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("effective_execution_config_sha256") != effective_execution_config.get(
            "effective_execution_config_sha256"
        ):
            raise RuntimeError("resume root has an incompatible effective execution config")
        previous = manifest.get("effective_execution_config", {})
        if not isinstance(previous, Mapping) or previous.get("execution_mode") != "real":
            raise RuntimeError("synthetic or malformed outputs cannot enter a real output root")
    required = max(
        2 * 1024**3,
        3
        * (
            int(estimate["estimated_raw_target_score_bytes"])
            + int(estimate["estimated_raw_route_weight_bytes"])
        ),
    )
    available = int(estimate["available_output_filesystem_bytes"])
    if available < required:
        raise RuntimeError(
            f"insufficient output disk: available={available}, required={required}"
        )


def _write_output_checksums(output_root: Path) -> None:
    manifest = json.loads((output_root / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    lines = [
        "# schema=coregraph_v5_2_output_checksums_v3",
        "# effective_execution_config_sha256="
        + str(manifest["effective_execution_config_sha256"]),
        "# output_schema_version=" + str(manifest["output_schema_version"]),
        "# metric_schema_version=" + str(manifest["metric_schema_version"]),
    ]
    lines.extend(
        f"{sha256_path(path)}  {path.relative_to(output_root)}"
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "OUTPUT_CHECKSUMS.sha256"
    )
    atomic_write_text(output_root / "OUTPUT_CHECKSUMS.sha256", "\n".join(lines) + "\n")


def _write_zip(output_root: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))
    os.replace(temporary, destination)


def _validate_extracted_zip(destination: Path, output_root_name: str) -> Mapping[str, Any]:
    with zipfile.ZipFile(destination) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC validation failed for {corrupt}")
        with tempfile.TemporaryDirectory(prefix="coregraph-v5-package-") as temporary:
            extraction = Path(temporary)
            archive.extractall(extraction)
            report, _ = validate_package_root(extraction / output_root_name)
    return report


def _package(output_root: Path) -> Path:
    for generated in (
        output_root / "PACKAGE_VALIDATION_REPORT.json",
        output_root / "PACKAGE_COORDINATE_MANIFEST.csv",
        output_root / "OUTPUT_CHECKSUMS.sha256",
    ):
        if generated.is_file():
            generated.unlink()
    pre_report, coordinate_manifest = validate_package_root(output_root)
    report = {
        **pre_report,
        "pre_zip_validation": "PASS",
        "post_extraction_validation": "PENDING",
        "zip_crc_validation": "PENDING",
    }
    write_package_validation_artifacts(output_root, report, coordinate_manifest)
    _write_output_checksums(output_root)
    destination = output_root.parent / f"{output_root.name}.zip"
    _write_zip(output_root, destination)
    _validate_extracted_zip(destination, output_root.name)
    report = {
        **report,
        "post_extraction_validation": "PASS",
        "zip_crc_validation": "PASS",
    }
    write_package_validation_artifacts(output_root, report, coordinate_manifest)
    _write_output_checksums(output_root)
    _write_zip(output_root, destination)
    _validate_extracted_zip(destination, output_root.name)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/coregraph/pilot/saved_output_v5.yaml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--package", action="store_true")
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-rows", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--authorization-token")
    arguments = parser.parse_args()
    if not any((arguments.plan, arguments.validate_only, arguments.execute, arguments.package)):
        parser.error("choose --plan, --validate-only, --execute, or --package")
    if arguments.synthetic_fixture and not arguments.execute:
        parser.error("--synthetic-fixture is valid only with --execute")
    if arguments.max_workers < 1:
        parser.error("--max-workers must be positive")
    if arguments.chunk_rows is not None and arguments.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    return arguments


def main() -> int:
    arguments = parse_args()
    output_root = arguments.output_root.expanduser().resolve()
    if arguments.package:
        destination = _package(output_root)
        print(json.dumps({"status": "PACKAGED_COMPLETE_RUN", "path": str(destination)}, indent=2))
        return 0
    if arguments.max_workers != 1:
        raise RuntimeError(
            "strict deterministic V5 execution currently permits --max-workers 1 only"
        )
    config = load_v5_config(
        (ROOT / arguments.config).resolve()
        if not Path(arguments.config).is_absolute()
        else Path(arguments.config)
    )
    code_sha, dirty, dirty_diagnostics = _git_state()
    dependency_lock_sha256 = _dependency_lock_hash()
    evidence_cache = arguments.evidence_cache.expanduser().resolve()
    if arguments.synthetic_fixture:
        synthetic_config, evidence_cache = build_synthetic_fixture(
            output_root / "synthetic_fixture", config
        )
        config = load_v5_config(synthetic_config)
    configured_chunk_rows = int(config.payload["streaming"]["chunk_rows"])
    effective_chunk_rows = arguments.chunk_rows or configured_chunk_rows
    if effective_chunk_rows != configured_chunk_rows:
        payload = dict(config.payload)
        payload["streaming"] = {
            **dict(config.payload["streaming"]),
            "chunk_rows": effective_chunk_rows,
        }
        config = replace(config, payload=payload)
    effective_execution_config = build_effective_execution_config(
        base_config_sha256=config.config_sha256,
        preregistration_sha256=config.preregistration_sha256,
        configured_chunk_rows=configured_chunk_rows,
        effective_chunk_rows=effective_chunk_rows,
        max_workers=arguments.max_workers,
        execution_mode="synthetic" if arguments.synthetic_fixture else "real",
        synthetic_fixture=arguments.synthetic_fixture,
        dependency_lock_sha256=dependency_lock_sha256,
        code_sha=code_sha,
        numeric_dtype=str(config.payload["numerics"]["scientific_compute_dtype"]),
        stored_score_dtype=str(config.payload["numerics"]["stored_score_dtype"]),
        stored_weight_dtype=str(config.payload["numerics"]["stored_weight_dtype"]),
        numerical_implementation_version=str(
            config.payload["numerics"]["implementation_version"]
        ),
        weight_negative_tolerance=float(
            config.payload["numerics"]["weight_negative_tolerance"]
        ),
        simplex_tolerance=float(config.payload["numerics"]["simplex_tolerance"]),
        hull_projection_tolerance=float(
            config.payload["numerics"]["hull_projection_tolerance"]
        ),
    )
    effective_execution_config_sha256 = str(
        effective_execution_config["effective_execution_config_sha256"]
    )
    authorized = bool(
        arguments.execute
        and not arguments.synthetic_fixture
        and arguments.authorization_token == config.payload["authorization"]["real_execute_token"]
    )
    if arguments.execute and not arguments.synthetic_fixture:
        if not authorized:
            raise RuntimeError(
                "real V5 execution requires the explicit later authorization token"
            )
        if dirty:
            raise RuntimeError("real V5 execution refuses a dirty Git tree")
    artifacts, scenarios = load_v5_surface(
        config, code_sha=code_sha, evidence_cache=evidence_cache
    )
    coordinates = build_pilot_coordinates(
        scenarios,
        config,
        effective_execution_config_sha256=effective_execution_config_sha256,
    )
    if not arguments.synthetic_fixture and len(coordinates) != 240:
        raise RuntimeError(f"canonical V5 plan must contain exactly 240 coordinates, got {len(coordinates)}")
    estimate = _resource_estimate(artifacts, config, output_root)
    if arguments.execute and not arguments.synthetic_fixture:
        _validate_real_output_root(
            output_root,
            resume=arguments.resume,
            effective_execution_config=effective_execution_config,
            estimate=estimate,
        )
    run_manifest = _write_plan(
        output_root,
        coordinates,
        code_sha=code_sha,
        config=config,
        evidence_cache=evidence_cache,
        dirty=dirty,
        dirty_diagnostics=dirty_diagnostics,
        command=sys.argv,
        execution_authorized=authorized,
        effective_execution_config=effective_execution_config,
    )
    atomic_write_json(output_root / "OPERATIONAL_ESTIMATE.json", estimate)
    if arguments.plan:
        payload = {
            "status": "PLANNED_NO_TRAINING",
            "archives": 6,
            "base_artifacts": len(artifacts),
            "scenarios": len(scenarios),
            "bindings": sum(len(item.source_bindings) + len(item.target_bindings) for item in scenarios),
            "coordinates": len(coordinates),
            "training_performed": False,
            "target_labels_loaded": False,
            "run_manifest": run_manifest,
            "operational_estimate": estimate,
        }
        atomic_write_json(output_root / "PLAN_REPORT.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    validation = validate_archive_surface(
        artifacts,
        evidence_cache=evidence_cache,
        verify_members=True,
    )
    validation = {
        **validation,
        "scenario_count": len(scenarios),
        "binding_count": sum(
            len(item.source_bindings) + len(item.target_bindings) for item in scenarios
        ),
        "coordinate_count": len(coordinates),
        "scenario_materialization_status": "PASS_6_SOURCE_3_TARGET_EACH",
        "member_index_alignment_status": "PASS_60_PROTOCOL_SEED_GROUPS",
        "status": "VALIDATED_NO_TRAINING",
    }
    atomic_write_json(output_root / "VALIDATION_REPORT.json", validation)
    if arguments.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    by_scenario = _scenario_by_id(scenarios)
    coordinate_lookup = {(item.scenario_id, item.method): item for item in coordinates}
    store = ArchiveStore(evidence_cache, dict(config.payload["archive_hashes"]))
    results: list[Mapping[str, Any]] = []
    failures = 0
    for scenario in scenarios:
        scenario_root = output_root / "scenarios" / scenario.definition.scenario_id
        source = assemble_source_environments(store, scenario, config)
        target = assemble_target_unlabeled(store, scenario, config)
        atomic_write_json(
            scenario_root / "scenario_manifest.json",
            {
                "schema": SCENARIO_MANIFEST_SCHEMA,
                "definition": asdict(scenario.definition),
                "scenario_fingerprint": scenario.scenario_fingerprint,
                "code_sha": code_sha,
                "base_config_sha256": config.config_sha256,
                "effective_execution_config_sha256": (
                    effective_execution_config_sha256
                ),
                "preregistration_sha256": config.preregistration_sha256,
                "dependency_lock_sha256": dependency_lock_sha256,
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "method_registry_version": METHOD_REGISTRY_VERSION,
                "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
                "scientific_compute_dtype": SCIENTIFIC_COMPUTE_DTYPE,
                "source_binding_count": len(scenario.source_bindings),
                "target_binding_count": len(scenario.target_bindings),
                "target_unlabelled": target.to_serializable(),
            },
        )
        atomic_write_json(
            scenario_root / "source_assembly_report.json",
            {
                "schema": "coregraph_v5_source_assembly_report_v1",
                "environments": [
                    {
                        "environment_id": item.environment_id,
                        "protocol": item.protocol,
                        "train_rows": int((item.splits == "train").sum()),
                        "validation_rows": int((item.splits == "validation").sum()),
                        "row_key_sha256": canonical_hash(item.row_keys),
                    }
                    for item in source
                ],
                "target_labels_loaded": False,
            },
        )
        for method in PRIMARY_METHODS:
            coordinate = coordinate_lookup[(scenario.definition.scenario_id, method)]
            try:
                results.append(
                    execute_coordinate(
                        coordinate=coordinate,
                        scenario=by_scenario[coordinate.scenario_id],
                        source=source,
                        target=target,
                        store=store,
                        config=config,
                        output_root=output_root,
                        code_sha=code_sha,
                        dependency_lock_sha256=dependency_lock_sha256,
                        effective_execution_config_sha256=(
                            effective_execution_config_sha256
                        ),
                        resume=arguments.resume,
                    )
                )
            except Exception:
                failures += 1
                if arguments.fail_fast:
                    raise
    all_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.glob("scenarios/*/methods/*/evaluation.json"))
    ]
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "code_sha": code_sha,
        "base_config_sha256": config.config_sha256,
        "effective_execution_config_sha256": effective_execution_config_sha256,
        "preregistration_sha256": config.preregistration_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "scientific_compute_dtype": SCIENTIFIC_COMPUTE_DTYPE,
        "result_count": len(all_results),
        "failure_count": failures,
        "results": all_results,
    }
    aggregate_root = output_root / "aggregates"
    atomic_write_json(aggregate_root / "pilot_results.json", aggregate)
    gate = compute_gate(all_results, coordinates=coordinates, config=config)
    atomic_write_json(output_root / "gates" / "PILOT_GATE_RESULT.json", gate)
    payload = {
        "status": (
            "SYNTHETIC_EXECUTION_COMPLETE"
            if arguments.synthetic_fixture and not failures and len(all_results) == len(coordinates)
            else "EXECUTION_COMPLETE"
            if not failures
            else "EXECUTION_COMPLETED_WITH_FAILURES"
        ),
        "completed_coordinates": len(all_results),
        "required_coordinates": len(coordinates),
        "failures": failures,
        "gate_outcome": gate["outcome"],
        "real_pilot": not arguments.synthetic_fixture,
        "effective_execution_config_sha256": effective_execution_config_sha256,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    completed_keys = {str(item.get("coordinate_key", "")) for item in all_results}
    required_keys = {item.key for item in coordinates}
    return 0 if not failures and completed_keys == required_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
