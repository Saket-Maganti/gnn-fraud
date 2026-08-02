"""Atomic, checksum-addressed V5 pilot output and resume primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from coregraph.experiments.v5_pilot_types import (
    METHOD_REGISTRY_VERSION,
    METRIC_SCHEMA_VERSION,
    PilotCheckpoint,
    PilotCoordinate,
    PilotStage,
)
from coregraph.utils.io import atomic_write_json, sha256_path


OUTPUT_SCHEMA_VERSION = "coregraph_v5_pilot_outputs_v2"
EFFECTIVE_EXECUTION_CONFIG_SCHEMA = "coregraph_v5_effective_execution_config_v1"


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_effective_execution_config(
    *,
    base_config_sha256: str,
    preregistration_sha256: str,
    configured_chunk_rows: int,
    effective_chunk_rows: int,
    max_workers: int,
    execution_mode: str,
    synthetic_fixture: bool,
    dependency_lock_sha256: str,
    code_sha: str,
    output_schema_version: str = OUTPUT_SCHEMA_VERSION,
    metric_schema_version: str = METRIC_SCHEMA_VERSION,
    numeric_dtype: str = "float32",
    deterministic_algorithms: bool = True,
    archive_streaming_mode: str = "verified_zip_member_stream_no_extraction",
    source_sampling_policy: str = "stable_sha256_rank_per_source_split_environment",
    target_inference_policy: str = "label_blind_bounded_chunked_inference",
) -> dict[str, Any]:
    """Return the complete canonical execution configuration and its hash."""

    if configured_chunk_rows < 1 or effective_chunk_rows < 1:
        raise ValueError("configured and effective chunk rows must be positive")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if execution_mode not in {"real", "synthetic"}:
        raise ValueError("execution_mode must be real or synthetic")
    if synthetic_fixture != (execution_mode == "synthetic"):
        raise ValueError("synthetic_fixture and execution_mode disagree")
    payload: dict[str, Any] = {
        "schema": EFFECTIVE_EXECUTION_CONFIG_SCHEMA,
        "base_config_sha256": base_config_sha256,
        "preregistration_sha256": preregistration_sha256,
        "configured_chunk_rows": configured_chunk_rows,
        "effective_chunk_rows": effective_chunk_rows,
        "max_workers": max_workers,
        "execution_mode": execution_mode,
        "synthetic_fixture": synthetic_fixture,
        "numeric_dtype": numeric_dtype,
        "deterministic_algorithms": deterministic_algorithms,
        "output_schema_version": output_schema_version,
        "metric_schema_version": metric_schema_version,
        "method_registry_version": METHOD_REGISTRY_VERSION,
        "archive_streaming_mode": archive_streaming_mode,
        "source_sampling_policy": source_sampling_policy,
        "target_inference_policy": target_inference_policy,
        "dependency_lock_sha256": dependency_lock_sha256,
        "code_sha": code_sha,
    }
    payload["effective_execution_config_sha256"] = canonical_hash(payload)
    return payload


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def coordinate_identity_hash(
    coordinate: PilotCoordinate,
    *,
    code_sha: str,
    config_sha256: str,
    preregistration_sha256: str,
    dependency_lock_sha256: str,
    effective_execution_config_sha256: str,
    metric_schema_version: str = METRIC_SCHEMA_VERSION,
) -> str:
    return canonical_hash(
        {
            "coordinate": asdict(coordinate),
            "code_sha": code_sha,
            "config_sha256": config_sha256,
            "preregistration_sha256": preregistration_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
            "effective_execution_config_sha256": effective_execution_config_sha256,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "metric_schema_version": metric_schema_version,
            "method_registry_version": METHOD_REGISTRY_VERSION,
        }
    )


def checkpoint_path(method_root: Path) -> Path:
    return method_root / "checkpoint.json"


def write_checkpoint(method_root: Path, checkpoint: PilotCheckpoint) -> Path:
    path = checkpoint_path(method_root)
    atomic_write_json(
        path,
        {
            **asdict(checkpoint),
            "stage": checkpoint.stage.value,
        },
    )
    return path


def load_checkpoint(method_root: Path) -> PilotCheckpoint | None:
    path = checkpoint_path(method_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PilotCheckpoint(
            coordinate_key=str(payload["coordinate_key"]),
            identity_hash=str(payload["identity_hash"]),
            stage=PilotStage(payload["stage"]),
            output_schema_version=str(payload["output_schema_version"]),
            metric_schema_version=str(payload["metric_schema_version"]),
            effective_execution_config_sha256=str(
                payload["effective_execution_config_sha256"]
            ),
            checksums={str(key): str(value) for key, value in payload["checksums"].items()},
            retry_count=int(payload.get("retry_count", 0)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return PilotCheckpoint(
            coordinate_key="invalid",
            identity_hash="invalid",
            stage=PilotStage.FAILED,
            output_schema_version="invalid",
            metric_schema_version="invalid",
            effective_execution_config_sha256="invalid",
            checksums={},
            retry_count=0,
        )


def reusable_complete(
    method_root: Path,
    *,
    coordinate: PilotCoordinate,
    identity_hash: str,
) -> tuple[bool, tuple[str, ...]]:
    checkpoint = load_checkpoint(method_root)
    if checkpoint is None:
        return False, ("checkpoint_missing",)
    reasons: list[str] = []
    if checkpoint.coordinate_key != coordinate.key:
        reasons.append("coordinate_key_mismatch")
    if checkpoint.identity_hash != identity_hash:
        reasons.append("identity_hash_mismatch")
    if checkpoint.output_schema_version != OUTPUT_SCHEMA_VERSION:
        reasons.append("output_schema_mismatch")
    if checkpoint.metric_schema_version != METRIC_SCHEMA_VERSION:
        reasons.append("metric_schema_mismatch")
    if (
        checkpoint.effective_execution_config_sha256
        != coordinate.effective_execution_config_sha256
    ):
        reasons.append("effective_execution_config_mismatch")
    if checkpoint.stage is not PilotStage.COMPLETE:
        reasons.append(f"stage_{checkpoint.stage.value.lower()}")
    if not (method_root / "COMPLETE").is_file():
        reasons.append("complete_marker_missing")
    for relative, expected in checkpoint.checksums.items():
        path = method_root / relative
        if not path.is_file():
            reasons.append(f"output_missing:{relative}")
        elif sha256_path(path) != expected:
            reasons.append(f"output_checksum_mismatch:{relative}")
    return not reasons, tuple(reasons)


def mark_complete(
    method_root: Path,
    *,
    coordinate: PilotCoordinate,
    identity_hash: str,
    outputs: Sequence[str],
    retry_count: int,
) -> PilotCheckpoint:
    checksums = {name: sha256_path(method_root / name) for name in outputs}
    marker_payload = {
        "coordinate_key": coordinate.key,
        "identity_hash": identity_hash,
        "effective_execution_config_sha256": (
            coordinate.effective_execution_config_sha256
        ),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "checksums": checksums,
    }
    atomic_write_text(method_root / "COMPLETE", canonical_hash(marker_payload) + "\n")
    checksums["COMPLETE"] = sha256_path(method_root / "COMPLETE")
    checkpoint = PilotCheckpoint(
        coordinate_key=coordinate.key,
        identity_hash=identity_hash,
        stage=PilotStage.COMPLETE,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        effective_execution_config_sha256=(
            coordinate.effective_execution_config_sha256
        ),
        checksums=checksums,
        retry_count=retry_count,
    )
    write_checkpoint(method_root, checkpoint)
    return checkpoint


def write_failure(
    method_root: Path,
    *,
    coordinate: PilotCoordinate,
    identity_hash: str,
    stage: PilotStage,
    exception: BaseException,
    traceback_text: str,
    retry_count: int,
) -> Path:
    failure_root = method_root.parents[1] / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    trace_path = failure_root / f"{coordinate.method}.traceback.txt"
    atomic_write_text(trace_path, traceback_text)
    failure_path = failure_root / f"{coordinate.method}.json"
    atomic_write_json(
        failure_path,
        {
            "schema": "coregraph_v5_pilot_failure_v2",
            "coordinate_key": coordinate.key,
            "method": coordinate.method,
            "stage": stage.value,
            "exception_class": type(exception).__name__,
            "message": str(exception),
            "traceback_path": str(trace_path.name),
            "recoverability": "RETRY_WITH_IDENTICAL_INPUTS",
            "retry_count": retry_count,
            "resource_status": "UNKNOWN_NOT_INFERRED",
            "partial_outputs_removed": False,
        },
    )
    write_checkpoint(
        method_root,
        PilotCheckpoint(
            coordinate_key=coordinate.key,
            identity_hash=identity_hash,
            stage=PilotStage.FAILED,
            output_schema_version=OUTPUT_SCHEMA_VERSION,
            metric_schema_version=METRIC_SCHEMA_VERSION,
            effective_execution_config_sha256=(
                coordinate.effective_execution_config_sha256
            ),
            checksums={"failure": sha256_path(failure_path)},
            retry_count=retry_count,
        ),
    )
    return failure_path
