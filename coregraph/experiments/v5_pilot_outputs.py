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

from coregraph.experiments.v5_pilot_types import PilotCheckpoint, PilotCoordinate, PilotStage
from coregraph.utils.io import atomic_write_json, sha256_path


OUTPUT_SCHEMA_VERSION = "coregraph_v5_pilot_outputs_v1"


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
) -> str:
    return canonical_hash(
        {
            "coordinate": asdict(coordinate),
            "code_sha": code_sha,
            "config_sha256": config_sha256,
            "preregistration_sha256": preregistration_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
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
            checksums={str(key): str(value) for key, value in payload["checksums"].items()},
            retry_count=int(payload.get("retry_count", 0)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return PilotCheckpoint(
            coordinate_key="invalid",
            identity_hash="invalid",
            stage=PilotStage.FAILED,
            output_schema_version="invalid",
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
        "checksums": checksums,
    }
    atomic_write_text(method_root / "COMPLETE", canonical_hash(marker_payload) + "\n")
    checksums["COMPLETE"] = sha256_path(method_root / "COMPLETE")
    checkpoint = PilotCheckpoint(
        coordinate_key=coordinate.key,
        identity_hash=identity_hash,
        stage=PilotStage.COMPLETE,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
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
            "schema": "coregraph_v5_pilot_failure_v1",
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
            checksums={"failure": sha256_path(failure_path)},
            retry_count=retry_count,
        ),
    )
    return failure_path
