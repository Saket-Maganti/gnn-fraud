"""Atomic experiment lifecycle with deterministic smoke execution."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from coregraph.contracts.serialization import to_primitive
from coregraph.experiments.config import RunConfig
from coregraph.experiments.hashing import config_hash, sha256_file, stable_run_id
from coregraph.experiments.manifests import RunManifest, RunStatus
from coregraph.experiments.resume import ResumeDecision, audit_resume
from coregraph.experiments.telemetry import Telemetry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(to_primitive(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ExperimentRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.run_id = stable_run_id(config)
        self.output_dir = Path(config.output.output_root) / self.run_id
        self.manifest_path = self.output_dir / "manifest.json"
        self.result_path = self.output_dir / "result.json"

    def plan(self) -> RunManifest:
        manifest = RunManifest(
            run_id=self.run_id,
            config_hash=config_hash(self.config),
            status=RunStatus.PLANNED,
            output_schema=self.config.output.schema,
            code_commit=self.config.code_commit,
            dataset_manifest_hash=self.config.dataset_manifest,
        )
        self._write_manifest(manifest)
        _atomic_json(self.output_dir / "config.json", self.config.to_dict())
        return manifest

    def run(
        self,
        execute: Callable[[RunConfig], Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> RunManifest:
        if self.manifest_path.exists() and not force:
            audit = audit_resume(self.config, self.manifest_path)
            if audit.decision is ResumeDecision.SKIP_COMPLETE:
                return RunManifest.from_dict(
                    json.loads(self.manifest_path.read_text(encoding="utf-8"))
                )
        started = _now()
        running = RunManifest(
            run_id=self.run_id,
            config_hash=config_hash(self.config),
            status=RunStatus.RUNNING,
            output_schema=self.config.output.schema,
            code_commit=self.config.code_commit,
            dataset_manifest_hash=self.config.dataset_manifest,
            started_at=started,
        )
        self._write_manifest(running)
        if self.config.dry_run:
            result = {
                "schema": self.config.output.schema,
                "run_id": self.run_id,
                "status": RunStatus.PLANNED.value,
                "execution": "DRY_RUN_ONLY",
                "config": self.config.to_dict(),
            }
            _atomic_json(self.result_path, result)
            planned = replace(
                running,
                status=RunStatus.PLANNED,
                completed_at=_now(),
                result_checksum=sha256_file(self.result_path),
            )
            self._write_manifest(planned)
            return planned
        try:
            with Telemetry() as telemetry:
                payload = dict(execute(self.config))
            result = {
                "schema": self.config.output.schema,
                "run_id": self.run_id,
                "status": (
                    RunStatus.SMOKE_PASS.value
                    if self.config.smoke
                    else RunStatus.COMPLETE.value
                ),
                "payload": payload,
                "telemetry": telemetry.to_dict(),
            }
            _atomic_json(self.result_path, result)
            prediction_path = ""
            prediction_checksum = ""
            if self.config.output.prediction_export:
                declared_path = payload.get("prediction_path")
                declared_checksum = payload.get("prediction_checksum")
                if not declared_path or not declared_checksum:
                    raise ValueError(
                        "prediction_export=True requires execute() to return "
                        "prediction_path and prediction_checksum"
                    )
                prediction = Path(str(declared_path))
                if not prediction.is_absolute():
                    prediction = self.output_dir / prediction
                if not prediction.is_file():
                    raise ValueError(f"declared prediction file is missing: {prediction}")
                actual_checksum = sha256_file(prediction)
                if actual_checksum != str(declared_checksum):
                    raise ValueError("declared prediction checksum mismatch")
                prediction_path = str(prediction)
                prediction_checksum = actual_checksum
            completed = replace(
                running,
                status=RunStatus.SMOKE_PASS if self.config.smoke else RunStatus.COMPLETE,
                completed_at=_now(),
                prediction_path=prediction_path,
                prediction_checksum=prediction_checksum,
                result_checksum=sha256_file(self.result_path),
            )
            self._write_manifest(completed)
            return completed
        except MemoryError as exc:
            blocked = replace(
                running,
                status=RunStatus.RESOURCE_BLOCKED,
                completed_at=_now(),
                errors=(str(exc),),
            )
            self._write_manifest(blocked)
            return blocked
        except Exception as exc:
            failed = replace(
                running,
                status=RunStatus.FAILED,
                completed_at=_now(),
                errors=(f"{type(exc).__name__}:{exc}",),
            )
            self._write_manifest(failed)
            raise

    def _write_manifest(self, manifest: RunManifest) -> None:
        _atomic_json(self.manifest_path, asdict(manifest))


def deterministic_early_stopping(
    validation_values: list[float],
    *,
    patience_checks: int,
    minimum_delta: float = 0.0,
    maximise: bool = True,
) -> int:
    """Return the number of checks consumed before stopping."""

    if patience_checks < 1:
        raise ValueError("patience_checks must be positive")
    best = -float("inf") if maximise else float("inf")
    stale = 0
    for index, value in enumerate(validation_values, start=1):
        improved = (
            value > best + minimum_delta
            if maximise
            else value < best - minimum_delta
        )
        if improved:
            best, stale = value, 0
        else:
            stale += 1
        if stale >= patience_checks:
            return index
    return len(validation_values)
