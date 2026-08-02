"""Hash- and schema-safe resume decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from coregraph.experiments.config import RunConfig
from coregraph.experiments.hashing import config_hash, sha256_file, stable_run_id
from coregraph.experiments.manifests import RunManifest, RunStatus


class ResumeDecision(str, Enum):
    SKIP_COMPLETE = "SKIP_COMPLETE"
    RUN_MISSING = "RUN_MISSING"
    RERUN_STALE_HASH = "RERUN_STALE_HASH"
    RERUN_INCOMPLETE = "RERUN_INCOMPLETE"
    RERUN_INVALID_OUTPUT = "RERUN_INVALID_OUTPUT"


@dataclass(frozen=True)
class ResumeAudit:
    decision: ResumeDecision
    reasons: tuple[str, ...]


def audit_resume(
    config: RunConfig,
    manifest_path: str | Path,
    *,
    dataset_manifest_hash: str | None = None,
) -> ResumeAudit:
    path = Path(manifest_path)
    if not path.exists():
        return ResumeAudit(ResumeDecision.RUN_MISSING, ("manifest_missing",))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = RunManifest.from_dict(payload)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        return ResumeAudit(
            ResumeDecision.RERUN_INVALID_OUTPUT,
            (f"manifest_invalid:{type(exc).__name__}",),
        )
    expected_config = config_hash(config)
    expected_run = stable_run_id(
        config,
        dataset_manifest_hash=dataset_manifest_hash,
    )
    stale = []
    if manifest.config_hash != expected_config:
        stale.append("config_hash_mismatch")
    if manifest.run_id != expected_run:
        stale.append("run_id_mismatch")
    if manifest.code_commit != config.code_commit:
        stale.append("code_commit_mismatch")
    if stale:
        return ResumeAudit(ResumeDecision.RERUN_STALE_HASH, tuple(stale))
    if manifest.output_schema != config.output.schema:
        return ResumeAudit(
            ResumeDecision.RERUN_INVALID_OUTPUT,
            ("output_schema_mismatch",),
        )
    if manifest.status not in {RunStatus.COMPLETE, RunStatus.SMOKE_PASS}:
        return ResumeAudit(
            ResumeDecision.RERUN_INCOMPLETE,
            (f"noncomplete_status:{manifest.status.value}",),
        )
    result_path = path.parent / "result.json"
    if not result_path.exists():
        return ResumeAudit(
            ResumeDecision.RERUN_INVALID_OUTPUT,
            ("result_missing",),
        )
    if sha256_file(result_path) != manifest.result_checksum:
        return ResumeAudit(
            ResumeDecision.RERUN_INVALID_OUTPUT,
            ("result_checksum_mismatch",),
        )
    if config.output.prediction_export:
        prediction = Path(manifest.prediction_path)
        if not prediction.exists():
            return ResumeAudit(
                ResumeDecision.RERUN_INVALID_OUTPUT,
                ("prediction_missing",),
            )
        if sha256_file(prediction) != manifest.prediction_checksum:
            return ResumeAudit(
                ResumeDecision.RERUN_INVALID_OUTPUT,
                ("prediction_checksum_mismatch",),
            )
    return ResumeAudit(ResumeDecision.SKIP_COMPLETE, ("hash_schema_checksum_valid",))
