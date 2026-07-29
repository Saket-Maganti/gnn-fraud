"""Run status and schema records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Tuple


class RunStatus(str, Enum):
    PLANNED = "PLANNED"
    SMOKE_PASS = "SMOKE_PASS"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    INVALID = "INVALID"
    EXCLUDED = "EXCLUDED"
    STALE_HASH = "STALE_HASH"


TERMINAL_STATUSES = {
    RunStatus.SMOKE_PASS,
    RunStatus.COMPLETE,
    RunStatus.FAILED,
    RunStatus.RESOURCE_BLOCKED,
    RunStatus.INVALID,
    RunStatus.EXCLUDED,
    RunStatus.STALE_HASH,
}


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    config_hash: str
    status: RunStatus
    output_schema: str
    code_commit: str
    dataset_manifest_hash: str
    prediction_path: str = ""
    prediction_checksum: str = ""
    result_checksum: str = ""
    started_at: str = ""
    completed_at: str = ""
    errors: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        return cls(
            run_id=str(payload["run_id"]),
            config_hash=str(payload["config_hash"]),
            status=RunStatus(payload["status"]),
            output_schema=str(payload["output_schema"]),
            code_commit=str(payload["code_commit"]),
            dataset_manifest_hash=str(payload["dataset_manifest_hash"]),
            prediction_path=str(payload.get("prediction_path", "")),
            prediction_checksum=str(payload.get("prediction_checksum", "")),
            result_checksum=str(payload.get("result_checksum", "")),
            started_at=str(payload.get("started_at", "")),
            completed_at=str(payload.get("completed_at", "")),
            errors=tuple(payload.get("errors", ())),
        )
