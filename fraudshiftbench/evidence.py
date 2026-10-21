"""Evidence-unit records and validators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceUnit:
    artifact_id: str
    dataset: str
    variant: str
    protocol: str
    model: str
    seed: int | None
    result_json_path: str
    prediction_path: str
    dry_run: bool
    diagnostic_only: bool
    resource_blocked: bool
    imported: bool
    validation_status: str
    sha256: str
    seed_coverage: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceUnit":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "EvidenceUnit":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_evidence_unit(unit: EvidenceUnit, *, require_prediction: bool = False) -> list[str]:
    reasons: list[str] = []
    if unit.dry_run:
        reasons.append("dry_run_evidence_rejected")
    if unit.diagnostic_only:
        reasons.append("diagnostic_only_evidence_rejected")
    if unit.resource_blocked:
        reasons.append("resource_blocked_evidence_rejected")
    if not unit.imported:
        reasons.append("not_imported")
    if unit.validation_status in {"", "PENDING", "PENDING_GPU_EXECUTION", "RESULT_REQUIRED", "PLACEHOLDER"}:
        reasons.append("pending_or_placeholder_status")
    if require_prediction and not unit.prediction_path:
        reasons.append("missing_prediction_export")
    return reasons


def evidence_units_from_v26_lock(lock_path: str | Path) -> list[EvidenceUnit]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    units: list[EvidenceUnit] = []
    for row in lock.get("v26_ibm_aml_variants", []):
        status = str(row.get("status", ""))
        resource_blocked = "RESOURCE" in status or str(row.get("variant", "")).endswith("large")
        units.append(
            EvidenceUnit(
                artifact_id=f"v26:{row.get('variant', '')}",
                dataset=str(row.get("dataset", "")),
                variant=str(row.get("variant", "")),
                protocol="aggregate",
                model="aggregate",
                seed=None,
                result_json_path="results/v26_imported/V26_IMPORTED_EVIDENCE_LOCK.json",
                prediction_path="results/v26_imported/V26_IMPORTED_EVIDENCE_LOCK.json" if int(row.get("actual_prediction_files", 0) or 0) > 0 else "",
                dry_run=False,
                diagnostic_only=False,
                resource_blocked=resource_blocked,
                imported=status in {"FULL10_PASS", "EXPLORATORY_PAIR_PASS"},
                validation_status=status,
                sha256="",
                seed_coverage=[int(seed) for seed in row.get("seed_coverage", [])],
            )
        )
    return units
