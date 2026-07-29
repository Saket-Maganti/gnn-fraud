"""Claim gates for FraudShiftBench evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fraudshiftbench.evidence import EvidenceUnit, validate_evidence_unit


@dataclass
class ClaimGate:
    claim_id: str
    claim_text: str
    required_datasets: list[str] = field(default_factory=list)
    required_variants: list[str] = field(default_factory=list)
    required_protocols: list[str] = field(default_factory=list)
    required_models: list[str] = field(default_factory=list)
    required_seed_count: int = 0
    required_prediction_exports: bool = False
    prohibited_artifact_statuses: list[str] = field(default_factory=lambda: ["PENDING", "RESULT_REQUIRED", "PLACEHOLDER"])
    allowed_scope: str = ""
    status: str = "PENDING"
    evidence_paths: list[str] = field(default_factory=list)
    blocker_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimGate":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ClaimGate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_claim_gate(gate: ClaimGate, evidence_units: list[EvidenceUnit]) -> ClaimGate:
    blockers: list[str] = []
    candidates = evidence_units
    if gate.required_datasets:
        candidates = [unit for unit in candidates if unit.dataset in gate.required_datasets]
    if gate.required_variants:
        candidates = [unit for unit in candidates if unit.variant in gate.required_variants]
    if gate.required_protocols:
        candidates = [unit for unit in candidates if unit.protocol in gate.required_protocols or unit.protocol == "aggregate"]
    if gate.required_models:
        candidates = [unit for unit in candidates if unit.model in gate.required_models or unit.model == "aggregate"]
    if not candidates:
        blockers.append("no_matching_evidence_units")
    for unit in candidates:
        blockers.extend(validate_evidence_unit(unit, require_prediction=gate.required_prediction_exports))
        if unit.validation_status in gate.prohibited_artifact_statuses:
            blockers.append(f"prohibited_status:{unit.validation_status}")
    seeds = sorted({seed for unit in candidates for seed in (unit.seed_coverage or ([unit.seed] if unit.seed is not None else []))})
    if gate.required_seed_count and len(seeds) < gate.required_seed_count:
        blockers.append(f"insufficient_seed_count:{len(seeds)}<{gate.required_seed_count}")
    for variant in gate.required_variants:
        if not any(unit.variant == variant for unit in candidates):
            blockers.append(f"missing_variant:{variant}")
    out = ClaimGate.from_dict(gate.to_dict())
    out.evidence_paths = sorted({unit.result_json_path for unit in candidates if unit.result_json_path})
    out.blocker_reasons = sorted(set(blockers))
    out.status = "PASS" if not out.blocker_reasons else "BLOCKED"
    return out
