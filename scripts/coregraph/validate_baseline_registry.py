#!/usr/bin/env python3
"""Validate official-baseline pins, task metadata, and licence gates."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PIN = re.compile(r"^[0-9a-f]{40}$")
STATUSES = {
    "OFFICIAL_CODE",
    "VALIDATED_REIMPLEMENTATION",
    "DIAGNOSTIC_APPROXIMATION",
    "PENDING_INTEGRATION",
    "UNAVAILABLE_LICENSE",
    "RESOURCE_BLOCKED",
}


def main() -> int:
    path = ROOT / "external_baselines/BASELINE_REGISTRY.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    blockers: list[str] = []
    for name, record in payload["baselines"].items():
        if not PIN.fullmatch(str(record.get("commit", ""))):
            failures.append(f"{name}:unpinned_commit")
        if record.get("status") not in STATUSES:
            failures.append(f"{name}:invalid_status")
        if not record.get("supported_tasks") or not record.get("entrypoint"):
            failures.append(f"{name}:missing_task_or_entrypoint")
        if record.get("status") == "UNAVAILABLE_LICENSE":
            if record.get("licence_file_verified"):
                failures.append(f"{name}:licence_state_contradiction")
            blockers.append(f"{name}:licence")
        elif not record.get("licence_file_verified"):
            failures.append(f"{name}:unverified_licence")
        if record.get("status") == "PENDING_INTEGRATION":
            blockers.append(f"{name}:parity")
    status = "INVALID" if failures else "VALID_WITH_BLOCKERS" if blockers else "PASS"
    report = {
        "schema": "coregraph_baseline_registry_audit_v1",
        "status": status,
        "failures": failures,
        "headline_blockers": blockers,
    }
    output = ROOT / "results/coregraph_build/BASELINE_REGISTRY_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
