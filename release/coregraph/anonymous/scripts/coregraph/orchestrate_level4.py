#!/usr/bin/env python3
"""Create and validate Level-4 execution plans without running experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "results" / "coregraph_build"
MATRIX = BUILD / "LEVEL4_FULL_RUN_MATRIX.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _write(name: str, payload: dict[str, object]) -> None:
    (BUILD / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def pilot_plan(rows: list[dict[str, str]]) -> int:
    selected = [row for row in rows if row["priority"] == "PILOT_MUST_RUN"]
    payload = {
        "schema": "coregraph_level4_pilot_execution_plan_v1",
        "status": "PLAN_ONLY_NO_EXECUTION",
        "code_sha": _git_sha(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "coordinate_count": len(selected),
        "datasets": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "seeds": sorted({int(row["seed"]) for row in selected}),
        "run_ids": [row["run_id"] for row in selected],
        "target_labels_used_for_fitting": False,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
    }
    _write("LEVEL4_PILOT_EXECUTION_PLAN.json", payload)
    return 0


def pilot_validate() -> int:
    evidence = json.loads((BUILD / "ARCHIVE_MEMBER_VALIDATION.json").read_text())
    leakage = json.loads((BUILD / "V5_LEAKAGE_AUDIT.json").read_text())
    base = _read_csv(BUILD / "V5_BASE_ARTIFACTS.csv")
    scenarios = _read_csv(BUILD / "V5_SCENARIOS.csv")
    bindings = _read_csv(BUILD / "V5_BINDINGS.csv")
    failures: list[str] = []
    if evidence.get("verdict") != "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS":
        failures.append("evidence")
    if leakage.get("overall_status") != "PASS_NO_TRAINING_BYTE_AND_STRUCTURE":
        failures.append("leakage")
    if (len(base), len(scenarios), len(bindings)) != (180, 60, 540):
        failures.append("cardinality")
    payload = {
        "schema": "coregraph_level4_pilot_input_validation_v1",
        "status": "READY_FOR_SAVED_OUTPUT_PILOT" if not failures else "BLOCKED",
        "archives": evidence.get("archive_verified", 0),
        "members": evidence.get("member_checksum_verified", 0),
        "base_artifacts": len(base),
        "scenarios": len(scenarios),
        "bindings": len(bindings),
        "pilot_executed": False,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
        "failures": failures,
    }
    _write("LEVEL4_PILOT_INPUT_VALIDATION.json", payload)
    return 0 if not failures else 1


def full_plan(rows: list[dict[str, str]]) -> int:
    payload = {
        "schema": "coregraph_level4_full_execution_plan_v1",
        "status": "PLAN_ONLY_NO_EXECUTION",
        "code_sha": _git_sha(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "coordinate_count": len(rows),
        "priority_counts": dict(sorted(Counter(row["priority"] for row in rows).items())),
        "device_counts": dict(
            sorted(Counter(row["expected_device"] for row in rows).items())
        ),
        "runtime_values_observed": 0,
        "gpu_jobs_launched": 0,
    }
    _write("LEVEL4_FULL_EXECUTION_PLAN.json", payload)
    return 0


def analysis() -> int:
    imported = ROOT / "results" / "coregraph_level4" / "VALIDATED_RUN_IMPORT.json"
    available = imported.is_file()
    payload = {
        "schema": "coregraph_level4_analysis_orchestration_v1",
        "status": "READY" if available else "BLOCKED_VALIDATED_RUN_IMPORT_ABSENT",
        "validated_run_import": available,
        "analysis_executed": False,
        "target_metrics_computed_by_this_command": 0,
        "instruction": (
            "Run the authorised analysis prompt only after validated outputs are imported."
        ),
    }
    _write("LEVEL4_ANALYSIS_STATUS.json", payload)
    return 0 if available else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("pilot-plan", "pilot-validate", "full-plan", "analysis")
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(MATRIX)
    if arguments.command == "pilot-plan":
        return pilot_plan(rows)
    if arguments.command == "pilot-validate":
        return pilot_validate()
    if arguments.command == "full-plan":
        return full_plan(rows)
    return analysis()


if __name__ == "__main__":
    raise SystemExit(main())
