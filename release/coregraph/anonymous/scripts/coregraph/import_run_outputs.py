#!/usr/bin/env python3
"""Verify imported run bundles without merging or mutating their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMPORT_ROOT = ROOT / "results/coregraph_import"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-checksums", action="store_true", required=True)
    args = parser.parse_args()
    del args
    failures: list[str] = []
    manifests = sorted(IMPORT_ROOT.glob("**/manifest.json"))
    run_ids: dict[str, str] = {}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "run_id",
            "config_hash",
            "status",
            "output_schema",
            "code_commit",
            "dataset_manifest_hash",
        }
        missing = required - set(payload)
        if missing:
            failures.append(f"{path}:missing:{sorted(missing)}")
            continue
        run_id = payload["run_id"]
        signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if run_id in run_ids and run_ids[run_id] != signature:
            failures.append(f"duplicate_conflict:{run_id}")
        run_ids[run_id] = signature
        if payload["status"] not in {"COMPLETE", "SMOKE_PASS", "FAILED", "RESOURCE_BLOCKED"}:
            failures.append(f"{path}:nonterminal:{payload['status']}")
        result = path.parent / "result.json"
        if payload["status"] in {"COMPLETE", "SMOKE_PASS"}:
            if not result.exists() or sha(result) != payload.get("result_checksum"):
                failures.append(f"{path}:result_checksum")
            prediction_path = payload.get("prediction_path", "")
            if prediction_path:
                prediction = Path(prediction_path)
                if not prediction.is_absolute():
                    prediction = path.parent / prediction
                if not prediction.exists() or sha(prediction) != payload.get("prediction_checksum"):
                    failures.append(f"{path}:prediction_checksum")
    status = "PASS" if manifests and not failures else "BLOCKED_NO_IMPORTS" if not manifests else "FAIL"
    report = {
        "schema": "coregraph_import_audit_v1",
        "status": status,
        "manifests": len(manifests),
        "unique_runs": len(run_ids),
        "failures": failures,
    }
    output = ROOT / "results/coregraph_build/RESULT_IMPORT_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2 if status == "BLOCKED_NO_IMPORTS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
