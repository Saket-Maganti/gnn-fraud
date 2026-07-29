#!/usr/bin/env python3
"""Fail if an empirical claim is promoted without registered evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ledger = ROOT / "paper_iclr/claims/ICLR_CLAIM_LEDGER.csv"
    rows = list(csv.DictReader(ledger.open(encoding="utf-8")))
    evidence_root = ROOT / "results/coregraph_evidence"
    empirical_evidence_exists = any(evidence_root.glob("*.json")) if evidence_root.exists() else False
    failures: list[str] = []
    if not empirical_evidence_exists:
        for row in rows:
            claim_number = int(row["claim_id"].rsplit("C", 1)[1])
            allowed = row["status"].startswith("CLAIM_BLOCKED") or row[
                "status"
            ] == "RESOURCE_BLOCKED"
            if claim_number >= 4 and not allowed:
                failures.append(f"unbacked_empirical_claim:{row['claim_id']}")
    report = {
        "schema": "coregraph_claim_validation_v1",
        "status": "PASS" if not failures else "FAIL",
        "registered_claims": len(rows),
        "empirical_evidence_present": empirical_evidence_exists,
        "failures": failures,
    }
    output = ROOT / "results/coregraph_build/CLAIM_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
