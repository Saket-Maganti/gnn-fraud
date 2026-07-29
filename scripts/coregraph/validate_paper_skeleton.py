#!/usr/bin/env python3
"""Validate anonymity, completeness, and result-placeholder integrity."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_iclr"
REQUIRED_SECTIONS = tuple(f"sections/{index:02d}_{name}.tex" for index, name in (
    (1, "abstract"),
    (2, "introduction"),
    (3, "related_work"),
    (4, "problem"),
    (5, "method"),
    (6, "theory"),
    (7, "experiments"),
    (8, "results_placeholder"),
    (9, "limitations"),
    (10, "conclusion"),
))


def main() -> int:
    failures: list[str] = []
    for relative in ("main.tex", "references.bib", "appendix/appendix.tex", *REQUIRED_SECTIONS):
        if not (PAPER / relative).is_file():
            failures.append(f"missing:{relative}")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PAPER.rglob("*"))
        if path.suffix in {".tex", ".md", ".csv"}
    )
    if "\\author{Anonymous Authors}" not in text:
        failures.append("anonymous_author_missing")
    if "RESULT_PENDING" not in text or "TABLE_PENDING_RUNS" not in text:
        failures.append("result_gates_missing")
    if re.search(r"\bSaket\s+Maganti\b", text, flags=re.IGNORECASE):
        failures.append("author_identity_present")
    ledger_path = PAPER / "claims/ICLR_CLAIM_LEDGER.csv"
    with ledger_path.open(encoding="utf-8") as handle:
        claims = list(csv.DictReader(handle))
    empirical = [row for row in claims if row["status"].startswith("CLAIM_BLOCKED")]
    if len(empirical) < 1:
        failures.append("empirical_claims_not_blocked")
    report = {
        "schema": "coregraph_paper_skeleton_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "sections": len(REQUIRED_SECTIONS),
        "claims": len(claims),
        "blocked_empirical_claims": len(empirical),
        "failures": failures,
        "target_year_template_status": "RECHECK_WHEN_PUBLISHED",
    }
    output = ROOT / "results/coregraph_build/PAPER_SKELETON_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
