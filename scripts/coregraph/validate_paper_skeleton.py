#!/usr/bin/env python3
"""Audit Level-4 paper structure, anonymity, claims, TODOs, and fake results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_iclr"
BUILD = ROOT / "results" / "coregraph_build"
REQUIRED_SECTIONS = tuple(
    f"sections/{index:02d}_{name}.tex"
    for index, name in (
        (1, "introduction"),
        (2, "related_work"),
        (3, "problem_formulation"),
        (4, "method"),
        (5, "theory"),
        (6, "benchmark"),
        (7, "experimental_protocol"),
        (8, "results"),
        (9, "failure_analysis"),
        (10, "resource_tradeoffs"),
        (11, "limitations_ethics"),
        (12, "conclusion"),
    )
)
REQUIRED_SUPPLEMENT = tuple(
    f"supplement/{name}.tex"
    for name in (
        "proofs",
        "implementation_details",
        "benchmark_details",
        "statistical_plan",
        "additional_results",
        "reproducibility",
        "checklist",
    )
)


def main() -> int:
    required = (
        "main.tex",
        "supplement.tex",
        "abstract.tex",
        "references.bib",
        "claims/LEVEL4_ICLR_CLAIM_LEDGER.csv",
        *REQUIRED_SECTIONS,
        *REQUIRED_SUPPLEMENT,
    )
    failures = [f"missing:{relative}" for relative in required if not (PAPER / relative).is_file()]
    sources = [
        path
        for path in sorted(PAPER.rglob("*"))
        if path.is_file() and path.suffix in {".tex", ".md", ".csv", ".bib"}
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in sources)
    if text.count("\\author{Anonymous Authors}") < 2:
        failures.append("anonymous_author_missing_main_or_supplement")
    if "\\ClaimBlocked{" not in text or "\\ResultPending{" not in text:
        failures.append("typed_result_gates_missing")
    if re.search(r"Saket\s+Maganti|saketmaganti|/Users/|/Volumes/", text, flags=re.I):
        failures.append("author_identity_or_private_path_present")
    if re.search(r"\b(?:achieves?|outperforms?|improves? by|reduces? by)\s+\d", text, flags=re.I):
        failures.append("unsupported_numeric_result_language")
    ledger_path = PAPER / "claims" / "LEVEL4_ICLR_CLAIM_LEDGER.csv"
    claims = list(csv.DictReader(ledger_path.open(encoding="utf-8"))) if ledger_path.is_file() else []
    empirical = [row for row in claims if row.get("claim_type") == "empirical"]
    unblocked = [row["claim_id"] for row in empirical if not row["current_status"].startswith("BLOCKED")]
    if unblocked:
        failures.append(f"empirical_claims_unblocked:{unblocked}")
    todo_rows = []
    for path in sources:
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for token in re.findall(r"(?:TODO|BLOCKED|PENDING|RECHECK)[A-Z0-9_\-]*", line):
                todo_rows.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "line": line_number,
                        "token": token,
                        "status": "OPEN_RESULTS_BLOCKED" if "PENDING" in token or "BLOCKED" in token else "OPEN_SCHOLARLY_REVIEW",
                    }
                )
    with (BUILD / "LEVEL4_PAPER_TODO_LEDGER.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "line", "token", "status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(todo_rows)
    report = {
        "schema": "coregraph_level4_paper_claim_audit_v1",
        "status": "PASS_RESULTS_BLOCKED" if not failures else "FAIL",
        "main_sections": len(REQUIRED_SECTIONS),
        "supplement_sections": len(REQUIRED_SUPPLEMENT),
        "claims": len(claims),
        "blocked_empirical_claims": len(empirical),
        "open_todo_tokens": len(todo_rows),
        "invented_numeric_results": False if not any("numeric_result" in item for item in failures) else "DETECTED",
        "target_year_template": "OFFICIAL_ICLR_2027_TEMPLATE_NOT_AVAILABLE_AT_BUILD_TIME",
        "failures": failures,
    }
    (BUILD / "LEVEL4_PAPER_CLAIM_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Legacy report name remains for existing CI consumers.
    (BUILD / "PAPER_SKELETON_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
