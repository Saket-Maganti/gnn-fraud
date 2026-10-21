#!/usr/bin/env python3
"""
Static safety scan for README/docs — no training.

Flags risky default quickstarts and unsupported verified claims.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.audit_utils import utc_now, write_json, write_text  # noqa: E402


SCAN_FILES = (
    "README.md",
    "runs_paper/paper_body.tex",
    "runs_paper/main.tex",
    "runs_expansion/CLAIM_GATE_POLICY.md",
    "gnnpaper/claim_evidence_map.md",
)

HEAVY_SCRIPTS = ("run_full_upgrade.sh", "run_overnight.sh")

UNSUPPORTED_VERIFIED_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"(?i)\bverified\b.{0,80}\bdgraphfin\b", "Verified DGraphFin claim without artifact gate"),
    (r"(?i)\bverified\b.{0,80}\bt-?finance\b", "Verified T-Finance claim without artifact gate"),
    (r"(?i)\bverified\b.{0,80}\btpc\+?tta\b", "Verified TPC+TTA claim without artifact"),
    (r"(?i)\bverified\b.{0,80}\bgnn random", "Verified GNN random-vs-chronological without artifact"),
    (r"(?i)\bstate of the art\b|\bsota\b", "SOTA claim without artifact"),
)

BUSINESS_COST_PATTERN = re.compile(r"(?i)\bbusiness cost\b")
ALLOWED_BUSINESS_COST = re.compile(
    r"(?i)(business_cost\.|business_cost_|filename|provenance|artifact|results/business_cost)"
)


@dataclass
class SafetyIssue:
    level: str  # error | warning
    file: str
    line: int
    message: str


@dataclass
class SafetyReport:
    issues: List[SafetyIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)


def scan_file(path: str, report: SafetyReport) -> None:
    rel_path = os.path.relpath(path, REPO_ROOT)
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    in_quickstart = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip().lower()
        if "quick start" in stripped or stripped.startswith("## quick"):
            in_quickstart = True
        if stripped.startswith("## ") and "quick" not in stripped:
            in_quickstart = False

        if in_quickstart:
            for script in HEAVY_SCRIPTS:
                if script in line and "optional" not in line.lower() and "heavy" not in line.lower():
                    report.issues.append(
                        SafetyIssue(
                            "warning",
                            rel_path,
                            idx,
                            f"Quickstart mentions {script} without optional/heavy qualifier",
                        )
                    )

        for pattern, msg in UNSUPPORTED_VERIFIED_PATTERNS:
            if re.search(pattern, line) and "pending" not in line.lower() and "unsafe" not in line.lower():
                report.issues.append(SafetyIssue("error", rel_path, idx, msg))

        if BUSINESS_COST_PATTERN.search(line) and rel_path.startswith("runs"):
            if not ALLOWED_BUSINESS_COST.search(line):
                report.issues.append(
                    SafetyIssue(
                        "warning",
                        rel_path,
                        idx,
                        "business cost wording outside filename/provenance context",
                    )
                )


def render_markdown(report: SafetyReport) -> str:
    lines = [
        "# Safety check: no heavy defaults",
        "",
        f"Generated: {utc_now()}",
        f"Status: {'PASS' if report.ok else 'FAIL'}",
        "",
    ]
    for level in ("error", "warning"):
        items = [i for i in report.issues if i.level == level]
        lines.append(f"## {level.title()}s")
        lines.append("")
        if items:
            for issue in items:
                lines.append(f"- `{issue.file}:{issue.line}` — {issue.message}")
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety scan for heavy defaults / unsafe claims.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "results", "runs"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = SafetyReport()
    for rel in SCAN_FILES:
        path = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(path):
            scan_file(path, report)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(
        os.path.join(args.output_dir, "safety_check.json"),
        {
            "created_at_utc": utc_now(),
            "ok": report.ok,
            "issues": [issue.__dict__ for issue in report.issues],
        },
    )
    write_text(
        os.path.join(args.output_dir, "safety_check.md"),
        render_markdown(report),
    )
    print(f"[safety] ok={report.ok} issues={len(report.issues)}")
    for issue in report.issues:
        print(f"  [{issue.level}] {issue.file}:{issue.line} {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
