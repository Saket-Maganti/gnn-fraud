#!/usr/bin/env python3
"""Double-blind anonymization checker for the paper + submission package.

Scans text sources for identity leaks that would break double-blind review —
author names, emails, usernames, absolute home paths, GitHub URLs, institution
strings — and (best-effort) PDF metadata. **Report only**: it never edits or
deletes anything, so it is safe to run any time.

Findings are split into ``paper_source`` (files that ship to reviewers — the
LaTeX sources, bib, paper outlines) versus ``repo`` (developer-side files where
a local path is expected and harmless for a non-blind artifact). The
``double_blind_ok`` flag keys off high-severity findings in *paper sources*
only, which is what ``scripts/score_top_tier_readiness.py`` consumes.

Usage
-----
    python scripts/check_anonymization.py --dry-run        # default; report only
    python scripts/check_anonymization.py --scan-root runs_paper
    python scripts/check_anonymization.py --fail-on-paper-leak   # CI gate

Outputs:
    top_tier_upgrade/ANONYMIZATION_REPORT.md
    results/top_tier_checks/anonymization_report.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runs_post_kaggle_utils import utc_now, write_json, write_text  # noqa: E402

DEFAULT_JSON = Path("results/top_tier_checks/anonymization_report.json")
DEFAULT_MD = Path("top_tier_upgrade/ANONYMIZATION_REPORT.md")

# Identity tokens to flag. Extend via --extra-name. Case-insensitive match.
DEFAULT_NAME_TOKENS = ["Saket", "Maganti", "saketmaganti", "Saket-Maganti"]

# Files/dirs that ship to reviewers — leaks here are double-blind-critical.
PAPER_SOURCE_PREFIXES = ("runs_paper/", "paper/", "gnnpaper/", "top_tier_upgrade/", "paper_templates/")
PAPER_SOURCE_SUFFIXES = (".tex", ".bib", ".cls", ".sty")

# Heavy / vendor / generated dirs we never scan.
EXCLUDE_DIRS = {
    ".git", ".claude", "worktrees", "gnn_env", "venv", ".venv", "env",
    "__pycache__", "node_modules", "data", "results", "artifacts", "newresults",
    "logs", ".mypy_cache", ".pytest_cache", "figures", "kaggle",
}

# Emails that are *intended* anonymization placeholders, not identity leaks.
ANON_EMAIL_LOCAL = {"anonymous", "anon", "noreply", "no-reply", "author", "first.last", "firstname.lastname"}
ANON_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "anonymous.org", "anon.org"}


def _is_anon_email(addr: str) -> bool:
    local, _, domain = addr.partition("@")
    return local.lower() in ANON_EMAIL_LOCAL or domain.lower() in ANON_EMAIL_DOMAINS
# Only scan these text extensions.
TEXT_SUFFIXES = {".tex", ".bib", ".md", ".py", ".txt", ".cfg", ".toml", ".sh",
                 ".yaml", ".yml", ".cls", ".sty", ".ini"}
MAX_FILE_BYTES = 2_000_000
MAX_FINDINGS_PER_FILE = 40

# Files whose *content* is patterns-about-patterns; scanning them is noise.
SELF_SKIP_BASENAMES = {
    "check_anonymization.py", "ANONYMIZATION_REPORT.md",
    "anonymization_report.json", "test_anonymization.py",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOME_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
GITHUB_RE = re.compile(r"github\.com[/:][A-Za-z0-9._/-]+", re.IGNORECASE)


def _is_paper_source(rel_path: str) -> bool:
    return rel_path.startswith(PAPER_SOURCE_PREFIXES) or rel_path.endswith(PAPER_SOURCE_SUFFIXES)


def _iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(REPO_ROOT).parts) if path.is_relative_to(REPO_ROOT) else set(path.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in SELF_SKIP_BASENAMES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _severity(kind: str, paper_source: bool) -> str:
    if kind in ("email",):
        return "high"
    if kind == "name":
        return "high" if paper_source else "medium"
    if kind == "abs_path":
        return "high" if paper_source else "low"
    if kind == "github":
        return "medium" if paper_source else "low"
    return "medium"


def scan_file(path: Path, name_tokens: Sequence[str]) -> List[Dict[str, Any]]:
    rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    paper = _is_paper_source(rel)
    name_res = [(tok, re.compile(re.escape(tok), re.IGNORECASE)) for tok in name_tokens]
    findings: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for lineno, line in enumerate(text.splitlines(), start=1):
        checks = [
            ("email", EMAIL_RE.findall(line)),
            ("abs_path", HOME_PATH_RE.findall(line)),
            ("github", GITHUB_RE.findall(line)),
        ]
        for tok, rx in name_res:
            if rx.search(line):
                checks.append(("name", [tok]))
        for kind, hits in checks:
            for hit in hits:
                if kind == "email" and _is_anon_email(hit):
                    continue  # intended anonymization placeholder, not a leak
                findings.append({
                    "file": rel,
                    "line": lineno,
                    "kind": kind,
                    "match": hit[:120],
                    "paper_source": paper,
                    "severity": _severity(kind, paper),
                })
                if len(findings) >= MAX_FINDINGS_PER_FILE:
                    return findings
    return findings


def _pdf_metadata_findings(scan_root: Path, name_tokens: Sequence[str]) -> List[Dict[str, Any]]:
    """Best-effort scan of paper PDF metadata. Skipped if no PDF lib present.

    Only inspects PDFs under ``scan_root`` so a scoped scan (or a unit test on a
    temp directory) never reaches the repo's real paper PDF.
    """
    out: List[Dict[str, Any]] = []
    for pdf in (scan_root / "runs_paper" / "main.pdf", scan_root / "main.pdf"):
        if not pdf.exists():
            continue
        meta_str = None
        try:
            from pypdf import PdfReader  # type: ignore
            meta_str = str(PdfReader(str(pdf)).metadata)
        except Exception:  # noqa: BLE001
            try:
                from PyPDF2 import PdfReader  # type: ignore
                meta_str = str(PdfReader(str(pdf)).metadata)
            except Exception:  # noqa: BLE001
                meta_str = None
        rel = str(pdf.relative_to(REPO_ROOT)) if pdf.is_relative_to(REPO_ROOT) else str(pdf)
        if meta_str is None:
            out.append({"file": rel, "line": 0, "kind": "pdf_meta",
                        "match": "skipped (no pypdf/PyPDF2 installed)",
                        "paper_source": True, "severity": "info"})
            continue
        for tok in (*name_tokens, "@"):
            if tok.lower() in meta_str.lower():
                out.append({"file": rel, "line": 0, "kind": "pdf_meta",
                            "match": f"metadata contains '{tok}'",
                            "paper_source": True, "severity": "high"})
    return out


def build_report(scan_root: Path, name_tokens: Sequence[str]) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    for path in _iter_text_files(scan_root):
        findings.extend(scan_file(path, name_tokens))
    findings.extend(_pdf_metadata_findings(scan_root, name_tokens))

    paper_high = [f for f in findings if f["paper_source"] and f["severity"] == "high"]
    by_kind: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}
    for f in findings:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    return {
        "created_at_utc": utc_now(),
        "scan_root": str(scan_root.relative_to(REPO_ROOT)) if scan_root.is_relative_to(REPO_ROOT) else str(scan_root),
        "name_tokens": list(name_tokens),
        "total_findings": len(findings),
        "paper_source_high_findings": len(paper_high),
        "double_blind_ok": len(paper_high) == 0,
        "counts_by_kind": by_kind,
        "counts_by_severity": by_sev,
        "findings": findings,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Anonymization report",
        "",
        f"Generated: {report['created_at_utc']}",
        f"Scan root: `{report['scan_root']}`",
        f"Total findings: {report['total_findings']}",
        f"Paper-source HIGH-severity findings: {report['paper_source_high_findings']}",
        f"Double-blind OK: {'YES' if report['double_blind_ok'] else 'NO'}",
        "",
        "This tool is **report-only** — it never edits files. Findings in "
        "developer-side files (absolute paths, etc.) are expected and harmless "
        "for a non-blind artifact; only **paper-source** high-severity findings "
        "block double-blind readiness.",
        "",
        "## Counts by kind",
        "",
    ]
    for k, v in sorted(report["counts_by_kind"].items()):
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Counts by severity", ""])
    for k, v in sorted(report["counts_by_severity"].items()):
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Paper-source HIGH findings (must fix before double-blind)", ""])
    paper_high = [f for f in report["findings"] if f["paper_source"] and f["severity"] == "high"]
    if not paper_high:
        lines.append("- none")
    else:
        lines.append("| file | line | kind | match |")
        lines.append("| --- | ---: | --- | --- |")
        for f in paper_high[:200]:
            lines.append(f"| `{f['file']}` | {f['line']} | {f['kind']} | `{f['match']}` |")
    lines.extend(["", "## All findings (first 300)", "", "| file | line | kind | severity | paper_source | match |", "| --- | ---: | --- | --- | --- | --- |"])
    for f in report["findings"][:300]:
        lines.append(f"| `{f['file']}` | {f['line']} | {f['kind']} | {f['severity']} | {f['paper_source']} | `{f['match']}` |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Double-blind anonymization checker (report only).")
    p.add_argument("--scan-root", default=str(REPO_ROOT), help="Directory to scan (default: repo root).")
    p.add_argument("--json-output", default=str(DEFAULT_JSON))
    p.add_argument("--markdown-output", default=str(DEFAULT_MD))
    p.add_argument("--extra-name", action="append", default=[], help="Additional identity token to flag.")
    p.add_argument("--dry-run", action="store_true", help="No-op flag for safety symmetry; this tool never edits.")
    p.add_argument("--fail-on-paper-leak", action="store_true",
                   help="Exit non-zero if any paper-source high-severity finding exists.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scan_root = Path(args.scan_root).resolve()
    name_tokens = DEFAULT_NAME_TOKENS + list(args.extra_name)
    report = build_report(scan_root, name_tokens)
    write_json(args.json_output, report)
    write_text(args.markdown_output, render_markdown(report))
    print(
        f"[anonymization] findings={report['total_findings']} "
        f"paper_high={report['paper_source_high_findings']} "
        f"double_blind_ok={report['double_blind_ok']}"
    )
    if args.fail_on_paper_leak and not report["double_blind_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
