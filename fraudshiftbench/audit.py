"""Static claim-language audits."""

from __future__ import annotations

from pathlib import Path

BLOCKED_PHRASES = (
    "all graph methods fail",
    "all gnns fail",
    "graph structure is always harmful",
    "ibm aml large passed",
    "p100 evidence",
    "graphsafe universally dominates",
    "all ibm aml variants passed",
)


def audit_text(text: str) -> list[str]:
    hits: list[str] = []
    safe_context = (
        "blocked",
        "forbidden",
        "not claim",
        "does not claim",
        "do not",
        "without strict",
        "unless strict",
        "remains resource-boundary",
    )
    in_safe_section = False
    for line in text.splitlines():
        lower = line.lower()
        if lower.lstrip().startswith("#"):
            in_safe_section = "blocked" in lower or "forbidden" in lower or "limitations" in lower
            continue
        if in_safe_section:
            continue
        if any(marker in lower for marker in safe_context):
            continue
        for phrase in BLOCKED_PHRASES:
            if phrase in lower:
                hits.append(phrase)
    return sorted(set(hits))


def audit_paths(paths: list[str | Path]) -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for path_value in paths:
        path = Path(path_value)
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.suffix.lower() in {".md", ".tex", ".txt"}]
        else:
            files = [path]
        for file_path in files:
            if not file_path.is_file():
                continue
            hits = audit_text(file_path.read_text(encoding="utf-8", errors="replace"))
            if hits:
                findings[str(file_path)] = hits
    return findings
