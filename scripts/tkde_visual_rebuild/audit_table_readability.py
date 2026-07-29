#!/usr/bin/env python3
"""Audit active TKDE LaTeX tables for readability and evidence safety.

The audit follows the active ``\\input`` closure rather than scanning every
historical ``.tex`` file in ``paper_tkde``.  It is intentionally fail-closed
for missing dependencies, microscopic main-paper tables, raw-row dumps,
numeric resource-blocked rows, and unsafe table construction.  Findings are
written below the visual-rebuild result namespace so the frozen TKDE rebuild
reports remain untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = ROOT / "results" / "tkde_visual_rebuild" / "audits"

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
TABLE_ENV_RE = re.compile(
    r"\\begin\{(?P<env>table\*?|longtable|sidewaystable\*?)\}"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}",
    re.DOTALL,
)
CAPTION_RE = re.compile(r"\\caption(?:\[[^]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
TABULAR_BEGIN_RE = re.compile(r"\\begin\{(?P<env>tabular\*?|tabularx|longtable)\}")
ROW_END_RE = re.compile(r"(?<!\\)\\\\(?:\[[^]]*\])?")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
BLOCKED_RE = re.compile(
    r"(?i)(?:resource[- ]?blocked|guard[- ]?blocked|t4[- ]?oom|cuda[- ]?oom|"
    r"unmeasured|not measured|not run|missing evidence|safe_resource_blocked)"
)
METRIC_DECIMAL_RE = re.compile(r"(?<![\w.])(?:0?\.\d{2,}|1\.0{2,})(?!\w)")
RAW_DUMP_RE = re.compile(
    r"(?i)(?:complete|full|raw|exhaustive).{0,20}(?:seed|row|manifest|path|checksum|"
    r"provenance|statistical)|(?:seed|manifest|checksum|provenance).{0,20}(?:rows|dump|inventory)"
)
PROSE_WALL_RE = re.compile(r"(?:[^&\\]\s+){34,}[^&\\]")
ALLOWED_DISPOSITIONS = {
    "KEEP_AS_IS",
    "KEEP_WITH_MINOR_EDIT",
    "REDESIGN",
    "SPLIT",
    "MERGE",
    "REPLACE_WITH_DIFFERENT_VISUAL_FORM",
    "MOVE_MAIN_TO_SUPPLEMENT",
    "MOVE_SUPPLEMENT_TO_ARTIFACT",
    "REMOVE_REDUNDANT",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    line: int
    object_label: str
    message: str


@dataclass(frozen=True)
class TableRecord:
    path: str
    document: str
    environment: str
    label: str
    caption: str
    caption_words: int
    estimated_rows: int
    estimated_columns: int
    font_command: str
    uses_resizebox: bool
    uses_vertical_rules: bool


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def strip_comments(text: str) -> str:
    """Remove unescaped TeX comments without disturbing escaped percent signs."""

    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(offset, 0)) + 1


def resolve_tex_dependency(target: str, compile_dir: Path) -> Path:
    candidate = Path(target.strip())
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    return (compile_dir / candidate).resolve()


def active_tex_closure(entrypoint: Path, root: Path) -> tuple[list[Path], list[Finding]]:
    """Return the active TeX closure and missing/escaping dependency findings."""

    findings: list[Finding] = []
    if not entrypoint.is_file():
        return [], [
            Finding(
                "ERROR",
                "MISSING_ENTRYPOINT",
                _relative(entrypoint, root),
                1,
                "",
                "Required LaTeX entrypoint does not exist.",
            )
        ]
    compile_dir = entrypoint.parent.resolve()
    permitted = (root / "paper_tkde").resolve()
    pending = [entrypoint.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if path != permitted and permitted not in path.parents:
            findings.append(
                Finding(
                    "ERROR",
                    "DEPENDENCY_PATH_ESCAPE",
                    _relative(path, root),
                    1,
                    "",
                    "LaTeX dependency escapes paper_tkde.",
                )
            )
            continue
        if not path.is_file():
            findings.append(
                Finding(
                    "ERROR",
                    "MISSING_TEX_DEPENDENCY",
                    _relative(path, root),
                    1,
                    "",
                    "Active LaTeX dependency does not exist.",
                )
            )
            continue
        visited.add(path)
        raw = path.read_text(encoding="utf-8")
        text = strip_comments(raw)
        for match in INPUT_RE.finditer(text):
            child = resolve_tex_dependency(match.group(1), compile_dir)
            if child != permitted and permitted not in child.parents:
                findings.append(
                    Finding(
                        "ERROR",
                        "DEPENDENCY_PATH_ESCAPE",
                        _relative(path, root),
                        line_number(text, match.start()),
                        "",
                        f"Input target escapes paper_tkde: {match.group(1)!r}.",
                    )
                )
            elif not child.is_file():
                findings.append(
                    Finding(
                        "ERROR",
                        "MISSING_TEX_DEPENDENCY",
                        _relative(path, root),
                        line_number(text, match.start()),
                        "",
                        f"Input target is missing: {match.group(1)!r}.",
                    )
                )
            else:
                pending.append(child)
    return sorted(visited), findings


def _font_command(prefix: str, body: str) -> str:
    scope = f"{prefix[-800:]}\n{body[:800]}"
    matches = list(re.finditer(r"\\(tiny|scriptsize|footnotesize|small|normalsize)\b", scope))
    return matches[-1].group(1) if matches else "inherited"


def _balanced_group(text: str, start: int) -> tuple[str, int] | None:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _column_count(body: str) -> tuple[int, bool]:
    match = TABULAR_BEGIN_RE.search(body)
    if not match:
        return 0, False
    first = _balanced_group(body, match.end())
    if first is None:
        return 0, False
    # tabularx has a width argument before its column specification.
    group = _balanced_group(body, first[1]) if match.group("env") == "tabularx" else first
    if group is None:
        return 0, False
    spec = group[0]
    vertical = "|" in spec

    def count_tokens(value: str) -> int:
        count = 0
        index = 0
        while index < len(value):
            char = value[index]
            if char == "\\":
                index += 1
                while index < len(value) and (value[index].isalpha() or value[index] == "@"):
                    index += 1
                continue
            if char in "@!><":
                group_value = _balanced_group(value, index + 1)
                index = group_value[1] if group_value else index + 1
                continue
            if char in "pmb":
                group_value = _balanced_group(value, index + 1)
                if group_value:
                    count += 1
                    index = group_value[1]
                    continue
            if char in "lcrX":
                count += 1
                index += 1
                continue
            if char == "*":
                repeats = _balanced_group(value, index + 1)
                nested = _balanced_group(value, repeats[1]) if repeats else None
                if repeats and nested and repeats[0].strip().isdigit():
                    count += int(repeats[0].strip()) * count_tokens(nested[0])
                    index = nested[1]
                    continue
            index += 1
        return count

    return count_tokens(spec), vertical


def _caption_text(body: str) -> str:
    match = CAPTION_RE.search(body)
    if not match:
        return ""
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", match.group(1))
    value = re.sub(r"[{}~]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _row_count(body: str) -> int:
    # Header/footer repetition commands are structural, not scientific rows.
    scrubbed = re.sub(r"\\(?:endfirsthead|endhead|endfoot|endlastfoot)\b", "", body)
    return len(ROW_END_RE.findall(scrubbed))


def extract_tables(path: Path, document: str, root: Path) -> tuple[list[TableRecord], list[Finding]]:
    raw = path.read_text(encoding="utf-8")
    text = strip_comments(raw)
    records: list[TableRecord] = []
    findings: list[Finding] = []
    relative = _relative(path, root)
    for match in re.finditer(r"(?<!\\)(?P<number>\d+(?:\.\d+)?)%(?=\s|[A-Za-z])", raw):
        findings.append(
            Finding(
                "ERROR",
                "UNESCAPED_PERCENT",
                relative,
                line_number(raw, match.start()),
                "",
                f"Numeric percentage {match.group('number')}% must be escaped as \\% in TeX.",
            )
        )

    # A table source can contain only tabular/longtable content and be wrapped
    # by its caller.  Treat a standalone tabular as a record as well.
    matches = list(TABLE_ENV_RE.finditer(text))
    if not matches and re.search(r"\\begin\{(?:tabular\*?|tabularx)\}", text):
        matches = [re.match(r"(?P<body>.*)", text, re.DOTALL)]  # type: ignore[list-item]

    for index, match in enumerate(matches, start=1):
        assert match is not None
        body = match.groupdict().get("body") or match.group(0)
        env = match.groupdict().get("env") or "tabular-fragment"
        caption = _caption_text(body)
        label_match = LABEL_RE.search(body)
        label = label_match.group(1) if label_match else f"unlabeled:{relative}:{index}"
        rows = _row_count(body)
        columns, vertical = _column_count(match.group(0))
        font = _font_command(text[: match.start()], body)
        record = TableRecord(
            path=relative,
            document=document,
            environment=env,
            label=label,
            caption=caption,
            caption_words=len(caption.split()),
            estimated_rows=rows,
            estimated_columns=columns,
            font_command=font,
            uses_resizebox=bool(re.search(r"\\resizebox\b", body)),
            uses_vertical_rules=vertical,
        )
        records.append(record)
        line = line_number(text, match.start())

        def add(severity: str, code: str, message: str) -> None:
            findings.append(Finding(severity, code, relative, line, label, message))

        if env.startswith(("table", "sidewaystable")) and not caption:
            add("ERROR", "MISSING_CAPTION", "A floating table has no caption.")
        if env.startswith(("table", "sidewaystable")) and label.startswith("unlabeled:"):
            add("ERROR", "MISSING_LABEL", "A floating table has no LaTeX label.")
        if font == "tiny":
            add("ERROR", "TINY_TABLE", "Active human-readable tables may not use \\tiny.")
        if document == "main" and font == "scriptsize":
            add(
                "ERROR",
                "MICROSCOPIC_MAIN_TABLE",
                "Main-paper table uses \\scriptsize, below the V2 print-size target.",
            )
        elif document == "supplement" and font == "scriptsize":
            add(
                "WARNING",
                "SUPPLEMENT_SCRIPTSIZE",
                "Supplement table uses the minimum-size \\scriptsize; confirm actual 7 pt print readability.",
            )
        if record.uses_resizebox:
            add(
                "ERROR" if document == "main" else "WARNING",
                "RESIZEBOX_TABLE",
                "Substantive tables must fit through information design, not whole-table scaling.",
            )
        if record.uses_vertical_rules:
            add("ERROR", "VERTICAL_TABLE_RULE", "Publication tables may not use vertical rules.")
        if columns > (9 if document == "main" else 11):
            add(
                "ERROR",
                "EXCESSIVE_COLUMNS",
                f"Estimated {columns} columns exceed the readable {document} limit.",
            )
        caption_limit = 45 if document == "main" else 60
        if record.caption_words > caption_limit:
            add(
                "ERROR",
                "LONG_CAPTION",
                f"Caption has {record.caption_words} words; limit is {caption_limit} without an exception.",
            )
        if rows > 80 or (rows > 24 and RAW_DUMP_RE.search(f"{caption} {label}")):
            add(
                "ERROR",
                "RAW_ROW_DUMP",
                f"Estimated {rows} rows form a machine-readable dump; summarize and archive exhaustive rows.",
            )
        elif rows > 45:
            add(
                "WARNING",
                "LARGE_HUMAN_TABLE",
                f"Estimated {rows} rows require explicit readability justification.",
            )
        if re.search(r"\\begin\{(?:landscape|sideways)\}", body) or env.startswith("sidewaystable"):
            add(
                "WARNING",
                "LANDSCAPE_TABLE",
                "Landscape orientation requires page-utilization evidence from the PDF layout audit.",
            )
        if PROSE_WALL_RE.search(body):
            add(
                "WARNING",
                "PROSE_WALL_CELL",
                "A cell appears to contain sentence-scale prose; move explanation into surrounding text.",
            )
        data_body = body
        if "\\midrule" in data_body:
            data_body = data_body.split("\\midrule", 1)[1]
        if "\\bottomrule" in data_body:
            data_body = data_body.split("\\bottomrule", 1)[0]
        for row_no, row_text in enumerate(ROW_END_RE.split(data_body), start=1):
            if BLOCKED_RE.search(row_text) and METRIC_DECIMAL_RE.search(row_text):
                findings.append(
                    Finding(
                        "ERROR",
                        "BLOCKED_ROW_HAS_NUMERIC_PERFORMANCE",
                        relative,
                        line + row_no - 1,
                        label,
                        "A resource-blocked/unmeasured row contains a metric-like decimal.",
                    )
                )
    return records, findings


def audit(
    root: Path,
    entrypoints: Sequence[Path] | None = None,
) -> tuple[list[TableRecord], list[Finding], list[str]]:
    paper = root / "paper_tkde"
    selected = list(entrypoints or (paper / "main.tex", paper / "supplement" / "supplement.tex"))
    records: list[TableRecord] = []
    findings: list[Finding] = []
    active_paths: set[Path] = set()
    for entrypoint in selected:
        closure, closure_findings = active_tex_closure(entrypoint, root)
        findings.extend(closure_findings)
        active_paths.update(closure)
        document = "supplement" if "supplement" in entrypoint.parts else "main"
        for path in closure:
            table_records, table_findings = extract_tables(path, document, root)
            records.extend(table_records)
            findings.extend(table_findings)

    labels: dict[str, TableRecord] = {}
    for record in records:
        if record.label.startswith("unlabeled:"):
            continue
        if record.label in labels:
            findings.append(
                Finding(
                    "ERROR",
                    "DUPLICATE_TABLE_LABEL",
                    record.path,
                    1,
                    record.label,
                    f"Duplicate active label also appears in {labels[record.label].path}.",
                )
            )
        labels[record.label] = record

    findings.sort(key=lambda item: (item.severity != "ERROR", item.path, item.line, item.code))
    return records, findings, sorted(_relative(path, root) for path in active_paths)


def write_reports(
    report_dir: Path,
    records: Sequence[TableRecord],
    findings: Sequence[Finding],
    active_paths: Sequence[str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    payload = {
        "verdict": "PASS" if errors == 0 else "FAIL",
        "active_tex_files": list(active_paths),
        "table_count": len(records),
        "error_count": errors,
        "warning_count": warnings,
        "tables": [asdict(item) for item in records],
        "findings": [asdict(item) for item in findings],
    }
    (report_dir / "TABLE_READABILITY_AUDIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Table readability audit",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        f"- Active TeX files: {len(active_paths)}",
        f"- Active tables: {len(records)}",
        f"- Errors: {errors}",
        f"- Warnings: {warnings}",
        "",
        "## Active table summary",
        "",
        "| Document | Label | Environment | Rows | Columns | Font | Caption words |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for item in records:
        lines.append(
            f"| {item.document} | `{item.label}` | {item.environment} | "
            f"{item.estimated_rows} | {item.estimated_columns} | {item.font_command} | "
            f"{item.caption_words} |"
        )
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("No readability, dependency, or scientific-status violations were detected.")
    else:
        for item in findings:
            where = f"{item.path}:{item.line}"
            label = f" `{item.object_label}`" if item.object_label else ""
            lines.append(f"- **{item.severity} {item.code}** {where}{label}: {item.message}")
    (report_dir / "TABLE_READABILITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any ERROR finding.")
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Also exit nonzero when warnings remain.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    report_dir = (args.report_dir or root / "results" / "tkde_visual_rebuild" / "audits").resolve()
    records, findings, active = audit(root)
    write_reports(report_dir, records, findings, active)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    print(f"tables={len(records)} errors={errors} warnings={warnings}")
    if args.strict and (errors or (args.warnings_as_errors and warnings)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
