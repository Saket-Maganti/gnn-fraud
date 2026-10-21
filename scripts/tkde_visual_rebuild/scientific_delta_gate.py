#!/usr/bin/env python3
"""Fail-closed scientific-preservation gate for the TKDE visual delta pass.

The gate does not recompute settled science.  It verifies the frozen input and
analysis hashes, claim ontology, citation set, object allocation, generated
asset provenance, and blocked-cell semantics.  A passing report therefore
means that the publication-design pass stayed on the approved scientific
surface; it is not a new statistical or empirical analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("results/tkde_visual_rebuild")
EXPECTED_BASELINE_ARCHIVES = {
    "release/tkde_manuscript_package.zip": "234b164d0553fd4d19b1e850c19e4b0924bfeee1a0342201f326d24d2faa2ba0",
    "release/tkde_source_tables.zip": "528780b5666dd1b97f0a12be39e20957af4a08f25de8d7b334179727be4d919c",
}
# These two files contain presentation paths and rendered-asset hashes.  A
# visual rebuild must update them, so byte identity is neither expected nor a
# scientific invariant.  Their existence, source-data closure, and current
# asset hashes are validated by ``provenance_findings`` below.
PRESENTATION_MUTABLE_PATHS = {
    "results/tkde_rebuild/FIGURE_DATA_PROVENANCE.csv",
    "results/tkde_rebuild/MANUSCRIPT_MACHINE_AUDIT.csv",
    "results/tkde_rebuild/TABLE_DATA_PROVENANCE.csv",
}
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
ALLOWED_DESTINATIONS = {"main", "supplement", "artifact", "removed"}
BLOCKED_RE = re.compile(
    r"(?i)(?:resource[-_ ]?blocked|guard[-_ ]?blocked|t4[-_ ]?oom|cuda[-_ ]?oom|"
    r"safe_resource_blocked|unmeasured|not[-_ ]?measured|missing[-_ ]?evidence)"
)
METRIC_FIELD_RE = re.compile(
    r"(?i)(?:^|_)(?:auprc|auroc|f1|precision|recall|metric_value|performance|"
    r"mean_score|score_mean|test_metric)(?:$|_)"
)
NA_VALUES = {"", "na", "n/a", "nan", "none", "null", "--", "---", "unmeasured", "blocked"}
CITE_RE = re.compile(r"\\cite\w*\s*\{([^}]+)\}")
BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    message: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_csv(path: Path, root: Path, findings: list[Finding]) -> list[dict[str, str]]:
    if not path.is_file():
        findings.append(Finding("ERROR", "MISSING_REQUIRED_FILE", relative(path, root), "Required CSV is absent."))
        return []
    try:
        return read_csv(path)
    except (csv.Error, UnicodeDecodeError) as exc:
        findings.append(Finding("ERROR", "INVALID_CSV", relative(path, root), str(exc)))
        return []


def frozen_hash_findings(root: Path, ledger: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    rows = require_csv(ledger, root, findings)
    seen: set[str] = set()
    for row in rows:
        relpath = (row.get("path") or "").strip()
        expected = (row.get("sha256_before") or "").strip().lower()
        if not relpath or not re.fullmatch(r"[0-9a-f]{64}", expected):
            findings.append(Finding("ERROR", "INVALID_FREEZE_ROW", relative(ledger, root), f"Malformed freeze row: {relpath!r}."))
            continue
        if relpath in seen:
            findings.append(Finding("ERROR", "DUPLICATE_FREEZE_PATH", relative(ledger, root), relpath))
            continue
        seen.add(relpath)
        path = (root / relpath).resolve()
        if path != root and root not in path.parents:
            findings.append(Finding("ERROR", "FREEZE_PATH_ESCAPE", relpath, "Frozen path escapes repository root."))
        elif not path.is_file():
            findings.append(Finding("ERROR", "MISSING_FROZEN_INPUT", relpath, "Frozen scientific file is absent."))
        else:
            actual = sha256_file(path)
            if actual != expected and relpath not in PRESENTATION_MUTABLE_PATHS:
                findings.append(
                    Finding(
                        "ERROR",
                        "FROZEN_HASH_CHANGED",
                        relpath,
                        f"Expected {expected}, observed {actual}.",
                    )
                )
    if len(rows) < 36:
        findings.append(
            Finding("ERROR", "INCOMPLETE_FREEZE_LEDGER", relative(ledger, root), f"Expected at least 36 frozen rows; found {len(rows)}.")
        )
    return findings, len(rows)


def baseline_archive_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relpath, expected in EXPECTED_BASELINE_ARCHIVES.items():
        path = root / relpath
        if not path.is_file():
            findings.append(Finding("ERROR", "MISSING_BASELINE_ARCHIVE", relpath, "Frozen baseline ZIP is absent."))
            continue
        actual = sha256_file(path)
        if actual != expected:
            findings.append(
                Finding(
                    "ERROR",
                    "BASELINE_ARCHIVE_MUTATED",
                    relpath,
                    f"Expected {expected}, observed {actual}.",
                )
            )
    return findings


def claim_findings(root: Path, ledger: Path) -> tuple[list[Finding], dict[str, int]]:
    findings: list[Finding] = []
    rows = require_csv(ledger, root, findings)
    ids = [row.get("claim_id", "").strip() for row in rows]
    if len(rows) != 22 or len(set(ids)) != 22 or any(not value for value in ids):
        findings.append(
            Finding(
                "ERROR",
                "CLAIM_LEDGER_SHAPE_CHANGED",
                relative(ledger, root),
                f"Expected 22 unique typed claims; observed {len(rows)} rows and {len(set(ids))} unique IDs.",
            )
        )
    statuses: dict[str, int] = {}
    for row in rows:
        status = (row.get("support_status") or "").strip()
        statuses[status] = statuses.get(status, 0) + 1
        if not (row.get("scope") or "").strip():
            findings.append(Finding("ERROR", "EMPTY_CLAIM_SCOPE", relative(ledger, root), f"Claim {row.get('claim_id')} has no scope."))
        if not (row.get("permitted_wording") or "").strip() or not (row.get("prohibited_wording") or "").strip():
            findings.append(
                Finding("ERROR", "INCOMPLETE_CLAIM_QUANTIFIER_BOUNDARY", relative(ledger, root), f"Claim {row.get('claim_id')} lacks permitted/prohibited wording.")
            )
    return findings, statuses


def _active_tex(entrypoint: Path, paper_root: Path) -> set[Path]:
    compile_dir = entrypoint.parent.resolve()
    pending = [entrypoint.resolve()]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited or not path.is_file():
            continue
        if path != paper_root and paper_root not in path.parents:
            continue
        visited.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in INPUT_RE.findall(text):
            candidate = Path(target)
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            pending.append((compile_dir / candidate).resolve())
    return visited


def citation_findings(root: Path) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    bib = root / "paper_tkde" / "references.bib"
    if not bib.is_file():
        return [Finding("ERROR", "MISSING_BIBLIOGRAPHY", relative(bib, root), "Verified bibliography is absent.")], 0, 0
    keys = BIB_ENTRY_RE.findall(bib.read_text(encoding="utf-8", errors="replace"))
    if len(keys) != 50 or len(set(keys)) != 50:
        findings.append(
            Finding("ERROR", "BIBLIOGRAPHY_SET_CHANGED", relative(bib, root), f"Expected 50 unique verified entries; found {len(keys)} rows / {len(set(keys))} keys.")
        )
    active: set[Path] = set()
    paper_root = (root / "paper_tkde").resolve()
    for entrypoint in (paper_root / "main.tex", paper_root / "supplement" / "supplement.tex"):
        if not entrypoint.is_file():
            findings.append(Finding("ERROR", "MISSING_TEX_ENTRYPOINT", relative(entrypoint, root), "Citation closure cannot be checked."))
        active.update(_active_tex(entrypoint, paper_root))
    cited: set[str] = set()
    for path in active:
        text = path.read_text(encoding="utf-8", errors="replace")
        for group in CITE_RE.findall(text):
            cited.update(key.strip() for key in group.split(",") if key.strip())
    unknown = sorted(cited - set(keys))
    unused = sorted(set(keys) - cited)
    if unknown:
        findings.append(Finding("ERROR", "UNKNOWN_CITATION_KEYS", "paper_tkde", ", ".join(unknown)))
    if unused:
        findings.append(
            Finding(
                "ERROR",
                "VERIFIED_REFERENCE_REMOVED_WITHOUT_EXCEPTION",
                "paper_tkde",
                f"Verified references are no longer cited: {', '.join(unused)}.",
            )
        )
    return findings, len(keys), len(cited)


def allocation_findings(root: Path, inventory: Path, allocation: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    inventory_rows = require_csv(inventory, root, findings)
    allocation_rows = require_csv(allocation, root, findings)
    inventory_ids = [row.get("object_id", "").strip() for row in inventory_rows]
    allocation_ids = [row.get("object_id", "").strip() for row in allocation_rows]
    if len(inventory_ids) != len(set(inventory_ids)) or len(allocation_ids) != len(set(allocation_ids)):
        findings.append(Finding("ERROR", "DUPLICATE_OBJECT_ID", "results/tkde_visual_rebuild", "Inventory/allocation IDs are not unique."))
    if set(inventory_ids) != set(allocation_ids):
        missing = sorted(set(inventory_ids) - set(allocation_ids))
        extra = sorted(set(allocation_ids) - set(inventory_ids))
        findings.append(
            Finding("ERROR", "OBJECT_ALLOCATION_MISMATCH", relative(allocation, root), f"Missing={missing}; extra={extra}.")
        )
    for row in allocation_rows:
        object_id = row.get("object_id", "")
        disposition = (row.get("final_disposition") or "").strip()
        destination = (row.get("final_destination") or "").strip().lower()
        if disposition not in ALLOWED_DISPOSITIONS:
            findings.append(Finding("ERROR", "INVALID_DISPOSITION", relative(allocation, root), f"{object_id}: {disposition!r}."))
        if destination not in ALLOWED_DESTINATIONS:
            findings.append(Finding("ERROR", "INVALID_DESTINATION", relative(allocation, root), f"{object_id}: {destination!r}."))
        if not (row.get("planned_replacement") or "").strip():
            findings.append(Finding("ERROR", "UNEXPLAINED_OBJECT_ALLOCATION", relative(allocation, root), f"{object_id} has no replacement/allocation rationale."))
        for token in (row.get("source_data_files") or "").split(";"):
            value = token.strip()
            value = re.sub(r"^(?:code-authored|generated)\s*;?\s*", "", value)
            if not value or value.lower() in {"n/a", "none", "latex source"}:
                continue
            if value.startswith(("results/", "paper_tkde/", "manuscript_assets/", "kaggle_workspace/")):
                path = root / value
                if not path.is_file():
                    findings.append(Finding("ERROR", "MISSING_OBJECT_SOURCE_DATA", value, f"Referenced by {object_id}."))
    return findings, len(allocation_rows)


def provenance_findings(root: Path) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    visual = root / "results" / "tkde_visual_rebuild"
    figure_manifest = visual / "FIGURE_DATA_PROVENANCE.csv"
    main_table_manifest = visual / "MAIN_TABLE_DATA_PROVENANCE.csv"
    supplement_manifest = visual / "CURATED_SUPPLEMENT_TABLE_MANIFEST.csv"
    figure_rows = require_csv(figure_manifest, root, findings)
    main_table_rows = require_csv(main_table_manifest, root, findings)
    supplement_rows = require_csv(supplement_manifest, root, findings)

    expected_figure_ids = {f"F{index:02d}" for index in range(1, 9)}
    observed_figure_ids = {row.get("figure_id", "").strip() for row in figure_rows}
    if len(figure_rows) != 8 or observed_figure_ids != expected_figure_ids:
        findings.append(
            Finding(
                "ERROR",
                "FIGURE_PROVENANCE_SHAPE_CHANGED",
                relative(figure_manifest, root),
                f"Expected F01--F08 exactly once; observed {sorted(observed_figure_ids)}.",
            )
        )
    for row in figure_rows:
        for field, hash_field in (("figure_file", "sha256_pdf"), ("png_preview", "sha256_png")):
            value = (row.get(field) or "").strip()
            expected = (row.get(hash_field) or "").strip().lower()
            path = root / value
            if not path.is_file():
                findings.append(Finding("ERROR", "MISSING_GENERATED_FIGURE", value, f"Figure {row.get('figure_id')} is absent."))
            elif not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
                findings.append(Finding("ERROR", "FIGURE_PROVENANCE_HASH_MISMATCH", value, f"Figure {row.get('figure_id')} does not match provenance."))
        source_value = (row.get("source_data_csv") or "").strip()
        source = root / source_value
        if source_value and "/" not in source_value and not source.is_file():
            source = root / "results" / "tkde_rebuild" / source_value
        if not source.is_file():
            findings.append(Finding("ERROR", "MISSING_FIGURE_SOURCE_DATA", relative(source, root), f"Figure {row.get('figure_id')} lacks source CSV."))
            continue
        expected_source = (row.get("sha256_source_data") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_source) or sha256_file(source) != expected_source:
            findings.append(
                Finding(
                    "ERROR",
                    "FIGURE_SOURCE_HASH_MISMATCH",
                    relative(source, root),
                    f"Figure {row.get('figure_id')} source-data hash is stale.",
                )
            )
        upstream_paths = [value.strip() for value in (row.get("upstream_evidence") or "").split(";") if value.strip()]
        upstream_hashes = [value.strip().lower() for value in (row.get("upstream_sha256") or "").split(";") if value.strip()]
        if len(upstream_paths) != len(upstream_hashes):
            findings.append(Finding("ERROR", "FIGURE_UPSTREAM_PROVENANCE_SHAPE", relative(figure_manifest, root), f"Figure {row.get('figure_id')} upstream path/hash counts differ."))
        for value, expected in zip(upstream_paths, upstream_hashes):
            path = root / value
            if not path.is_file():
                findings.append(Finding("ERROR", "MISSING_FIGURE_UPSTREAM", value, f"Figure {row.get('figure_id')} upstream input is absent."))
            elif not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
                findings.append(Finding("ERROR", "FIGURE_UPSTREAM_HASH_MISMATCH", value, f"Figure {row.get('figure_id')} upstream hash is stale."))
        for field in ("generation_script", "style_authority"):
            value = (row.get(field) or "").strip()
            if not value or not (root / value).is_file():
                findings.append(Finding("ERROR", "MISSING_FIGURE_GENERATOR", value or relative(figure_manifest, root), f"Figure {row.get('figure_id')} lacks {field}."))

    expected_table_ids = {f"T{index:02d}" for index in range(1, 9)}
    observed_table_ids = {row.get("table_id", "").strip() for row in main_table_rows}
    if len(main_table_rows) != 8 or observed_table_ids != expected_table_ids:
        findings.append(
            Finding(
                "ERROR",
                "MAIN_TABLE_PROVENANCE_SHAPE_CHANGED",
                relative(main_table_manifest, root),
                f"Expected T01--T08 exactly once; observed {sorted(observed_table_ids)}.",
            )
        )
    manifested_tables: set[Path] = set()
    for row in main_table_rows:
        table_id = row.get("table_id")
        for field, hash_field, code in (
            ("table_file", "table_sha256", "MAIN_TABLE_HASH_MISMATCH"),
            ("source_data_csv", "source_data_sha256", "MAIN_TABLE_SOURCE_HASH_MISMATCH"),
        ):
            value = (row.get(field) or "").strip()
            path = (root / value).resolve()
            expected = (row.get(hash_field) or "").strip().lower()
            if not path.is_file():
                findings.append(Finding("ERROR", "MISSING_MAIN_TABLE_PROVENANCE_FILE", value, f"Table {table_id} provenance is incomplete."))
            elif not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
                findings.append(Finding("ERROR", code, value, f"Table {table_id} does not match its V2 provenance."))
            if field == "table_file" and path.is_file():
                manifested_tables.add(path)
        upstream_paths = [value.strip() for value in (row.get("upstream_sources") or "").split(";") if value.strip()]
        upstream_hashes = [value.strip().lower() for value in (row.get("upstream_sha256") or "").split(";") if value.strip()]
        if len(upstream_paths) != len(upstream_hashes):
            findings.append(Finding("ERROR", "MAIN_TABLE_UPSTREAM_PROVENANCE_SHAPE", relative(main_table_manifest, root), f"Table {table_id} upstream path/hash counts differ."))
        for value, expected in zip(upstream_paths, upstream_hashes):
            path = root / value
            if not path.is_file():
                findings.append(Finding("ERROR", "MISSING_MAIN_TABLE_UPSTREAM", value, f"Table {table_id} upstream input is absent."))
            elif not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
                findings.append(Finding("ERROR", "MAIN_TABLE_UPSTREAM_HASH_MISMATCH", value, f"Table {table_id} upstream hash is stale."))

    if len(supplement_rows) != 43:
        findings.append(Finding("ERROR", "SUPPLEMENT_MANIFEST_SHAPE_CHANGED", relative(supplement_manifest, root), f"Expected 43 curated fragments; observed {len(supplement_rows)}."))
    for row in supplement_rows:
        value = (row.get("table_fragment") or "").strip()
        path = (root / value).resolve()
        expected = (row.get("sha256") or "").strip().lower()
        if not path.is_file():
            findings.append(Finding("ERROR", "MISSING_CURATED_SUPPLEMENT_TABLE", value, "Curated supplement fragment is absent."))
        elif not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
            findings.append(Finding("ERROR", "CURATED_SUPPLEMENT_TABLE_HASH_MISMATCH", value, "Curated supplement fragment does not match its manifest."))
        else:
            manifested_tables.add(path)
        if row.get("orientation") != "portrait" or row.get("body_size") != "footnotesize" or row.get("raw_row_dump") != "False":
            findings.append(Finding("ERROR", "CURATED_SUPPLEMENT_LAYOUT_CONTRACT", value, "Fragment is not declared portrait/footnotesize/non-raw."))

    paper_root = (root / "paper_tkde").resolve()
    active = _active_tex(paper_root / "main.tex", paper_root)
    active.update(_active_tex(paper_root / "supplement" / "supplement.tex", paper_root))
    active_tables = {
        path.resolve()
        for path in active
        if path.parent.name == "tables" and path.suffix == ".tex"
    }
    if active_tables != manifested_tables:
        missing = sorted(relative(path, root) for path in active_tables - manifested_tables)
        extra = sorted(relative(path, root) for path in manifested_tables - active_tables)
        findings.append(Finding("ERROR", "ACTIVE_TABLE_MANIFEST_MISMATCH", "paper_tkde", f"Missing active={missing}; inactive manifested={extra}."))
    return findings, len(figure_rows), len(main_table_rows) + len(supplement_rows)


def blocked_numeric_findings(root: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    paths: set[Path] = set()
    for directory in (
        root / "results" / "tkde_rebuild" / "figure_data",
        root / "results" / "tkde_rebuild" / "table_data",
        root / "results" / "tkde_visual_rebuild",
    ):
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*.csv") if path.is_file())
    # Resource and runtime tables are presentation inputs even when not copied
    # into a figure_data/table_data subdirectory.
    for name in ("RESOURCE_BOUNDARIES.csv", "IBM_RUNTIME_FEASIBILITY.csv"):
        path = root / "results" / "tkde_rebuild" / name
        if path.is_file():
            paths.add(path)
    checked = 0
    for path in sorted(paths):
        try:
            rows = read_csv(path)
        except (csv.Error, UnicodeDecodeError):
            continue
        checked += 1
        for index, row in enumerate(rows, start=2):
            if not BLOCKED_RE.search(" ".join(str(value) for value in row.values())):
                continue
            for field, value in row.items():
                if not METRIC_FIELD_RE.search(field or ""):
                    continue
                normalized = str(value).strip().lower()
                if normalized in NA_VALUES:
                    continue
                try:
                    float(normalized)
                except ValueError:
                    continue
                findings.append(
                    Finding(
                        "ERROR",
                        "BLOCKED_CELL_HAS_NUMERIC_METRIC",
                        relative(path, root),
                        f"Row {index}, field {field!r} contains {value!r} despite blocked/unmeasured status.",
                    )
                )
    return findings, checked


def delta_ledger_findings(root: Path, ledger: Path) -> list[Finding]:
    if not ledger.is_file():
        return [Finding("ERROR", "MISSING_DELTA_LEDGER", relative(ledger, root), "Scientific delta ledger is absent.")]
    text = ledger.read_text(encoding="utf-8", errors="replace")
    findings: list[Finding] = []
    if not re.search(r"(?m)^`?ZERO_SCIENTIFIC_DELTAS`?\s*$", text):
        findings.append(Finding("ERROR", "NONZERO_OR_UNDECLARED_SCIENTIFIC_DELTA", relative(ledger, root), "Ledger does not declare ZERO_SCIENTIFIC_DELTAS on a standalone line."))
    if re.search(r"(?im)^\s*(?:[-*]\s*)?(?:scientific exception|nonzero delta)\s*:\s*(?!none\b)", text):
        findings.append(Finding("ERROR", "UNAPPROVED_SCIENTIFIC_EXCEPTION", relative(ledger, root), "Ledger contains a nonempty scientific exception."))
    return findings


def write_reports(
    report_dir: Path,
    findings: Sequence[Finding],
    counts: dict[str, object],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    verdict = "ZERO_SCIENTIFIC_DELTAS" if errors == 0 else "SCIENTIFIC_DELTA_GATE_FAILED"
    payload = {
        "verdict": verdict,
        "error_count": errors,
        "warning_count": warnings,
        "counts": counts,
        "baseline_archive_hashes": EXPECTED_BASELINE_ARCHIVES,
        "findings": [asdict(item) for item in findings],
    }
    (report_dir / "SCIENTIFIC_DELTA_GATE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Scientific delta gate",
        "",
        f"Verdict: **`{verdict}`**",
        "",
        (
            "This delta-only gate verifies the frozen scientific files, baseline ZIPs, typed claims, citation set, "
            "content allocation, generated provenance, and blocked-cell semantics. It does not rerun the settled "
            "statistical or empirical rebuild."
            if counts.get("baseline_archives_verified")
            else
            "This delta-only gate verifies the frozen scientific files, typed claims, citation set, content "
            "allocation, generated provenance, and blocked-cell semantics. The excluded baseline ZIPs were "
            "verified by the publisher before this public-package check. It does not rerun the settled "
            "statistical or empirical rebuild."
        ),
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("No scientific delta or evidence-status promotion was detected.")
    else:
        for item in findings:
            lines.append(f"- **{item.severity} {item.code}** `{item.path}`: {item.message}")
    (report_dir / "SCIENTIFIC_DELTA_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--skip-baseline-archives",
        action="store_true",
        help="For an extracted source archive only; the publisher must verify baseline ZIPs before packaging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    visual = root / RESULTS
    findings: list[Finding] = []
    counts: dict[str, object] = {}

    rows_findings, frozen_count = frozen_hash_findings(root, visual / "FROZEN_SCIENTIFIC_INPUT_HASHES.csv")
    findings.extend(rows_findings)
    counts["frozen_hash_rows"] = frozen_count
    if not args.skip_baseline_archives:
        findings.extend(baseline_archive_findings(root))
        counts["baseline_archives_verified"] = len(EXPECTED_BASELINE_ARCHIVES)
    else:
        counts["baseline_archives_verified"] = 0
    claim_issues, statuses = claim_findings(root, root / "results" / "tkde_rebuild" / "CLAIM_EVIDENCE_LEDGER.csv")
    findings.extend(claim_issues)
    counts["typed_claims"] = sum(statuses.values())
    counts["claim_status_counts"] = statuses
    citation_issues, bib_count, cited_count = citation_findings(root)
    findings.extend(citation_issues)
    counts["verified_references"] = bib_count
    counts["cited_references"] = cited_count
    allocation_issues, object_count = allocation_findings(
        root,
        visual / "VISUAL_OBJECT_INVENTORY.csv",
        visual / "CONTENT_ALLOCATION_MAP.csv",
    )
    findings.extend(allocation_issues)
    counts["allocated_objects"] = object_count
    provenance_issues, figure_count, table_count = provenance_findings(root)
    findings.extend(provenance_issues)
    counts["provenance_figures"] = figure_count
    counts["provenance_tables"] = table_count
    blocked_issues, csv_count = blocked_numeric_findings(root)
    findings.extend(blocked_issues)
    counts["blocked_semantics_csvs_checked"] = csv_count
    findings.extend(delta_ledger_findings(root, visual / "SCIENTIFIC_DELTA_LEDGER.md"))
    findings.sort(key=lambda item: (item.severity != "ERROR", item.path, item.code, item.message))

    report_dir = (args.report_dir or visual / "audits").resolve()
    write_reports(report_dir, findings, counts)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    print(f"scientific_delta_errors={errors} warnings={warnings} frozen_rows={frozen_count}")
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
