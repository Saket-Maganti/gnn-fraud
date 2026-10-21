#!/usr/bin/env python3
"""Build deterministic, curated TKDE manuscript and source-data archives.

The release is intentionally allowlisted.  It contains the authoritative
LaTeX dependency closure, the figures and tables referenced by that closure,
the rebuilt PDFs, derived analysis/provenance surfaces, and the scripts needed
to regenerate those derived surfaces.  It never packages raw datasets,
prediction exports, imported Kaggle workspaces, LaTeX build debris, or stale
manuscript assets.

Two independent builds of each archive must be byte-identical before either is
published.  A before/after hash snapshot also makes the command fail if an
input changes while packaging is in progress.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_tkde"
RESULTS = ROOT / "results" / "tkde_rebuild"
SCRIPTS = ROOT / "scripts" / "tkde_rebuild"
RELEASE = ROOT / "release"

MANUSCRIPT_ZIP = RELEASE / "tkde_manuscript_package.zip"
SOURCE_ZIP = RELEASE / "tkde_source_tables.zip"
ARTIFACT_MANIFEST = RELEASE / "tkde_artifact_manifest.csv"
EXCLUDED_MANIFEST = RELEASE / "tkde_excluded_file_manifest.csv"
REPRO_README = RELEASE / "tkde_reproducibility_readme.md"

REQUIRED_AUDIT_OUTPUTS = (
    "STARTING_STATE.md",
    "CURRENT_MANUSCRIPT_FORENSIC_AUDIT.md",
    "EVIDENCE_INVENTORY.csv",
    "CLAIM_EVIDENCE_LEDGER.csv",
    "NUMBER_PROVENANCE_MAP.csv",
    "LITERATURE_MATRIX.csv",
    "LITERATURE_SYNTHESIS.md",
    "NOVELTY_DIFFERENTIATION_TABLE.csv",
    "PAPER_IDENTITY_DECISION.md",
    "FIGURE_AUDIT.md",
    "TABLE_AND_FLOAT_PLACEMENT_AUDIT.md",
    "CITATION_VERIFICATION.md",
    "CITATION_COVERAGE_AUDIT.md",
    "REVIEWER_1_BENCHMARK.md",
    "REVIEWER_2_GRAPH_ML.md",
    "REVIEWER_3_AML.md",
    "REVIEWER_4_STATS_ARTIFACT.md",
    "REBUTTAL_AND_REPAIR_MATRIX.md",
    "FINAL_TKDE_READINESS_REPORT.md",
    "COMMAND_LOG.md",
)

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
PRIVATE_PATH_PATTERNS = (
    re.compile(b"/" + rb"Users/[A-Za-z0-9._-]+/"),
    re.compile(b"/" + rb"home/[A-Za-z0-9._-]+/"),
    # Concatenation prevents the packager source itself from containing the
    # exact private-path token that it rejects in release members.
    re.compile(b"/private" + b"/var/"),
    re.compile(rb"[A-Za-z]:\\\\" + rb"Users\\\\[^\\\\\r\n]+"),
)
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(b"BEGIN " + b"PRIVATE KEY"),
)
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".aux",
    ".blg",
    ".log",
    ".out",
    ".toc",
    ".zip.temp",
}


@dataclass(frozen=True)
class Entry:
    """One source-backed or generated archive member."""

    arcname: str
    data: bytes
    source: str
    role: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required release input is missing: {rel(path)}")
    if path.is_symlink():
        raise RuntimeError(f"Symlinks are not permitted in the release: {rel(path)}")
    return path


def read_entry(path: Path, role: str, arcname: str | None = None) -> Entry:
    require_file(path)
    return Entry(arcname or rel(path), path.read_bytes(), rel(path), role)


def safe_tex_target(base: Path, target: str) -> Path:
    """Resolve a local ``\\input`` target and reject path escape."""

    candidate = Path(target)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    resolved = (base / candidate).resolve()
    paper_root = PAPER.resolve()
    if resolved != paper_root and paper_root not in resolved.parents:
        raise RuntimeError(f"LaTeX dependency escapes paper_tkde: {target}")
    return require_file(resolved)


def manuscript_dependency_closure(entrypoint: Path) -> tuple[set[Path], set[Path]]:
    """Return current TeX/table sources and referenced figure files."""

    tex_files: set[Path] = set()
    figures: set[Path] = set()
    pending = [require_file(entrypoint)]
    compile_directory = entrypoint.parent
    input_pattern = re.compile(r"\\input\{([^}]+)\}")
    figure_pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")

    while pending:
        path = pending.pop()
        if path in tex_files:
            continue
        tex_files.add(path)
        text = path.read_text(encoding="utf-8")
        for target in input_pattern.findall(text):
            # TeX resolves \input paths from the compile working directory,
            # not from the directory of the currently expanded section.
            child = safe_tex_target(compile_directory, target)
            if child not in tex_files:
                pending.append(child)
        for target in figure_pattern.findall(text):
            requested = Path(target)
            candidates: list[Path] = []
            if requested.suffix:
                candidates.extend(
                    [path.parent / requested, PAPER / requested, PAPER / "figures" / requested.name]
                )
            else:
                for suffix in (".pdf", ".png"):
                    candidates.extend(
                        [
                            path.parent / requested.with_suffix(suffix),
                            PAPER / requested.with_suffix(suffix),
                            PAPER / "figures" / requested.with_suffix(suffix).name,
                        ]
                    )
            existing = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if existing is None:
                raise FileNotFoundError(f"Referenced figure is missing: {target} from {rel(path)}")
            if PAPER.resolve() not in existing.parents:
                raise RuntimeError(f"Figure dependency escapes paper_tkde: {target}")
            figures.add(existing)
            # Ship both current vector and raster renderings when they exist.
            for suffix in (".pdf", ".png"):
                sibling = existing.with_suffix(suffix)
                if sibling.is_file():
                    figures.add(sibling.resolve())
    return tex_files, figures


def private_path_match(data: bytes) -> str | None:
    for pattern in PRIVATE_PATH_PATTERNS:
        match = pattern.search(data)
        if match:
            return match.group(0).decode("utf-8", errors="replace")
    return None


def secret_match(data: bytes) -> str | None:
    for pattern in SECRET_PATTERNS:
        match = pattern.search(data)
        if match:
            return match.group(0).decode("utf-8", errors="replace")
    return None


def validate_entry(entry: Entry) -> None:
    path = PurePosixPath(entry.arcname)
    if path.is_absolute() or ".." in path.parts or entry.arcname.startswith("/"):
        raise RuntimeError(f"Unsafe archive member name: {entry.arcname}")
    lower = entry.arcname.lower()
    for suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
        if lower.endswith(suffix.lower()):
            raise RuntimeError(f"Forbidden build artifact selected: {entry.arcname}")
    normalized_parts = {part.lower() for part in path.parts}
    if ".ds_store" in normalized_parts or "__macosx" in normalized_parts:
        raise RuntimeError(f"Forbidden metadata selected: {entry.arcname}")
    if any(token in lower for token in ("data/raw/", "imported_outputs/", "kaggle_workspace/")):
        raise RuntimeError(f"Forbidden evidence payload selected: {entry.arcname}")
    if any(part in {"predictions", "prediction_exports"} for part in normalized_parts):
        raise RuntimeError(f"Prediction payload selected: {entry.arcname}")
    private = private_path_match(entry.data)
    if private:
        raise RuntimeError(
            f"Private absolute path found in selected file {entry.source}: {private!r}"
        )
    secret = secret_match(entry.data)
    if secret:
        raise RuntimeError(
            f"Credential-like secret found in selected file {entry.source}: {secret[:12]!r}"
        )


def role_for_result(path: Path) -> str:
    name = path.name.lower()
    if "audit" in name or "verification" in name or "validation" in name:
        return "audit_provenance"
    if "ledger" in name or "inventory" in name or "provenance" in name:
        return "evidence_provenance"
    if "literature" in name or "novelty" in name or "citation" in name:
        return "literature_provenance"
    return "derived_analysis"


def current_manuscript_entries() -> tuple[list[Entry], set[Path]]:
    main_tex, main_figures = manuscript_dependency_closure(PAPER / "main.tex")
    supp_tex, supp_figures = manuscript_dependency_closure(PAPER / "supplement" / "supplement.tex")
    dependency_paths = main_tex | supp_tex | main_figures | supp_figures
    fixed = {
        PAPER / "main.pdf": "rendered_main_pdf",
        PAPER / "supplement" / "supplement.pdf": "rendered_supplement_pdf",
        PAPER / "references.bib": "verified_bibliography",
        PAPER / "README_BUILD.md": "build_instructions",
    }
    dependency_paths.update(fixed)

    entries: list[Entry] = []
    for path in sorted(main_tex | supp_tex, key=rel):
        role = "generated_table" if "/tables/" in f"/{rel(path)}" else "manuscript_source"
        entries.append(read_entry(path, role))
    for path in sorted(main_figures | supp_figures, key=rel):
        entries.append(read_entry(path, "generated_figure"))
    for path, role in sorted(fixed.items(), key=lambda pair: rel(pair[0])):
        entries.append(read_entry(path, role))
    return entries, dependency_paths


def result_entries() -> tuple[list[Entry], set[Path]]:
    """Select only rebuild outputs; deliberately omit private starting-state notes."""

    for name in REQUIRED_AUDIT_OUTPUTS:
        require_file(RESULTS / name)
    paths: set[Path] = set()
    for path in RESULTS.glob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md"}:
            if path.name != "STARTING_STATE.md":
                paths.add(path)
    for subdir in (RESULTS / "figure_data", RESULTS / "table_data"):
        if subdir.is_dir():
            paths.update(path for path in subdir.glob("*.csv") if path.is_file())
    return [read_entry(path, role_for_result(path)) for path in sorted(paths, key=rel)], paths


def script_entries() -> tuple[list[Entry], set[Path]]:
    paths = {
        path
        for path in SCRIPTS.iterdir()
        if path.is_file() and path.suffix.lower() in {".py", ".sh"}
    }
    entries = [read_entry(path, "regeneration_script") for path in sorted(paths, key=rel)]
    return entries, paths


def environment_entries() -> tuple[list[Entry], set[Path]]:
    paths = {
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "pyproject.toml",
    }
    entries = [read_entry(path, "environment_specification") for path in sorted(paths, key=rel)]
    return entries, paths


def deduplicate(entries: Iterable[Entry]) -> list[Entry]:
    by_name: dict[str, Entry] = {}
    for entry in entries:
        previous = by_name.get(entry.arcname)
        if previous is not None and previous.data != entry.data:
            raise RuntimeError(f"Conflicting archive members: {entry.arcname}")
        by_name[entry.arcname] = entry
    result = [by_name[name] for name in sorted(by_name)]
    for entry in result:
        validate_entry(entry)
    return result


def managed_files() -> set[Path]:
    roots = (PAPER, RESULTS, SCRIPTS)
    return {
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def snapshot(paths: Iterable[Path]) -> dict[str, tuple[int, str]]:
    return {rel(path): (path.stat().st_size, sha256_file(path)) for path in sorted(paths, key=rel)}


def exclusion_reason(path: Path) -> tuple[str, str]:
    relative = rel(path)
    lower = relative.lower()
    if path.name == "STARTING_STATE.md":
        return "private_path_metadata", "Local backup and absolute workstation paths are not releasable."
    if path.name == ".DS_Store" or "__MACOSX" in path.parts:
        return "os_metadata", "Operating-system metadata is excluded."
    if path.suffix.lower() in {".aux", ".bbl", ".blg", ".log", ".out", ".toc"}:
        return "latex_build_debris", "Local LaTeX intermediates are reproducible and may disclose build details."
    if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
        return "python_build_debris", "Interpreter cache files are excluded."
    if "/template" in lower:
        return "vendor_template", "Reference templates are not part of the authored release."
    if relative.startswith("paper_tkde/figures/"):
        return "stale_figure", "The figure is not referenced by the authoritative rebuilt TeX closure."
    if relative.startswith("paper_tkde/tables/") or relative.startswith("paper_tkde/supplement/tables/"):
        return "stale_table", "The table is not referenced by the authoritative rebuilt TeX closure."
    if relative.startswith("paper_tkde/sections/") or relative.startswith("paper_tkde/supplement/sections/"):
        return "stale_section", "The section is not referenced by the authoritative rebuilt TeX closure."
    if relative.startswith("paper_tkde/"):
        return "historical_manuscript_artifact", "Planning notes and superseded manuscript surfaces are excluded."
    return "unselected_build_artifact", "The file is outside the curated release allowlist."


def excluded_manifest_bytes(selected_paths: set[Path], all_managed: set[Path]) -> bytes:
    output = io.StringIO(newline="")
    fields = ["kind", "path_or_pattern", "sha256", "bytes", "reason"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    policies = [
        ("raw_datasets", "data/raw/**", "Raw and staged datasets are intentionally not distributed."),
        ("prediction_payloads", "**/predictions/**; **/prediction_exports/**", "Per-example predictions are omitted from the compact manuscript release."),
        ("imported_workspaces", "kaggle_workspace/imported_outputs/**", "Imported run payloads stay under their evidence locks and are not duplicated."),
        ("credentials", "**/.env; **/*token*; Kaggle credentials", "Credentials and tokens are never release material."),
        ("latex_build_debris", "**/*.aux; **/*.bbl; **/*.blg; **/*.log; **/*.out; **/*.toc", "Local LaTeX intermediates are reproducible and excluded."),
        ("temporary_files", "**/.DS_Store; **/__MACOSX/**; **/tmp/**; **/*.zip.temp", "Metadata and temporary files fail release hygiene."),
        ("private_paths", "absolute workstation home directories", "Private absolute workstation paths fail content validation."),
    ]
    for kind, pattern, reason in policies:
        writer.writerow(
            {"kind": f"policy:{kind}", "path_or_pattern": pattern, "sha256": "", "bytes": "", "reason": reason}
        )
    for path in sorted(all_managed - selected_paths, key=rel):
        kind, reason = exclusion_reason(path)
        writer.writerow(
            {
                "kind": kind,
                "path_or_pattern": rel(path),
                # Excluded intermediates are intentionally not checksum
                # authorities; their volatile bytes must not perturb a
                # deterministic archive built from unchanged selected inputs.
                "sha256": "",
                "bytes": "",
                "reason": reason,
            }
        )
    return output.getvalue().encode("utf-8")


def reproducibility_readme_bytes() -> bytes:
    text = """# FraudShiftBench TKDE release: reproducibility notes

This is a curated no-training release for the rebuilt TKDE manuscript.  The
archives contain the authoritative main-paper and supplement sources/PDFs,
only the figures and tables referenced by those sources, derived analysis and
provenance tables, and deterministic regeneration scripts.

## Evidence boundary

Raw datasets, per-example prediction exports, imported Kaggle workspaces, and
resource-blocked payloads are deliberately absent.  The included CSV files are
derived, claim-bounded analysis surfaces; they are not substitutes for the
locked source evidence.  Recomputing the analyses from raw/locked evidence
therefore requires the full repository and its locally staged datasets.  The
source-data archive is sufficient to regenerate the paper figures, LaTeX
tables, and bibliography from the included derived CSV surfaces without model
training.

## Archive contents

- `tkde_manuscript_package.zip`: main paper, supplement, bibliography,
  referenced tables/figures, and manuscript audit provenance.
- `tkde_source_tables.zip`: deterministic builders, derived CSV/Markdown
  analyses, figure/table source data, current generated assets, environment
  specifications, and audit provenance.
- `tkde_artifact_manifest.csv`: SHA-256 and size for every archive member and
  each completed ZIP.
- `tkde_excluded_file_manifest.csv`: policy and file-level exclusions.

Every ZIP member has a fixed timestamp and normalized permissions.  Members
are sorted, path-safe, free of private absolute workstation paths, and listed
in an internal `ARCHIVE_MANIFEST.csv`.  The packager builds each ZIP twice and
publishes it only if the two SHA-256 hashes agree.

## Regenerate derived assets

From an extracted source-data archive, install the declared Python dependencies
and run:

```bash
python3 scripts/tkde_rebuild/make_figures.py
python3 scripts/tkde_rebuild/build_tables.py
python3 scripts/tkde_rebuild/build_bibliography.py
```

The complete full-repository, no-training regeneration order is documented in
`paper_tkde/README_BUILD.md`.  It starts with the evidence inventory and claim
ledger, then recomputes analyses before rendering figures and tables.

## Compile and audit the PDFs

The local build uses IEEEtran, `pdflatex`, and BibTeX:

```bash
bash scripts/tkde_rebuild/compile_papers.sh
python3 scripts/tkde_rebuild/audit_manuscript.py
```

The release excludes `.aux`, `.bbl`, `.blg`, `.log`, `.out`, and `.toc` files.
They are regenerated by the strict BibTeX build cycle.

## Verify checksums

Use `tkde_artifact_manifest.csv` as the external checksum authority.  For
example, on macOS:

```bash
shasum -a 256 release/tkde_manuscript_package.zip
shasum -a 256 release/tkde_source_tables.zip
```
"""
    return text.encode("utf-8")


def archive_manifest_entry(entries: list[Entry]) -> Entry:
    output = io.StringIO(newline="")
    fields = ["path", "sha256", "bytes", "source", "role"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        writer.writerow(
            {
                "path": entry.arcname,
                "sha256": entry.sha256,
                "bytes": len(entry.data),
                "source": entry.source,
                "role": entry.role,
            }
        )
    return Entry(
        "ARCHIVE_MANIFEST.csv",
        output.getvalue().encode("utf-8"),
        "generated by scripts/tkde_rebuild/build_release.py",
        "archive_manifest",
    )


def zip_info(entry: Entry) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(entry.arcname, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if entry.arcname.endswith(".sh") or entry.arcname.endswith(".py") else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.flag_bits = 0
    return info


def write_zip(path: Path, entries: list[Entry]) -> list[Entry]:
    base_entries = deduplicate(entries)
    all_entries = deduplicate([*base_entries, archive_manifest_entry(base_entries)])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in all_entries:
            archive.writestr(zip_info(entry), entry.data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return all_entries


def validate_zip(path: Path, expected: list[Entry]) -> None:
    expected_by_name = {entry.arcname: entry for entry in expected}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != sorted(names):
            raise RuntimeError(f"Archive member order is not canonical: {path.name}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate archive member: {path.name}")
        if set(names) != set(expected_by_name):
            missing = sorted(set(expected_by_name) - set(names))
            extra = sorted(set(names) - set(expected_by_name))
            raise RuntimeError(f"Archive listing mismatch for {path.name}: missing={missing}, extra={extra}")
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"CRC failure in {path.name}: {corrupt}")
        for info in archive.infolist():
            if info.date_time != FIXED_ZIP_TIME:
                raise RuntimeError(f"Non-deterministic timestamp in {path.name}: {info.filename}")
            data = archive.read(info.filename)
            entry = expected_by_name[info.filename]
            if sha256_bytes(data) != entry.sha256:
                raise RuntimeError(f"Member checksum mismatch: {path.name}:{info.filename}")
            validate_entry(entry)


def build_twice(destination: Path, entries: list[Entry], temp_dir: Path) -> tuple[list[Entry], str]:
    first = temp_dir / f"{destination.stem}.first.zip"
    second = temp_dir / f"{destination.stem}.second.zip"
    expected_first = write_zip(first, entries)
    expected_second = write_zip(second, entries)
    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    if first_hash != second_hash:
        raise RuntimeError(f"Non-deterministic ZIP build: {destination.name}")
    validate_zip(first, expected_first)
    validate_zip(second, expected_second)
    if [(entry.arcname, entry.sha256) for entry in expected_first] != [
        (entry.arcname, entry.sha256) for entry in expected_second
    ]:
        raise RuntimeError(f"Archive manifests differ between builds: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(first, destination)
    validate_zip(destination, expected_first)
    return expected_first, first_hash


def artifact_manifest_bytes(
    archives: Mapping[str, tuple[Path, list[Entry], str]],
) -> bytes:
    output = io.StringIO(newline="")
    fields = ["archive", "path", "sha256", "bytes", "source", "role"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for archive_name in sorted(archives):
        archive_path, entries, archive_hash = archives[archive_name]
        for entry in sorted(entries, key=lambda item: item.arcname):
            writer.writerow(
                {
                    "archive": archive_name,
                    "path": entry.arcname,
                    "sha256": entry.sha256,
                    "bytes": len(entry.data),
                    "source": entry.source,
                    "role": entry.role,
                }
            )
        writer.writerow(
            {
                "archive": archive_name,
                "path": "__ARCHIVE__",
                "sha256": archive_hash,
                "bytes": archive_path.stat().st_size,
                "source": rel(archive_path),
                "role": "deterministic_zip",
            }
        )
    return output.getvalue().encode("utf-8")


def validate_external_manifest(archives: Mapping[str, tuple[Path, list[Entry], str]]) -> None:
    with ARTIFACT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    archive_rows = [row for row in rows if row["path"] == "__ARCHIVE__"]
    if len(archive_rows) != len(archives):
        raise RuntimeError("External manifest does not contain one summary row per archive.")
    for row in archive_rows:
        archive_path, _, expected_hash = archives[row["archive"]]
        if row["sha256"] != expected_hash or sha256_file(archive_path) != expected_hash:
            raise RuntimeError(f"External manifest archive hash mismatch: {row['archive']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the existing release archives and external manifest without rebuilding.",
    )
    return parser.parse_args()


def check_existing() -> None:
    for path in (MANUSCRIPT_ZIP, SOURCE_ZIP, ARTIFACT_MANIFEST, EXCLUDED_MANIFEST, REPRO_README):
        require_file(path)
    with ARTIFACT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for archive in (MANUSCRIPT_ZIP, SOURCE_ZIP):
        summary = next(
            (row for row in rows if row["archive"] == archive.name and row["path"] == "__ARCHIVE__"),
            None,
        )
        if summary is None or summary["sha256"] != sha256_file(archive):
            raise RuntimeError(f"Existing archive does not match external manifest: {archive.name}")
        with zipfile.ZipFile(archive, "r") as handle:
            if handle.testzip():
                raise RuntimeError(f"Existing archive failed CRC validation: {archive.name}")
            names = handle.namelist()
            if names != sorted(names):
                raise RuntimeError(f"Existing archive is not canonically ordered: {archive.name}")
            member_rows = {
                row["path"]: row
                for row in rows
                if row["archive"] == archive.name and row["path"] != "__ARCHIVE__"
            }
            if set(names) != set(member_rows):
                raise RuntimeError(f"Existing archive listing does not match manifest: {archive.name}")
            for info in handle.infolist():
                if info.date_time != FIXED_ZIP_TIME:
                    raise RuntimeError(f"Existing archive has non-fixed timestamps: {archive.name}")
                data = handle.read(info.filename)
                private = private_path_match(data)
                if private:
                    raise RuntimeError(f"Private path in {archive.name}:{info.filename}")
                row = member_rows[info.filename]
                if row["sha256"] != sha256_bytes(data) or int(row["bytes"]) != len(data):
                    raise RuntimeError(
                        f"Existing member does not match manifest: {archive.name}:{info.filename}"
                    )
                validate_entry(
                    Entry(info.filename, data, row["source"], row["role"])
                )
    for metadata in (ARTIFACT_MANIFEST, EXCLUDED_MANIFEST, REPRO_README):
        private = private_path_match(metadata.read_bytes())
        if private:
            raise RuntimeError(f"Private path found in release metadata: {metadata.name}")
    print("PASS: existing TKDE release archives match their manifest and hygiene policy.")


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_existing()
        return 0

    RELEASE.mkdir(parents=True, exist_ok=True)
    initial_managed = managed_files()
    initial_snapshot = snapshot(initial_managed)

    manuscript, manuscript_paths = current_manuscript_entries()
    results, result_paths = result_entries()
    scripts, script_paths = script_entries()
    environment, environment_paths = environment_entries()
    selected_paths = manuscript_paths | result_paths | script_paths | environment_paths

    # These generated files are deterministic and intentionally included in both archives.
    excluded_bytes = excluded_manifest_bytes(selected_paths, initial_managed)
    readme_bytes = reproducibility_readme_bytes()
    if private_path_match(excluded_bytes) or private_path_match(readme_bytes):
        raise RuntimeError("Generated release metadata contains a private absolute path.")
    if secret_match(excluded_bytes) or secret_match(readme_bytes):
        raise RuntimeError("Generated release metadata contains credential-like content.")
    release_metadata = [
        Entry(
            "tkde_reproducibility_readme.md",
            readme_bytes,
            rel(REPRO_README),
            "release_instructions",
        ),
        Entry(
            "tkde_excluded_file_manifest.csv",
            excluded_bytes,
            rel(EXCLUDED_MANIFEST),
            "exclusion_manifest",
        ),
    ]

    manuscript_entries = deduplicate([*manuscript, *results, *release_metadata])
    source_entries = deduplicate(
        [*scripts, *results, *environment, *manuscript, *release_metadata]
    )

    with tempfile.TemporaryDirectory(prefix="tkde_release_", dir=RELEASE) as directory:
        temp_dir = Path(directory)
        staged_manuscript = temp_dir / MANUSCRIPT_ZIP.name
        staged_source = temp_dir / SOURCE_ZIP.name
        manuscript_members, manuscript_hash = build_twice(
            staged_manuscript, manuscript_entries, temp_dir
        )
        source_members, source_hash = build_twice(staged_source, source_entries, temp_dir)

        final_managed = managed_files()
        final_snapshot = snapshot(final_managed)
        if initial_snapshot != final_snapshot:
            raise RuntimeError(
                "A managed manuscript/source input changed during packaging; no ZIPs were published."
            )

        # Publish only after both duplicate builds and the stability snapshot pass.
        os.replace(staged_manuscript, MANUSCRIPT_ZIP)
        os.replace(staged_source, SOURCE_ZIP)

    EXCLUDED_MANIFEST.write_bytes(excluded_bytes)
    REPRO_README.write_bytes(readme_bytes)

    archives = {
        MANUSCRIPT_ZIP.name: (MANUSCRIPT_ZIP, manuscript_members, manuscript_hash),
        SOURCE_ZIP.name: (SOURCE_ZIP, source_members, source_hash),
    }
    ARTIFACT_MANIFEST.write_bytes(artifact_manifest_bytes(archives))
    validate_external_manifest(archives)

    print(f"PASS: {MANUSCRIPT_ZIP.name} members={len(manuscript_members)} sha256={manuscript_hash}")
    print(f"PASS: {SOURCE_ZIP.name} members={len(source_members)} sha256={source_hash}")
    print(f"PASS: artifact manifest rows={sum(len(value[1]) + 1 for value in archives.values())}")
    print(f"PASS: excluded manifest bytes={EXCLUDED_MANIFEST.stat().st_size}")
    print("PASS: no raw data, predictions, build debris, stale assets, or private paths packaged")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # concise CLI failure with a nonzero exit code
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
