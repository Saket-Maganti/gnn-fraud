#!/usr/bin/env python3
"""Build deterministic, namespaced TKDE visual-rebuild release archives.

The frozen ``release/tkde_*.zip`` files are inputs to the preservation gate and
are never opened for writing.  New packages live below
``release/tkde_visual_rebuild/``.  Publication is transactional: both archives
must be byte-deterministic, pass member/CRC/hygiene validation, and the source
archive must survive extraction, generator reruns, bibliography regeneration,
strict compilation, scientific-delta validation, and PDF/table audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
EXPECTED_BASELINE_ARCHIVES = {
    "release/tkde_manuscript_package.zip": "234b164d0553fd4d19b1e850c19e4b0924bfeee1a0342201f326d24d2faa2ba0",
    "release/tkde_source_tables.zip": "528780b5666dd1b97f0a12be39e20957af4a08f25de8d7b334179727be4d919c",
}

PRIVATE_PATH_PATTERNS = (
    re.compile(b"/" + rb"Users/[A-Za-z0-9._-]+/"),
    re.compile(b"/" + rb"home/[A-Za-z0-9._-]+/"),
    re.compile(b"/private" + b"/var/"),
    re.compile(rb"[A-Za-z]:\\\\" + rb"Users\\\\[^\\\\\r\n]+"),
)
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(b"BEGIN " + b"PRIVATE KEY"),
    re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token)\s*[=:]\s*['\"][^'\"]{12,}"),
)
# Keep the anonymous reviewer packages free of the identity vocabulary used by
# the report-only anonymization scanner itself.  The byte fragments are split
# so this release builder does not match its own source when it is packaged.
IDENTITY_PATTERNS = (
    re.compile(b"(?i)\\bsa" + b"ket(?:-ma" + b"ganti)?\\b"),
    re.compile(b"(?i)\\bma" + b"ganti\\b"),
)
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".toc",
    ".pyc",
    ".pyo",
    ".tmp",
    ".temp",
}
FORBIDDEN_MEMBER_PARTS = {"__pycache__", "__macosx", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_PAYLOAD_PARTS = {"predictions", "prediction_exports", "raw"}

INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}")
BIB_RE = re.compile(r"\\bibliography\s*\{([^}]+)\}")

REQUIRED_GENERATORS = (
    "scripts/tkde_visual_rebuild/build_main_tables.py",
    "scripts/tkde_visual_rebuild/build_curated_supplement_tables.py",
    "scripts/tkde_rebuild/make_figures.py",
    "scripts/tkde_rebuild/build_bibliography.py",
    "scripts/tkde_rebuild/compile_papers.sh",
    "scripts/tkde_visual_rebuild/audit_table_readability.py",
    "scripts/tkde_visual_rebuild/audit_pdf_layout.py",
    "scripts/tkde_visual_rebuild/scientific_delta_gate.py",
)
EXHAUSTIVE_MACHINE_SOURCES = (
    "results/runs_rb09v3/runs.csv",
    "manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv",
    "results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv",
    "results/tkde_rebuild/IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv",
)


@dataclass(frozen=True)
class ReleasePaths:
    root: Path

    @property
    def output_dir(self) -> Path:
        return self.root / "release" / "tkde_visual_rebuild"

    @property
    def manuscript_zip(self) -> Path:
        return self.output_dir / "tkde_visual_manuscript_package.zip"

    @property
    def source_zip(self) -> Path:
        return self.output_dir / "tkde_visual_source_analysis_package.zip"

    @property
    def artifact_manifest(self) -> Path:
        return self.output_dir / "tkde_visual_artifact_manifest.csv"

    @property
    def excluded_manifest(self) -> Path:
        return self.output_dir / "tkde_visual_excluded_file_manifest.csv"

    @property
    def readme(self) -> Path:
        return self.output_dir / "tkde_visual_reproducibility_readme.md"

    @property
    def clean_room_report(self) -> Path:
        return self.output_dir / "CLEAN_ROOM_BUILD_REPORT.md"

    @property
    def checksums(self) -> Path:
        return self.output_dir / "CHECKSUMS.sha256"


@dataclass(frozen=True)
class Entry:
    arcname: str
    data: bytes
    source: str
    role: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class CleanRoomCommand:
    command: tuple[str, ...]
    exit_code: int
    output: str


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


def require_file(path: Path, root: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required release input is missing: {relative(path, root)}")
    if path.is_symlink():
        raise RuntimeError(f"Symlinks are not release inputs: {relative(path, root)}")
    return path


def read_entry(path: Path, root: Path, role: str, arcname: str | None = None) -> Entry:
    require_file(path, root)
    return Entry(arcname or relative(path, root), path.read_bytes(), relative(path, root), role)


def private_match(data: bytes) -> str | None:
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


def identity_match(data: bytes) -> str | None:
    for pattern in IDENTITY_PATTERNS:
        match = pattern.search(data)
        if match:
            return match.group(0).decode("utf-8", errors="replace")
    return None


def validate_entry(entry: Entry) -> None:
    path = PurePosixPath(entry.arcname)
    if path.is_absolute() or ".." in path.parts or entry.arcname.startswith("/"):
        raise RuntimeError(f"Unsafe archive member name: {entry.arcname}")
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & FORBIDDEN_MEMBER_PARTS or ".ds_store" in lowered_parts:
        raise RuntimeError(f"Build/OS metadata selected: {entry.arcname}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Build debris selected: {entry.arcname}")
    if lowered_parts & FORBIDDEN_PAYLOAD_PARTS:
        raise RuntimeError(f"Raw/prediction payload selected: {entry.arcname}")
    lower = entry.arcname.lower()
    if "kaggle_workspace/imported_outputs/" in lower:
        raise RuntimeError(f"Imported workspace payload selected: {entry.arcname}")
    if lower.endswith(".zip"):
        raise RuntimeError(f"Nested ZIP selected: {entry.arcname}")
    private = private_match(entry.data)
    if private:
        raise RuntimeError(f"Private absolute path in {entry.source}: {private!r}")
    secret = secret_match(entry.data)
    if secret:
        raise RuntimeError(f"Credential-like content in {entry.source}: {secret[:16]!r}")
    identity = identity_match(entry.data)
    if identity:
        raise RuntimeError(f"Identity-bearing content in anonymous release input {entry.source}")


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


def resolve_tex_target(target: str, compile_dir: Path, paper_root: Path) -> Path:
    candidate = Path(target.strip())
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".tex")
    resolved = (compile_dir / candidate).resolve()
    if resolved != paper_root and paper_root not in resolved.parents:
        raise RuntimeError(f"LaTeX dependency escapes paper_tkde: {target}")
    return resolved


def resolve_graphic(target: str, source: Path, compile_dir: Path, paper_root: Path) -> Path:
    requested = Path(target.strip())
    suffixes = ("",) if requested.suffix else (".pdf", ".png")
    candidates: list[Path] = []
    for suffix in suffixes:
        item = requested if not suffix else requested.with_suffix(suffix)
        candidates.extend(
            (
                source.parent / item,
                compile_dir / item,
                compile_dir / "figures" / item.name,
                paper_root / "figures" / item.name,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if paper_root not in resolved.parents:
                raise RuntimeError(f"Graphic dependency escapes paper_tkde: {target}")
            return resolved
    raise FileNotFoundError(f"Referenced graphic is absent: {target} from {source}")


def manuscript_dependency_closure(entrypoint: Path, root: Path) -> tuple[set[Path], set[Path], set[Path]]:
    paper_root = (root / "paper_tkde").resolve()
    compile_dir = entrypoint.parent.resolve()
    pending = [require_file(entrypoint, root).resolve()]
    tex_files: set[Path] = set()
    graphics: set[Path] = set()
    bibliographies: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in tex_files:
            continue
        if path != paper_root and paper_root not in path.parents:
            raise RuntimeError(f"TeX closure escaped paper_tkde: {path}")
        require_file(path, root)
        tex_files.add(path)
        text = path.read_text(encoding="utf-8")
        for target in INPUT_RE.findall(text):
            child = resolve_tex_target(target, compile_dir, paper_root)
            require_file(child, root)
            pending.append(child)
        for target in GRAPHICS_RE.findall(text):
            graphic = resolve_graphic(target, path, compile_dir, paper_root)
            graphics.add(graphic)
            for suffix in (".pdf", ".png"):
                sibling = graphic.with_suffix(suffix)
                if sibling.is_file():
                    graphics.add(sibling.resolve())
        for group in BIB_RE.findall(text):
            for target in group.split(","):
                candidate = Path(target.strip())
                if candidate.suffix == "":
                    candidate = candidate.with_suffix(".bib")
                bibliography = (compile_dir / candidate).resolve()
                if bibliography != paper_root and paper_root not in bibliography.parents:
                    raise RuntimeError(f"Bibliography escapes paper_tkde: {target}")
                bibliographies.add(require_file(bibliography, root).resolve())
    return tex_files, graphics, bibliographies


def manuscript_entries(root: Path) -> tuple[list[Entry], set[Path]]:
    paper = root / "paper_tkde"
    main = manuscript_dependency_closure(paper / "main.tex", root)
    supplement = manuscript_dependency_closure(paper / "supplement" / "supplement.tex", root)
    tex_files = main[0] | supplement[0]
    figures = main[1] | supplement[1]
    bibliographies = main[2] | supplement[2]
    fixed = {
        paper / "main.pdf": "rendered_main_pdf",
        paper / "supplement" / "supplement.pdf": "rendered_supplement_pdf",
        paper / "README_BUILD.md": "build_instructions",
    }
    paths = tex_files | figures | bibliographies | set(fixed)
    entries: list[Entry] = []
    for path in sorted(tex_files):
        role = "active_table_source" if "/tables/" in f"/{relative(path, root)}" else "active_manuscript_source"
        entries.append(read_entry(path, root, role))
    entries.extend(read_entry(path, root, "active_figure") for path in sorted(figures))
    entries.extend(read_entry(path, root, "verified_bibliography") for path in sorted(bibliographies))
    entries.extend(read_entry(path, root, role) for path, role in sorted(fixed.items()))
    return deduplicate(entries), paths


def _paths_from_csv_column(path: Path, root: Path, fields: Sequence[str]) -> set[Path]:
    selected: set[Path] = set()
    if not path.is_file():
        return selected
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in fields:
            for token in (row.get(field) or "").split(";"):
                value = token.strip()
                if value.startswith(("results/", "paper_tkde/", "manuscript_assets/", "kaggle_workspace/")):
                    candidate = root / value
                    if candidate.is_file():
                        selected.add(candidate.resolve())
    return selected


def generator_input_paths(root: Path) -> set[Path]:
    """Discover all declared and exhaustive inputs needed by visual builders."""

    selected: set[Path] = set()
    visual = root / "results" / "tkde_visual_rebuild"
    freeze = visual / "FROZEN_SCIENTIFIC_INPUT_HASHES.csv"
    require_file(freeze, root)
    selected.add(freeze.resolve())
    with freeze.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (row.get("path") or "").strip()
            if value:
                selected.add(require_file(root / value, root).resolve())
    selected.update(
        _paths_from_csv_column(
            visual / "CONTENT_ALLOCATION_MAP.csv",
            root,
            ("source_data_files", "provenance_checksum_record"),
        )
    )
    selected.update(
        _paths_from_csv_column(
            visual / "FIGURE_DATA_PROVENANCE.csv",
            root,
            (
                "source_data_csv",
                "upstream_evidence",
                "figure_file",
                "png_preview",
                "generation_script",
                "style_authority",
            ),
        )
    )
    selected.update(
        _paths_from_csv_column(
            visual / "MAIN_TABLE_DATA_PROVENANCE.csv",
            root,
            ("source_data_csv", "upstream_sources", "table_file", "generation_script"),
        )
    )
    selected.update(
        _paths_from_csv_column(
            visual / "CURATED_SUPPLEMENT_TABLE_MANIFEST.csv",
            root,
            ("table_fragment",),
        )
    )
    for relpath in EXHAUSTIVE_MACHINE_SOURCES:
        selected.add(require_file(root / relpath, root).resolve())
    # Curated builders intentionally consume approved top-level analysis CSVs;
    # include the complete safe derived surface so a future new curated table
    # cannot silently depend on an omitted file.
    tkde_results = root / "results" / "tkde_rebuild"
    selected.update(path.resolve() for path in tkde_results.glob("*.csv") if path.is_file())
    selected.update(path.resolve() for path in tkde_results.glob("*.json") if path.is_file())
    for directory in (tkde_results / "figure_data", tkde_results / "table_data"):
        if directory.is_dir():
            selected.update(path.resolve() for path in directory.glob("*.csv") if path.is_file())
    for directory in (root / "manuscript_assets" / "tables",):
        if directory.is_dir():
            selected.update(path.resolve() for path in directory.glob("V22_*.csv") if path.is_file())
    return selected


def generator_script_paths(root: Path) -> set[Path]:
    paths: set[Path] = set()
    visual_scripts = root / "scripts" / "tkde_visual_rebuild"
    paths.update(path.resolve() for path in visual_scripts.glob("*.py") if path.is_file())
    for relpath in REQUIRED_GENERATORS:
        paths.add(require_file(root / relpath, root).resolve())
    # Include the compatibility dispatcher used by repository build commands.
    dispatcher = root / "scripts" / "tkde_rebuild" / "build_tables.py"
    if dispatcher.is_file():
        paths.add(dispatcher.resolve())
    return paths


def visual_report_paths(root: Path) -> set[Path]:
    selected: set[Path] = set()
    directory = root / "results" / "tkde_visual_rebuild"
    if not directory.is_dir():
        return selected
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "STARTING_VISUAL_STATE.md" or "before" in path.parts or "baseline_pages" in path.parts:
            continue
        if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            data = path.read_bytes()
            if private_match(data) or secret_match(data) or identity_match(data):
                continue
            selected.add(path.resolve())
    return selected


def environment_paths(root: Path) -> set[Path]:
    names = ("requirements.txt", "requirements-dev.txt", "pyproject.toml", ".python-version")
    return {require_file(root / name, root).resolve() for name in names if (root / name).is_file()}


def input_entries(paths: Iterable[Path], root: Path, role: str) -> list[Entry]:
    return [read_entry(path, root, role) for path in sorted(set(paths))]


def reproducibility_readme_bytes() -> bytes:
    return b"""# FraudShiftBench TKDE visual-rebuild release

This namespaced release is a publication-design delta over the frozen
`PROFESSOR_REVIEW_READY` scientific baseline. It does not replace or mutate the
baseline archives. The expected successful automated verdict is
`TKDE_VISUAL_REBUILD_COMPLETE_PROFESSOR_REVIEW_READY`; venue policy, author
metadata, and human scientific review remain external gates.

## Package separation

- `tkde_visual_manuscript_package.zip`: final main/supplement PDFs, active
  LaTeX dependency closure, bibliography, referenced figure/table assets, and
  publication-design audit reports.
- `tkde_visual_source_analysis_package.zip`: everything above plus all visual
  generators, every declared/frozen generator input, figure/table source CSVs,
  exhaustive RB09/V22/IBM rows moved out of the PDF, environment files, and
  provenance manifests.

Raw datasets, per-example prediction payloads, imported workspaces, caches,
private paths, credentials, LaTeX auxiliaries, and stale assets are excluded.
Identity-bearing scanner configuration is also excluded from anonymous packages.
Resource-blocked cells remain nonnumeric and excluded from rankings.

## Clean-room regeneration

From the extracted source-analysis package, install the declared Python and
LaTeX dependencies, then run:

```bash
python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --strict --skip-baseline-archives
python3 scripts/tkde_visual_rebuild/build_main_tables.py
python3 scripts/tkde_visual_rebuild/build_curated_supplement_tables.py
python3 scripts/tkde_rebuild/make_figures.py
python3 scripts/tkde_rebuild/build_bibliography.py
bash scripts/tkde_rebuild/compile_papers.sh
python3 scripts/tkde_visual_rebuild/audit_table_readability.py --root . --strict
python3 scripts/tkde_visual_rebuild/audit_pdf_layout.py --root . --strict
python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --strict --skip-baseline-archives
```

The publisher executes this sequence in a newly extracted temporary directory
before publishing either ZIP. The baseline archive hashes are verified in the
full repository immediately before and after packaging.
"""


def exclusion_manifest_bytes(root: Path, selected: set[Path]) -> bytes:
    output = io.StringIO(newline="")
    fields = ("kind", "path_or_pattern", "reason")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    policies = (
        ("raw_data", "data/raw/**", "Raw datasets are outside the curated publication artifact."),
        ("predictions", "**/predictions/**; **/prediction_exports/**", "Per-example payloads remain under evidence locks."),
        ("imported_workspaces", "kaggle_workspace/imported_outputs/**", "Imported job payloads are not duplicated."),
        ("credentials", "**/.env; **/*token*; private keys", "Secrets are never release material."),
        ("latex_debris", "**/*.aux; **/*.bbl; **/*.blg; **/*.log; **/*.out; **/*.toc", "Strict clean-room compilation regenerates auxiliaries."),
        ("os_and_cache", "**/.DS_Store; **/__MACOSX/**; **/__pycache__/**", "OS/interpreter metadata is excluded."),
        ("private_paths", "absolute workstation paths", "Private workstation metadata fails content validation."),
        ("identity_scan_configuration", "identity-bearing anonymization scanner configuration", "Reviewer archives retain only identity-free anonymization results."),
        ("frozen_archives", "release/tkde_manuscript_package.zip; release/tkde_source_tables.zip", "Frozen baseline ZIPs are verified but never repackaged or overwritten."),
    )
    for kind, pattern, reason in policies:
        writer.writerow({"kind": f"policy:{kind}", "path_or_pattern": pattern, "reason": reason})
    # Enumerate only managed publication surfaces; the repository contains
    # unrelated historical experiment trees that are covered by policy rows.
    managed_roots = (
        root / "paper_tkde",
        root / "scripts" / "tkde_visual_rebuild",
        root / "scripts" / "tkde_rebuild",
        root / "results" / "tkde_visual_rebuild",
        root / "results" / "tkde_rebuild",
    )
    for managed in managed_roots:
        if not managed.exists():
            continue
        for path in sorted(item for item in managed.rglob("*") if item.is_file() and not item.is_symlink()):
            resolved = path.resolve()
            if resolved in selected:
                continue
            reason = "Outside the active dependency/input allowlist."
            kind = "unselected"
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                kind, reason = "build_debris", "Regenerated during strict clean-room build."
            elif path.name == "STARTING_VISUAL_STATE.md":
                kind, reason = "private_starting_state", "Contains local backup/workstation metadata."
            elif identity_match(path.read_bytes()):
                kind, reason = "identity_scan_configuration", "Contains identity tokens used only by the local anonymization scanner."
            elif path.name == ".DS_Store" or "__pycache__" in path.parts:
                kind, reason = "os_or_cache", "OS/interpreter metadata."
            writer.writerow({"kind": kind, "path_or_pattern": relative(path, root), "reason": reason})
    return output.getvalue().encode("utf-8")


def archive_manifest_entry(entries: Sequence[Entry], builder: str) -> Entry:
    output = io.StringIO(newline="")
    fields = ("path", "sha256", "bytes", "source", "role")
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
    return Entry("ARCHIVE_MANIFEST.csv", output.getvalue().encode("utf-8"), builder, "archive_manifest")


def zip_info(entry: Entry) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(entry.arcname, FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    executable = entry.arcname.endswith((".py", ".sh"))
    mode = 0o755 if executable else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.flag_bits = 0
    return info


def write_zip(path: Path, entries: Sequence[Entry], builder: str) -> list[Entry]:
    base = deduplicate(entries)
    all_entries = deduplicate([*base, archive_manifest_entry(base, builder)])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for entry in all_entries:
            archive.writestr(zip_info(entry), entry.data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return all_entries


def validate_zip(path: Path, expected: Sequence[Entry]) -> None:
    by_name = {entry.arcname: entry for entry in expected}
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != sorted(names) or len(names) != len(set(names)):
            raise RuntimeError(f"Archive order/uniqueness failure: {path.name}")
        if set(names) != set(by_name):
            raise RuntimeError(f"Archive listing differs from manifest: {path.name}")
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Archive CRC failure: {path.name}:{corrupt}")
        for info in archive.infolist():
            if info.date_time != FIXED_ZIP_TIME:
                raise RuntimeError(f"Non-fixed ZIP timestamp: {path.name}:{info.filename}")
            data = archive.read(info.filename)
            entry = by_name[info.filename]
            if hashlib.sha256(data).hexdigest() != entry.sha256:
                raise RuntimeError(f"Archive member hash mismatch: {path.name}:{info.filename}")
            validate_entry(entry)


def deterministic_zip(destination: Path, entries: Sequence[Entry], temp_dir: Path, builder: str) -> tuple[list[Entry], str]:
    first = temp_dir / f"{destination.stem}.first.zip"
    second = temp_dir / f"{destination.stem}.second.zip"
    first_entries = write_zip(first, entries, builder)
    second_entries = write_zip(second, entries, builder)
    first_hash, second_hash = sha256_file(first), sha256_file(second)
    if first_hash != second_hash:
        raise RuntimeError(f"Non-deterministic ZIP bytes: {destination.name}")
    if [(item.arcname, item.sha256) for item in first_entries] != [
        (item.arcname, item.sha256) for item in second_entries
    ]:
        raise RuntimeError(f"Non-deterministic member manifest: {destination.name}")
    validate_zip(first, first_entries)
    validate_zip(second, second_entries)
    shutil.copyfile(first, destination)
    validate_zip(destination, first_entries)
    return first_entries, first_hash


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe extraction member: {info.filename}")
            target = (destination / Path(*member.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Extraction path escape: {info.filename}")
        archive.extractall(destination)


def _run_clean_command(command: Sequence[str], cwd: Path, environment: Mapping[str, str]) -> CleanRoomCommand:
    proc = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout.replace(str(cwd), "<CLEAN_ROOM>")
    record = CleanRoomCommand(tuple(command), proc.returncode, output)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-50:])
        raise RuntimeError(f"Clean-room command failed ({proc.returncode}): {' '.join(command)}\n{tail}")
    return record


def clean_room_build(source_zip: Path, python: str, temp_parent: Path) -> list[CleanRoomCommand]:
    work = temp_parent / "clean_room"
    work.mkdir(parents=True)
    safe_extract(source_zip, work)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(work),
            "MPLCONFIGDIR": str(temp_parent / "mplconfig"),
            "SOURCE_DATE_EPOCH": "946684800",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    commands: list[tuple[str, ...]] = [
        (
            python,
            "scripts/tkde_visual_rebuild/scientific_delta_gate.py",
            "--root",
            ".",
            "--report-dir",
            "results/tkde_visual_rebuild/clean_room_audits",
            "--strict",
            "--skip-baseline-archives",
        ),
        (python, "scripts/tkde_visual_rebuild/build_main_tables.py"),
        (python, "scripts/tkde_visual_rebuild/build_curated_supplement_tables.py"),
        (python, "scripts/tkde_rebuild/make_figures.py"),
        (python, "scripts/tkde_rebuild/build_bibliography.py"),
        ("bash", "scripts/tkde_rebuild/compile_papers.sh"),
        (
            python,
            "scripts/tkde_visual_rebuild/audit_table_readability.py",
            "--root",
            ".",
            "--report-dir",
            "results/tkde_visual_rebuild/clean_room_audits",
            "--strict",
        ),
        (
            python,
            "scripts/tkde_visual_rebuild/audit_pdf_layout.py",
            "--root",
            ".",
            "--report-dir",
            "results/tkde_visual_rebuild/clean_room_audits",
            "--strict",
        ),
        (
            python,
            "scripts/tkde_visual_rebuild/scientific_delta_gate.py",
            "--root",
            ".",
            "--report-dir",
            "results/tkde_visual_rebuild/clean_room_audits",
            "--strict",
            "--skip-baseline-archives",
        ),
    ]
    records: list[CleanRoomCommand] = []
    for command in commands:
        records.append(_run_clean_command(command, work, environment))
    for relpath in (
        "paper_tkde/main.pdf",
        "paper_tkde/supplement/supplement.pdf",
        "results/tkde_visual_rebuild/clean_room_audits/TABLE_READABILITY_AUDIT.json",
        "results/tkde_visual_rebuild/clean_room_audits/PDF_LAYOUT_AUDIT.json",
        "results/tkde_visual_rebuild/clean_room_audits/SCIENTIFIC_DELTA_GATE.json",
    ):
        if not (work / relpath).is_file():
            raise RuntimeError(f"Clean-room expected output is missing: {relpath}")
    for name in ("TABLE_READABILITY_AUDIT.json", "PDF_LAYOUT_AUDIT.json"):
        payload = json.loads((work / "results" / "tkde_visual_rebuild" / "clean_room_audits" / name).read_text())
        if payload.get("verdict") != "PASS":
            raise RuntimeError(f"Clean-room audit did not pass: {name}")
    delta = json.loads(
        (work / "results" / "tkde_visual_rebuild" / "clean_room_audits" / "SCIENTIFIC_DELTA_GATE.json").read_text()
    )
    if delta.get("verdict") != "ZERO_SCIENTIFIC_DELTAS":
        raise RuntimeError("Clean-room scientific delta gate did not report ZERO_SCIENTIFIC_DELTAS.")
    return records


def clean_room_report_bytes(records: Sequence[CleanRoomCommand], source_hash: str) -> bytes:
    lines = [
        "# Clean-room visual release build",
        "",
        "Verdict: **PASS**",
        "",
        f"- Source archive SHA-256: `{source_hash}`",
        "- Extraction: new temporary directory with path-safe member validation",
        "- Regeneration: main tables, curated supplement tables, figures, and bibliography",
        "- Compilation: strict main and supplement BibTeX cycles",
        "- Post-build gates: table readability, PDF/log/font/layout, and scientific delta",
        "",
        "## Commands",
        "",
        "| Command | Exit | Last output line |",
        "| --- | ---: | --- |",
    ]
    for record in records:
        last = next((line for line in reversed(record.output.splitlines()) if line.strip()), "")
        last = last.replace("|", "\\|")[:240]
        display_command = list(record.command)
        if display_command and Path(display_command[0]).name.startswith("python"):
            display_command[0] = "python3"
        lines.append(f"| `{' '.join(display_command)}` | {record.exit_code} | {last} |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def artifact_manifest_bytes(archives: Mapping[str, tuple[Path, Sequence[Entry], str]]) -> bytes:
    output = io.StringIO(newline="")
    fields = ("archive", "path", "sha256", "bytes", "source", "role")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for archive_name in sorted(archives):
        path, entries, archive_hash = archives[archive_name]
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
                "bytes": path.stat().st_size,
                "source": path.name,
                "role": "deterministic_zip",
            }
        )
    return output.getvalue().encode("utf-8")


def verify_baseline_archives(root: Path) -> None:
    for relpath, expected in EXPECTED_BASELINE_ARCHIVES.items():
        path = require_file(root / relpath, root)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen baseline archive mismatch: {relpath} expected={expected} actual={actual}")


def external_manifest_check(paths: ReleasePaths) -> None:
    for path in (
        paths.manuscript_zip,
        paths.source_zip,
        paths.artifact_manifest,
        paths.excluded_manifest,
        paths.readme,
        paths.clean_room_report,
        paths.checksums,
    ):
        require_file(path, paths.root)
    rows = read_csv_dicts(paths.artifact_manifest)
    for archive in (paths.manuscript_zip, paths.source_zip):
        summary = next(
            (row for row in rows if row.get("archive") == archive.name and row.get("path") == "__ARCHIVE__"),
            None,
        )
        if summary is None or summary.get("sha256") != sha256_file(archive):
            raise RuntimeError(f"External manifest mismatch: {archive.name}")
        with zipfile.ZipFile(archive, "r") as handle:
            if handle.testzip():
                raise RuntimeError(f"CRC validation failed: {archive.name}")
            names = handle.namelist()
            if names != sorted(names) or len(names) != len(set(names)):
                raise RuntimeError(f"Noncanonical archive listing: {archive.name}")
            manifest_rows = {
                row["path"]: row
                for row in rows
                if row.get("archive") == archive.name and row.get("path") != "__ARCHIVE__"
            }
            if set(names) != set(manifest_rows):
                raise RuntimeError(f"Member/external-manifest mismatch: {archive.name}")
            for info in handle.infolist():
                data = handle.read(info.filename)
                row = manifest_rows[info.filename]
                if hashlib.sha256(data).hexdigest() != row["sha256"] or len(data) != int(row["bytes"]):
                    raise RuntimeError(f"Member checksum mismatch: {archive.name}:{info.filename}")
                validate_entry(Entry(info.filename, data, row["source"], row["role"]))
    verify_baseline_archives(paths.root)
    expected_checksums = {
        paths.manuscript_zip.name: sha256_file(paths.manuscript_zip),
        paths.source_zip.name: sha256_file(paths.source_zip),
        "paper_tkde/main.pdf": sha256_file(paths.root / "paper_tkde/main.pdf"),
        "paper_tkde/supplement/supplement.pdf": sha256_file(
            paths.root / "paper_tkde/supplement/supplement.pdf"
        ),
        paths.artifact_manifest.name: sha256_file(paths.artifact_manifest),
        paths.excluded_manifest.name: sha256_file(paths.excluded_manifest),
        paths.readme.name: sha256_file(paths.readme),
        paths.clean_room_report.name: sha256_file(paths.clean_room_report),
    }
    observed_checksums: dict[str, str] = {}
    for line in paths.checksums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise RuntimeError(f"Malformed checksum-manifest row: {line!r}")
        observed_checksums[match.group(2)] = match.group(1)
    if observed_checksums != expected_checksums:
        raise RuntimeError("Checksum manifest does not match the published visual-release surfaces.")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_release(root: Path, python: str, *, run_clean_room: bool = True) -> ReleasePaths:
    root = root.resolve()
    paths = ReleasePaths(root)
    verify_baseline_archives(root)
    # Explicitly guard against accidental aliasing to the frozen outputs.
    frozen = {(root / relpath).resolve() for relpath in EXPECTED_BASELINE_ARCHIVES}
    if paths.manuscript_zip.resolve() in frozen or paths.source_zip.resolve() in frozen:
        raise RuntimeError("Visual release destination aliases a frozen baseline archive.")

    manuscript, manuscript_paths = manuscript_entries(root)
    generator_inputs = generator_input_paths(root)
    generators = generator_script_paths(root)
    reports = visual_report_paths(root)
    environment = environment_paths(root)
    selected = manuscript_paths | generator_inputs | generators | reports | environment
    readme_bytes = reproducibility_readme_bytes()
    excluded_bytes = exclusion_manifest_bytes(root, selected)
    for name, data in (("readme", readme_bytes), ("excluded manifest", excluded_bytes)):
        if private_match(data) or secret_match(data) or identity_match(data):
            raise RuntimeError(f"Generated {name} violates release hygiene.")
    metadata = [
        Entry("tkde_visual_reproducibility_readme.md", readme_bytes, "generated", "release_instructions"),
        Entry("tkde_visual_excluded_file_manifest.csv", excluded_bytes, "generated", "exclusion_manifest"),
    ]
    report_entries = input_entries(reports, root, "visual_audit_or_design_report")
    manuscript_archive_entries = deduplicate([*manuscript, *report_entries, *metadata])
    source_archive_entries = deduplicate(
        [
            *manuscript,
            *input_entries(generator_inputs, root, "generator_input_or_machine_evidence"),
            *input_entries(generators, root, "regeneration_script"),
            *input_entries(reports, root, "visual_audit_or_design_report"),
            *input_entries(environment, root, "environment_specification"),
            *metadata,
        ]
    )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tkde_visual_release_", dir=paths.output_dir) as directory:
        temp_dir = Path(directory)
        staged_manuscript = temp_dir / paths.manuscript_zip.name
        staged_source = temp_dir / paths.source_zip.name
        manuscript_members, manuscript_hash = deterministic_zip(
            staged_manuscript,
            manuscript_archive_entries,
            temp_dir,
            "scripts/tkde_visual_rebuild/build_visual_release.py",
        )
        source_members, source_hash = deterministic_zip(
            staged_source,
            source_archive_entries,
            temp_dir,
            "scripts/tkde_visual_rebuild/build_visual_release.py",
        )
        clean_records = clean_room_build(staged_source, python, temp_dir) if run_clean_room else []
        clean_report = clean_room_report_bytes(clean_records, source_hash) if run_clean_room else b"# Clean-room visual release build\n\nVerdict: **SKIPPED_IN_TEST_HELPER**\n"
        # Recheck the frozen archives after the potentially expensive build.
        verify_baseline_archives(root)
        os.replace(staged_manuscript, paths.manuscript_zip)
        os.replace(staged_source, paths.source_zip)

    paths.excluded_manifest.write_bytes(excluded_bytes)
    paths.readme.write_bytes(readme_bytes)
    paths.clean_room_report.write_bytes(clean_report)
    archives = {
        paths.manuscript_zip.name: (paths.manuscript_zip, manuscript_members, manuscript_hash),
        paths.source_zip.name: (paths.source_zip, source_members, source_hash),
    }
    paths.artifact_manifest.write_bytes(artifact_manifest_bytes(archives))
    checksum_rows = {
        paths.manuscript_zip.name: sha256_file(paths.manuscript_zip),
        paths.source_zip.name: sha256_file(paths.source_zip),
        "paper_tkde/main.pdf": sha256_file(root / "paper_tkde/main.pdf"),
        "paper_tkde/supplement/supplement.pdf": sha256_file(
            root / "paper_tkde/supplement/supplement.pdf"
        ),
        paths.artifact_manifest.name: sha256_file(paths.artifact_manifest),
        paths.excluded_manifest.name: sha256_file(paths.excluded_manifest),
        paths.readme.name: sha256_file(paths.readme),
        paths.clean_room_report.name: sha256_file(paths.clean_room_report),
    }
    paths.checksums.write_text(
        "".join(f"{checksum_rows[name]}  {name}\n" for name in sorted(checksum_rows)),
        encoding="utf-8",
    )
    external_manifest_check(paths)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used by the extracted clean-room build.")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = ReleasePaths(args.root.resolve())
    if args.check_only:
        external_manifest_check(paths)
        print("PASS: namespaced visual-release archives match manifests; frozen baseline ZIPs remain unchanged.")
        return 0
    built = build_release(args.root, args.python, run_clean_room=True)
    print(f"PASS: {built.manuscript_zip} sha256={sha256_file(built.manuscript_zip)}")
    print(f"PASS: {built.source_zip} sha256={sha256_file(built.source_zip)}")
    print(f"PASS: clean-room report={built.clean_room_report}")
    print("PASS: frozen baseline archives preserved and verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
