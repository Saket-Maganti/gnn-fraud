#!/usr/bin/env python3
"""Build deterministic source-inventory and large-file publication audits.

This script is intentionally read-only outside ``results/github_publish``.
It inventories the active research tree without following symlinks and without
opening raw datasets or prediction payloads.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024

IMPORTANT_ROOTS = {
    "_rb_refresh_archive_2026-06-20",
    "aaai_upgrade",
    "artifacts",
    "data",
    "docs",
    "experiments",
    "figures",
    "fraudshiftbench",
    "fraudshiftbench_paper",
    "freezes",
    "gnn-fraud_tkde_rebuild_backup_20260710_110548_IST",
    "gnn-fraud_tkde_visual_backup_20260710_224421_IST",
    "gnn_env",
    "gnnpaper",
    "gpu_runbooks_package",
    "journal_kbs_tkde_package",
    "kaggle",
    "kaggle_workspace",
    "kaggleresults",
    "manuscript_assets",
    "models",
    "notebooks",
    "output",
    "paper",
    "paper_tkde",
    "promptpacks",
    "release",
    "results",
    "runs_expansion",
    "scripts",
    "tests",
    "tmp",
    "top_tier_upgrade",
    "utils",
    "v24_evidence_package",
    "v25_evidence_package",
}

EXCLUDE_PREFIXES = (
    ".claude",
    ".cursor",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "_rb_refresh_archive_2026-06-20",
    "gnn-fraud_tkde_rebuild_backup_",
    "gnn-fraud_tkde_visual_backup_",
    "gnn_env",
    "kaggle_workspace/download",
    "kaggle_workspace/upload_bundles",
    "kaggle_workspace/datasets",
    "kaggleresults",
    "promptpacks",
    "tmp",
)

MANIFEST_ONLY_PREFIXES = (
    "artifacts",
    "freezes",
    "gpu_runbooks_package",
    "journal_kbs_tkde_package",
    "release",
    "results",
    "v24_evidence_package",
    "v25_evidence_package",
)

CANONICAL_PREFIXES = (
    ".github",
    "configs",
    "data",
    "docs",
    "examples",
    "experiments",
    "fraudshiftbench",
    "models",
    "notebooks",
    "output/pdf",
    "paper_tkde",
    "scripts",
    "tests",
    "utils",
)

RAW_OR_PREDICTION_PARTS = {
    "data/raw",
    "predictions",
    "prediction_exports",
    "raw_predictions",
}


@dataclass
class TreeStats:
    size_bytes: int
    file_count: int
    largest_files: list[tuple[int, str]]
    private_path_risk: bool
    credential_filename_risk: bool


def _path_starts(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _classify(path: str, is_dir: bool) -> dict[str, str]:
    lower = path.lower()
    name = Path(path).name
    if (
        _path_starts(path, EXCLUDE_PREFIXES)
        or name in {".DS_Store", "__MACOSX"}
        or "__pycache__" in Path(path).parts
        or ".ipynb_checkpoints" in Path(path).parts
        or "backup_" in lower
        or lower.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    ):
        category = "historical/local-only/environment/generated"
        include = "no"
        manifest = "no"
        exclude = "yes"
        purpose = "Local environment, cache, archive, backup, or duplicate package."
        justification = "Explicitly excluded by the governing public-repository safety rules."
        risk = "high"
    elif any(part in lower for part in RAW_OR_PREDICTION_PARTS):
        category = "raw data or raw predictions"
        include = "no"
        manifest = "yes"
        exclude = "yes"
        purpose = "Dataset or prediction-bearing working material."
        justification = "Raw data and raw prediction payloads must not be published; retain only safe manifests."
        risk = "critical"
    elif _path_starts(path, MANIFEST_ONLY_PREFIXES):
        category = "generated evidence/release output"
        include = "allowlist only"
        manifest = "yes"
        exclude = "partial"
        purpose = "Evidence, reports, release metadata, or generated publication outputs."
        justification = "Include only frozen aggregates, manifests, readiness reports, and final PDFs."
        risk = "medium"
    elif _path_starts(path, CANONICAL_PREFIXES):
        category = "canonical candidate"
        include = "allowlist only"
        manifest = "no"
        exclude = "partial"
        purpose = "Candidate source, tests, documentation, manuscript, or lightweight configuration."
        justification = "Audit active imports, final manifests, and clean-room commands before inclusion."
        risk = "medium"
    elif is_dir:
        category = "unknown or historical candidate"
        include = "no by default"
        manifest = "possible"
        exclude = "yes pending proof"
        purpose = "Project surface not part of the final dependency closure by name alone."
        justification = "The allowlist is explicit; inclusion requires canonical-source evidence."
        risk = "medium"
    else:
        category = "root-level candidate or report"
        include = "allowlist only"
        manifest = "possible"
        exclude = "partial"
        purpose = "Root configuration, documentation, script, report, or archive."
        justification = "Include only current portable configuration and required final documentation."
        risk = "medium"
    return {
        "purpose": purpose,
        "classification": category,
        "include_in_git": include,
        "manifest_only": manifest,
        "exclude": exclude,
        "justification": justification,
        "risk_level": risk,
    }


def _iter_files(root: Path, path: Path):
    if path.is_symlink():
        return
    if path.is_file():
        yield path
        return
    for current, dirnames, filenames in os.walk(path, followlinks=False):
        rel_current = Path(current).relative_to(root)
        if rel_current.parts and rel_current.parts[0] == ".git":
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not (Path(current) / name).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = Path(current) / filename
            if not candidate.is_symlink():
                yield candidate


def _stats(root: Path, path: Path) -> TreeStats:
    total = 0
    count = 0
    largest: list[tuple[int, str]] = []
    private = False
    credential = False
    credential_names = {
        ".env",
        ".netrc",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "kaggle.json",
    }
    for file_path in _iter_files(root, path) or []:
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        total += size
        count += 1
        rel = file_path.relative_to(root).as_posix()
        largest.append((size, rel))
        private = private or ("/" + "Users/") in rel or ("/" + "Volumes/") in rel
        credential = credential or file_path.name in credential_names or file_path.suffix in {".pem", ".key"}
    largest.sort(reverse=True)
    return TreeStats(total, count, largest[:5], private, credential)


def _inventory_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        paths.append(path)
        if path.is_dir() and path.name in IMPORTANT_ROOTS:
            try:
                children = sorted(path.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            paths.extend(children)
    return paths


def _paper_relationship(path: str) -> str:
    if _path_starts(path, ("paper_tkde", "output/pdf", "results/tkde_rebuild", "results/tkde_visual_rebuild")):
        return "Direct final TKDE source, PDF, evidence, or validation relationship."
    if _path_starts(path, ("fraudshiftbench", "scripts/tkde_rebuild", "scripts/tkde_visual_rebuild")):
        return "Canonical benchmark/framework or paper regeneration implementation."
    if _path_starts(path, ("models", "experiments", "data", "utils")):
        return "Scientific implementation surface; include only current canonical modules."
    return "Indirect, historical, local-only, or no required final-paper relationship."


def _future_relationship(path: str) -> str:
    if _path_starts(path, ("fraudshiftbench", "models", "experiments", "data", "utils", "tests")):
        return "Potential reusable infrastructure; evidence status remains contract-bound."
    if _path_starts(path, ("results/tkde_rebuild", "paper_tkde")):
        return "Reusable baseline/evidence boundary, not automatic method novelty."
    return "Not a preferred ICLR-method development surface."


def build_inventory(root: Path, output_dir: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for path in _inventory_paths(root):
        rel = path.relative_to(root).as_posix()
        values = _classify(rel, path.is_dir())
        stats = _stats(root, path)
        largest = "; ".join(f"{name} ({size} B)" for size, name in stats.largest_files)
        rows.append(
            {
                "path": rel,
                "size_bytes": stats.size_bytes,
                "file_count": stats.file_count,
                **values,
                "largest_files": largest,
                "private_path_or_mount_name_in_path": str(stats.private_path_risk),
                "credential_like_filename_present": str(stats.credential_filename_risk),
                "final_paper_relationship": _paper_relationship(rel),
                "future_iclr_relationship": _future_relationship(rel),
            }
        )

    csv_path = output_dir / "GITHUB_SOURCE_INVENTORY.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(path.stat().st_size for path in _iter_files(root, root) or [])
    md_path = output_dir / "GITHUB_SOURCE_INVENTORY.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# GitHub source inventory\n\n")
        handle.write(
            "This inventory records every top-level entry and the immediate children of the "
            "governing prompt's important project surfaces. Classification is conservative: "
            "candidate directories remain allowlist-only until the canonical-source map selects "
            "specific paths. The local `.git` directory is inventoried as a top-level local-only "
            "surface but its object database is not traversed.\n\n"
        )
        handle.write(f"- Logical file bytes outside `.git`: {total_bytes:,}\n")
        handle.write(f"- Inventory rows: {len(rows):,}\n")
        handle.write("- Raw data, predictions, environments, caches, backups, prompt packs, and duplicate archives: excluded.\n")
        handle.write("- Frozen aggregates, final PDFs, manifests, and readiness reports: manifest/allowlist review.\n\n")
        handle.write(
            "| Path | Size (MiB) | Files | Classification | Git treatment | Risk | Largest files |\n"
            "| --- | ---: | ---: | --- | --- | --- | --- |\n"
        )
        for row in rows:
            largest = str(row["largest_files"]).replace("|", "\\|")
            treatment = (
                f"include={row['include_in_git']}; manifest={row['manifest_only']}; "
                f"exclude={row['exclude']}"
            )
            handle.write(
                f"| `{row['path']}` | {int(row['size_bytes']) / MIB:.2f} | "
                f"{row['file_count']} | {row['classification']} | {treatment} | "
                f"{row['risk_level']} | {largest} |\n"
            )


def build_large_file_audit(root: Path, output_dir: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for path in _iter_files(root, root) or []:
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 10 * MIB:
            continue
        if size > 100 * MIB:
            band = ">100 MiB"
        elif size > 50 * MIB:
            band = ">50 MiB"
        elif size > 25 * MIB:
            band = ">25 MiB"
        else:
            band = ">10 MiB"
        lower = rel.lower()
        if rel == "results/tkde_rebuild/NUMBER_PROVENANCE_MAP.csv":
            decision = "include normal Git"
            rationale = "Frozen 6,796-record scalar provenance map; below 25 MiB."
        elif any(token in lower for token in ("gnn_env/", "data/raw/", "kaggle_workspace/", "kaggleresults/", "backup_", ".claude/")):
            decision = "exclude"
            rationale = "Environment, raw data, workspace output, backup, or private worktree."
        elif "prediction" in lower or "node_level" in lower or "harm_predictor_features" in lower:
            decision = "replace by manifest/checksum"
            rationale = "Row-level or prediction-adjacent payload is not needed in the public source tree."
        else:
            decision = "exclude pending explicit review"
            rationale = "Not selected by the final dependency allowlist."
        rows.append(
            {
                "path": rel,
                "size_bytes": size,
                "size_mib": f"{size / MIB:.3f}",
                "threshold_band": band,
                "decision": decision,
                "rationale": rationale,
            }
        )
    rows.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    output = output_dir / "LARGE_FILE_AUDIT.csv"
    fields = ["path", "size_bytes", "size_mib", "threshold_band", "decision", "rationale"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/github_publish"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    build_inventory(root, output_dir)
    build_large_file_audit(root, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
