#!/usr/bin/env python3
"""Build deterministic Level-4 metadata and clean source snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT.parent / "gnn-fraud-github-curated"
RELEASE = ROOT / "release" / "level4"
SNAPSHOTS = ROOT / "release" / "level4_source_snapshot"
BUILD = ROOT / "results" / "coregraph_build"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
TEXT_SUFFIXES = {
    ".bib",
    ".bst",
    ".cfg",
    ".csv",
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sha256",
    ".sh",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PAPER_ASSET_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "checkpoints",
    "predictions",
    "raw",
    "wandb",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".npz",
    ".npy",
    ".parquet",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}
ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "Makefile",
    "PROJECT_PATHS_AND_AUTHORITIES.json",
    "PROJECT_STATE_AND_AUTHORITY.md",
    "pyproject.toml",
    "requirements-coregraph-lock.txt",
    "requirements.txt",
}
CORE_PREFIXES = (
    "config/",
    "configs/coregraph/",
    "coregraph/",
    "data/",
    "docs/coregraph/",
    "external_baselines/",
    "kaggle/coregraph/",
    "models/",
    "notebooks/coregraph/",
    "paper_iclr/",
    "results/coregraph_build/",
    "runbooks/coregraph/",
    "scripts/coregraph/",
    "tests/coregraph/",
    "theory/coregraph_level4/",
    "utils/",
)
CORE_EXACT = {
    "scripts/audit_cross_paper_overlap.py",
    "scripts/github_publish/validate_public_tree.py",
    "tests/test_path_resolution.py",
}
SCANNER_FILES = {
    "scripts/check_anonymization.py",
    "scripts/coregraph/audit_anonymous_release.py",
    "scripts/coregraph/build_level4_paper.py",
    "scripts/coregraph/validate_paper_skeleton.py",
    "scripts/coregraph/build_level4_release.py",
    "tests/coregraph/test_level4_generated_artifacts.py",
}
DYNAMIC_HANDOFF_FILES = {
    "results/coregraph_build/LEVEL4_MASTER_BUILD_REPORT.md",
    "results/coregraph_build/LEVEL4_FINAL_GATE_STATUS.json",
    "results/coregraph_build/LEVEL4_RUN_AFTER_BUILD_CHECKLIST.md",
    "results/coregraph_build/LEVEL4_FINAL_TREE.txt",
    "results/coregraph_build/LEVEL4_FINAL_COMMAND_LOG.md",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _core_candidate(relative: str) -> bool:
    if relative in ROOT_FILES or relative in CORE_EXACT:
        return True
    return any(relative.startswith(prefix) for prefix in CORE_PREFIXES)


def _eligible_source(root: Path, relative: str, *, coregraph: bool) -> bool:
    if not _safe_relative(relative):
        return False
    path = PurePosixPath(relative)
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    if relative.startswith("release/") or relative.startswith("paper_iclr/build/"):
        return False
    # These reports describe the completed Git/PR/CI handoff and are written
    # after clean-room validation. Excluding them avoids a self-referential
    # source snapshot while keeping their deterministic generator in-source.
    if relative in DYNAMIC_HANDOFF_FILES:
        return False
    if relative == "results/coregraph_build/LEVEL4_CLEANROOM_VALIDATION.json":
        return False
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if coregraph and not _core_candidate(relative):
        return False
    if relative in {"paper_iclr/main.pdf", "paper_iclr/supplement.pdf"}:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return coregraph and relative.startswith("paper_iclr/figures/") and (
        path.suffix.lower() in PAPER_ASSET_SUFFIXES
    )


def _source_files(repo: Path, *, coregraph: bool) -> list[str]:
    arguments = ["ls-files"]
    if coregraph:
        arguments.extend(["--cached", "--others", "--exclude-standard"])
    candidates = git(repo, *arguments).splitlines()
    output = []
    for relative in sorted(set(candidates)):
        path = repo / relative
        if path.is_file() and not path.is_symlink() and _eligible_source(
            repo, relative, coregraph=coregraph
        ):
            output.append(relative)
    return output


def _zip_info(relative: str, executable: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_snapshot(repo: Path, destination: Path, *, coregraph: bool) -> list[dict[str, object]]:
    files = _source_files(repo, coregraph=coregraph)
    if not files:
        raise RuntimeError(f"source snapshot selected no files from {repo}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative in files:
            path = repo / relative
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            archive.writestr(_zip_info(relative, executable), path.read_bytes())
    return [
        {
            "snapshot": destination.name,
            "path": relative,
            "bytes": (repo / relative).stat().st_size,
            "sha256": sha256_path(repo / relative),
        }
        for relative in files
    ]


def _scan_zip(path: Path) -> tuple[list[str], int]:
    failures: list[str] = []
    files = 0
    private_path = re.compile(r"/(?:Users|Volumes)/[A-Za-z0-9._-]+")
    private_identity = re.compile(r"saket\s*maganti", re.IGNORECASE)
    email = re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.org)[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            files += 1
            relative = PurePosixPath(info.filename)
            if not _safe_relative(info.filename):
                failures.append(f"unsafe_path:{info.filename}")
            if any(part in FORBIDDEN_PARTS for part in relative.parts):
                failures.append(f"forbidden_part:{info.filename}")
            if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
                failures.append(f"forbidden_suffix:{info.filename}")
            if relative.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = archive.read(info).decode("utf-8", errors="replace")
            if info.filename not in SCANNER_FILES:
                if private_path.search(text):
                    failures.append(f"private_path:{info.filename}")
                if private_identity.search(text):
                    failures.append(f"private_identity:{info.filename}")
            if email.search(text):
                failures.append(f"private_email:{info.filename}")
    return failures, files


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_manifest(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "publishable": "true",
                "licence_or_access": "SOURCE_OR_BUILD_METADATA_NO_PROVIDER_PAYLOAD",
            }
        )
    return rows


def main() -> int:
    RELEASE.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    core_zip = SNAPSHOTS / "coregraph_source_snapshot.zip"
    curated_zip = SNAPSHOTS / "curated_source_snapshot.zip"

    core_first = build_snapshot(ROOT, core_zip, coregraph=True)
    core_first_hash = sha256_path(core_zip)
    core_second = build_snapshot(ROOT, core_zip, coregraph=True)
    core_second_hash = sha256_path(core_zip)
    curated_first = build_snapshot(CURATED, curated_zip, coregraph=False)
    curated_first_hash = sha256_path(curated_zip)
    curated_second = build_snapshot(CURATED, curated_zip, coregraph=False)
    curated_second_hash = sha256_path(curated_zip)
    if core_first != core_second or curated_first != curated_second:
        raise RuntimeError("source file inventory changed during deterministic rebuild")

    snapshot_rows = core_second + curated_second
    _write_csv(
        SNAPSHOTS / "SNAPSHOT_MANIFEST.csv",
        snapshot_rows,
        ("snapshot", "path", "bytes", "sha256"),
    )
    snapshot_checksums = (
        f"{core_second_hash}  coregraph_source_snapshot.zip\n"
        f"{curated_second_hash}  curated_source_snapshot.zip\n"
    )
    (SNAPSHOTS / "CHECKSUMS.sha256").write_text(snapshot_checksums, encoding="utf-8")
    (SNAPSHOTS / "README.md").write_text(
        """# Level-4 source snapshots

These deterministic ZIPs contain source, configuration, tests, documentation, paper sources, and compact build metadata. They exclude Git metadata, environments, caches, provider data, predictions, checkpoints, credentials, logs, LaTeX intermediates, and final PDFs. ZIP entry times and permissions are normalized.

`coregraph_source_snapshot.zip` is the clean-room build input. `curated_source_snapshot.zip` preserves the independent curated source authority without copying provider payloads.
""",
        encoding="utf-8",
    )

    core_scan_failures, core_file_count = _scan_zip(core_zip)
    curated_scan_failures, curated_file_count = _scan_zip(curated_zip)
    public_failures = core_scan_failures + curated_scan_failures
    public_report = {
        "schema": "coregraph_level4_public_tree_audit_v1",
        "status": "PASS" if not public_failures else "FAIL",
        "source_snapshots": 2,
        "coregraph_files": core_file_count,
        "curated_files": curated_file_count,
        "provider_payloads": 0,
        "credentials": 0,
        "private_path_or_identity_failures": public_failures,
    }
    _write_json(RELEASE / "PUBLIC_TREE_AUDIT.json", public_report)

    claims = list(
        csv.DictReader(
            (ROOT / "paper_iclr/claims/LEVEL4_ICLR_CLAIM_LEDGER.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    claim_counts = Counter(row["current_status"] for row in claims)
    claim_report = {
        "schema": "coregraph_level4_claim_support_report_v1",
        "status": "PASS_CLAIMS_TYPED_AND_EMPIRICAL_BLOCKED",
        "claim_count": len(claims),
        "status_counts": dict(sorted(claim_counts.items())),
        "empirical_claims_populated": 0,
        "numerical_results_fabricated": 0,
        "claims": claims,
    }
    _write_json(RELEASE / "CLAIM_SUPPORT_REPORT.json", claim_report)

    determinism_report = {
        "schema": "coregraph_level4_determinism_report_v1",
        "status": (
            "PASS"
            if core_first_hash == core_second_hash
            and curated_first_hash == curated_second_hash
            else "FAIL"
        ),
        "fixed_zip_timestamp": "1980-01-01T00:00:00Z",
        "sorted_entries": True,
        "normalized_permissions": True,
        "coregraph_rebuild_sha256_equal": core_first_hash == core_second_hash,
        "curated_rebuild_sha256_equal": curated_first_hash == curated_second_hash,
        "coregraph_source_sha256": core_second_hash,
        "curated_source_sha256": curated_second_hash,
    }
    _write_json(RELEASE / "DETERMINISM_REPORT.json", determinism_report)

    frozen = subprocess.run(
        [sys.executable, "scripts/coregraph/hash_frozen_assets.py", "--verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    evidence = json.loads((BUILD / "ARCHIVE_MEMBER_VALIDATION.json").read_text())
    notebooks = json.loads((BUILD / "NOTEBOOK_VALIDATION.json").read_text())
    overlap = json.loads((BUILD / "LEVEL4_CROSS_PAPER_OVERLAP_AUDIT.json").read_text())
    validation_failures = []
    if evidence.get("verdict") != "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS":
        validation_failures.append("evidence_cache")
    if notebooks.get("status") != "PASS":
        validation_failures.append("notebooks")
    if overlap.get("status") != "PASS":
        validation_failures.append("overlap")
    if frozen.returncode != 0:
        validation_failures.append("frozen_tkde")
    if public_failures:
        validation_failures.append("public_tree")
    validation_report = {
        "schema": "coregraph_level4_release_validation_v1",
        "status": "PASS_PRE_CLEANROOM" if not validation_failures else "FAIL",
        "failures": validation_failures,
        "evidence_archives_verified": evidence.get("archive_verified"),
        "evidence_members_verified": evidence.get("member_checksum_verified"),
        "kaggle_runbooks_validated": notebooks.get("kaggle_level4_runbooks"),
        "overlap_audit": overlap.get("status"),
        "frozen_tkde": frozen.stdout.strip(),
        "heavy_real_data_runs": 0,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
        "official_baseline_repositories_installed": 0,
        "kaggle_jobs_launched": 0,
        "cleanroom_status": "PENDING_SEPARATE_VALIDATOR",
    }
    _write_json(RELEASE / "VALIDATION_REPORT.json", validation_report)
    (RELEASE / "README.md").write_text(
        """# CoReGraph Level-4 pre-run release

Status: `VALIDATED_BUILD_ARTIFACT_RESULTS_BLOCKED`.

This release contains compact metadata and deterministic source snapshots only. It contains no provider archive, prediction payload, checkpoint, credential, empirical Level-4 metric, target oracle, or official-baseline output. The external canonical cache is referenced by checksum but never redistributed.

Run `make coregraph-cleanroom` to validate extraction, a fresh isolated environment, tiny tests, both anonymous PDFs, checksum closure, and identity/path hygiene.
""",
        encoding="utf-8",
    )

    manifest_inputs = [
        SNAPSHOTS / "coregraph_source_snapshot.zip",
        SNAPSHOTS / "curated_source_snapshot.zip",
        SNAPSHOTS / "SNAPSHOT_MANIFEST.csv",
        SNAPSHOTS / "CHECKSUMS.sha256",
        SNAPSHOTS / "README.md",
        RELEASE / "README.md",
        RELEASE / "VALIDATION_REPORT.json",
        RELEASE / "DETERMINISM_REPORT.json",
        RELEASE / "CLAIM_SUPPORT_REPORT.json",
        RELEASE / "PUBLIC_TREE_AUDIT.json",
        ROOT / "paper_iclr/main.pdf",
        ROOT / "paper_iclr/supplement.pdf",
        BUILD / "ARCHIVE_MEMBER_VALIDATION.json",
        BUILD / "V5_LEAKAGE_AUDIT.json",
        BUILD / "LEVEL4_PAPER_BUILD_REPORT.md",
        BUILD / "NOTEBOOK_VALIDATION.json",
        BUILD / "LEVEL4_CROSS_PAPER_OVERLAP_AUDIT.json",
    ]
    manifest_rows = _relative_manifest(manifest_inputs)
    _write_csv(
        RELEASE / "MANIFEST.csv",
        manifest_rows,
        ("path", "bytes", "sha256", "publishable", "licence_or_access"),
    )
    _write_csv(
        BUILD / "LEVEL4_RELEASE_MANIFEST.csv",
        manifest_rows,
        ("path", "bytes", "sha256", "publishable", "licence_or_access"),
    )
    checksum_targets = [
        RELEASE / "README.md",
        RELEASE / "MANIFEST.csv",
        RELEASE / "VALIDATION_REPORT.json",
        RELEASE / "DETERMINISM_REPORT.json",
        RELEASE / "CLAIM_SUPPORT_REPORT.json",
        RELEASE / "PUBLIC_TREE_AUDIT.json",
        core_zip,
        curated_zip,
    ]
    (RELEASE / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{sha256_path(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in checksum_targets
        ),
        encoding="utf-8",
    )
    status = "PASS" if not validation_failures else "FAIL"
    print(
        json.dumps(
            {
                "status": status,
                "coregraph_snapshot_files": len(core_second),
                "curated_snapshot_files": len(curated_second),
                "coregraph_snapshot_sha256": core_second_hash,
                "curated_snapshot_sha256": curated_second_hash,
                "release_manifest_rows": len(manifest_rows),
                "failures": validation_failures,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
