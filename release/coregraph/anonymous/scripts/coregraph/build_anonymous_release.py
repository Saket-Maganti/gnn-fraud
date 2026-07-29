#!/usr/bin/env python3
"""Build a deterministic anonymous source package without scientific outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "release/coregraph/anonymous"
INCLUDE = (
    "coregraph",
    "models",
    "docs/coregraph",
    "configs/coregraph",
    "external_baselines",
    "paper_iclr",
    "scripts/coregraph",
    "runbooks/coregraph",
    "kaggle/coregraph",
    "notebooks/coregraph",
    "tests/coregraph",
)
ROOT_FILES = (
    "pyproject.toml",
    "requirements-coregraph-lock.txt",
    "Makefile",
)
SPECIFICATION_FILES = (
    "results/coregraph_build/PILOT_GATE_FROZEN_SPEC.json",
    "results/coregraph_build/PILOT_V3_SPECIFICATION.md",
    "results/coregraph_build/PILOT_GATE_FROZEN_SPEC_V4.json",
    "results/coregraph_build/PILOT_V4_SPECIFICATION.md",
    "results/coregraph_build/CONTRACT_PROTOCOL_REGISTRY.schema.json",
    "results/coregraph_build/CONTRACT_PROTOCOL_REGISTRY_V4.json",
)
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "anonymous",
}
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".parquet",
    ".npz",
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".synctex.gz",
}


def include_file(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.name.endswith(".synctex.gz"):
        return False
    if path.name == "main.pdf" and "paper_iclr" in path.parts:
        return False
    return path.suffix.lower() not in EXCLUDE_SUFFIXES


def main() -> int:
    if DESTINATION.exists():
        # The target is an explicit generated-artifact directory, never user data.
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)
    for relative in INCLUDE:
        source = ROOT / relative
        if not source.exists():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file() and include_file(path):
                target = DESTINATION / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for relative in ROOT_FILES:
        source = ROOT / relative
        if source.exists():
            shutil.copy2(source, DESTINATION / relative)
    specification_dir = DESTINATION / "specifications"
    specification_dir.mkdir()
    for relative in SPECIFICATION_FILES:
        source = ROOT / relative
        shutil.copy2(source, specification_dir / source.name)
    files = {}
    for path in sorted(DESTINATION.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(DESTINATION))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": "coregraph_anonymous_release_v1",
        "aggregate_sha256": aggregate,
        "files": files,
        "scientific_results_included": False,
    }
    (DESTINATION / "ANONYMOUS_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(files), "aggregate_sha256": aggregate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
