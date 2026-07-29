#!/usr/bin/env python3
"""Copy only audited allowlist files into an existing clean Git clone.

The destination must already be a clone on a non-default branch. Existing
remote files are preserved unless they are explicit operating-system metadata.
All recorded mappings are repository-relative.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


HARD_FORBIDDEN_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    "__MACOSX",
    "__pycache__",
    "gnn_env",
}

HARD_FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".netrc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "kaggle.json",
}

HARD_FORBIDDEN_SUFFIXES = {
    ".7z",
    ".key",
    ".npz",
    ".npy",
    ".parquet",
    ".pem",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}

SPECIFIC_EXCLUSIONS = {
    "paper_tkde/main.pdf",
    "paper_tkde/supplement/supplement.pdf",
    "results/tkde_rebuild/STARTING_STATE.md",
    "results/tkde_visual_rebuild/STARTING_VISUAL_STATE.md",
    "results/tkde_visual_rebuild/validation/anonymization.json",
}

DESTINATION_MAP = {
    "output/pdf/FraudShiftBench_TKDE_main.pdf": "paper/pdf/FraudShiftBench_TKDE_main.pdf",
    "output/pdf/FraudShiftBench_TKDE_supplement.pdf": "paper/pdf/FraudShiftBench_TKDE_supplement.pdf",
}


def command_output(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def parse_allowlist(path: Path) -> list[str]:
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def safe_source(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in SPECIFIC_EXCLUSIONS:
        return False
    if path.is_symlink() or not path.is_file():
        return False
    if any(part in HARD_FORBIDDEN_PARTS for part in path.relative_to(root).parts):
        return False
    if path.name in HARD_FORBIDDEN_NAMES:
        return False
    lower = path.name.lower()
    if any(lower.endswith(suffix) for suffix in HARD_FORBIDDEN_SUFFIXES):
        return False
    if rel.startswith("data/raw/"):
        return False
    if path.stat().st_size > 100 * 1024 * 1024:
        raise RuntimeError(f"Allowlisted file exceeds 100 MiB: {rel}")
    return True


def expand(root: Path, patterns: list[str]) -> list[Path]:
    selected: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if safe_source(root, path):
                selected.add(path.resolve())
    return sorted(selected, key=lambda item: item.relative_to(root).as_posix())


def assert_destination(destination: Path) -> None:
    if not (destination / ".git").is_dir():
        raise RuntimeError("Destination is not an existing Git clone.")
    branch = command_output("git", "branch", "--show-current", cwd=destination)
    if branch in {"main", "master", ""}:
        raise RuntimeError(f"Refusing curated sync on default/unknown branch: {branch!r}")
    remote = command_output("git", "remote", "get-url", "origin", cwd=destination)
    if "Saket-Maganti/gnn-fraud" not in remote:
        raise RuntimeError(f"Unexpected origin remote: {remote}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    assert_destination(destination)
    selected = expand(source, parse_allowlist(args.allowlist.resolve()))
    if not selected:
        raise RuntimeError("Allowlist expanded to zero files.")

    mappings: list[dict[str, str]] = []
    for source_path in selected:
        source_rel = source_path.relative_to(source).as_posix()
        destination_rel = DESTINATION_MAP.get(source_rel, source_rel)
        destination_path = destination / destination_rel
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        mappings.append(
            {
                "source_path": source_rel,
                "github_path": destination_rel,
                "mapping_reason": (
                    "Final PDF mapped to the GitHub-facing paper/pdf directory."
                    if source_rel in DESTINATION_MAP
                    else "Canonical relative path preserved."
                ),
            }
        )

    os_metadata = destination / ".DS_Store"
    if os_metadata.exists():
        os_metadata.unlink()

    map_path = destination / "docs" / "SOURCE_TO_GITHUB_PATH_MAP.csv"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "github_path", "mapping_reason"],
        )
        writer.writeheader()
        writer.writerows(mappings)
    print(f"selected_files={len(selected)}")
    print(f"path_map={map_path.relative_to(destination).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
