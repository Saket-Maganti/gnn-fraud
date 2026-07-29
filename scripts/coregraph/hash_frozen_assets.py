#!/usr/bin/env python3
"""Create or verify the immutable TKDE scientific boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "coregraph_build" / "FROZEN_TKDE_INPUT_HASHES.csv"
FROZEN_PREFIXES = (
    "results/tkde_rebuild/",
    "results/tkde_visual_rebuild/",
    "results/v24_imported/",
    "results/v26_imported/",
    "results/v27_imported/",
    "results/v28_imported/",
    "results/runs_rb09v3/",
    "results/runs_rb17_review_budget_worst_block/",
    "paper_tkde/",
    "paper/pdf/",
    "manuscript_assets/",
)
FROZEN_FILES = {
    "kaggle_workspace/manifests/V22_FINAL_GPU_EVIDENCE_LOCK.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def tracked_frozen_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return sorted(
        path
        for path in output
        if path in FROZEN_FILES or any(path.startswith(prefix) for prefix in FROZEN_PREFIXES)
    )


def write_manifest() -> None:
    paths = tracked_frozen_paths()
    if not paths:
        raise RuntimeError("frozen boundary resolved to no tracked files")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "sha256_initial", "sha256_current", "identical"),
            lineterminator="\n",
        )
        writer.writeheader()
        for relative in paths:
            value = digest(ROOT / relative)
            writer.writerow(
                {
                    "path": relative,
                    "sha256_initial": value,
                    "sha256_current": value,
                    "identical": "true",
                }
            )
    print(f"WROTE {len(paths)} frozen hashes to {OUTPUT.relative_to(ROOT)}")


def verify_manifest() -> int:
    if not OUTPUT.is_file():
        print("FROZEN HASH MANIFEST MISSING")
        return 2
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    recorded = {row["path"]: row["sha256_initial"] for row in rows}
    current_paths = tracked_frozen_paths()
    failures: list[str] = []
    if set(recorded) != set(current_paths):
        failures.append("frozen_path_set_changed")
    for relative in current_paths:
        actual = digest(ROOT / relative)
        if recorded.get(relative) != actual:
            failures.append(f"hash_changed:{relative}")
    if failures:
        print("TKDE SCIENTIFIC DELTAS DETECTED")
        print("\n".join(failures))
        return 1
    print(f"ZERO_TKDE_SCIENTIFIC_DELTAS ({len(rows)} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write == args.verify:
        parser.error("choose exactly one of --write or --verify")
    if args.write:
        write_manifest()
        return 0
    return verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
