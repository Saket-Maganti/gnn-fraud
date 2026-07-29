#!/usr/bin/env python3
"""
Generate a no-training manifest for figure-like artifacts.

The script hashes existing image/PDF/SVG files only. It does not create,
modify, or regenerate figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCAN_DIRS = (
    "gnnpaper/paperfigures",
    "results/figures",
    "results",
    "figures",
    "figures1",
    "figures2",
)

FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

CATEGORY_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("calibration", ("calibration", "temperature", "threshold")),
    ("business_cost", ("business_cost", "business", "cost")),
    ("leakage", ("leakage", "transductive", "inductive", "protocol_gap")),
    ("pr_curve", ("pr_curve", "pr_curves", "precision_recall", "pr-curve")),
    ("confusion", ("confusion",)),
    ("drift", ("drift", "distribution_shift", "shift")),
    ("temporal", ("temporal", "timestep", "time_step", "per_timestep")),
    ("graph", ("graph", "ablation", "community", "edge", "structure")),
)


def _rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT)


def _utc_mtime(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guessed_category(relative_path: str) -> str:
    text = relative_path.lower()
    for category, needles in CATEGORY_PATTERNS:
        if any(needle in text for needle in needles):
            return category
    return "unknown"


def iter_figures(scan_dirs: Iterable[str]) -> Tuple[List[Dict[str, object]], List[str]]:
    records: List[Dict[str, object]] = []
    warnings: List[str] = []
    seen: set[str] = set()

    for rel_dir in scan_dirs:
        abs_dir = os.path.join(REPO_ROOT, rel_dir)
        if not os.path.isdir(abs_dir):
            warnings.append(f"Missing folder: {rel_dir}/")
            continue

        found_here = 0
        for root, _, files in os.walk(abs_dir):
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                if ext not in FIGURE_EXTENSIONS:
                    continue
                path = os.path.join(root, name)
                rel_path = _rel(path)
                if rel_path in seen:
                    continue
                seen.add(rel_path)
                found_here += 1
                records.append({
                    "relative_path": rel_path,
                    "filename": name,
                    "extension": ext,
                    "size_bytes": os.path.getsize(path),
                    "modified_time_utc": _utc_mtime(path),
                    "sha256": _sha256(path),
                    "guessed_category": guessed_category(rel_path),
                })
        if found_here == 0:
            warnings.append(f"No figure-like files found in: {rel_dir}/")

    records.sort(key=lambda row: str(row["relative_path"]))
    return records, warnings


def write_json(path: str, records: List[Dict[str, object]], warnings: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "generated_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": REPO_ROOT,
        "scan_dirs": list(SCAN_DIRS),
        "extensions": sorted(FIGURE_EXTENSIONS),
        "warnings": warnings,
        "count": len(records),
        "category_counts": dict(Counter(str(row["guessed_category"]) for row in records)),
        "figures": records,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def write_markdown(path: str, records: List[Dict[str, object]], warnings: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    counts = Counter(str(row["guessed_category"]) for row in records)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Figure manifest",
        "",
        f"Generated: {generated}",
        "",
        "This report inventories existing figure-like files only. It does not regenerate figures or infer figure purpose beyond filename/path heuristics.",
        "",
        "## Summary",
        "",
        f"- Figures found: {len(records)}",
        f"- JSON manifest: `gnnpaper/figure_manifest.json`",
        f"- Scanned folders: {', '.join(f'`{d}/`' for d in SCAN_DIRS)}",
        "",
        "## Category counts",
        "",
        "| guessed_category | count |",
        "| --- | ---: |",
    ]
    for category in (
        "temporal",
        "drift",
        "graph",
        "calibration",
        "business_cost",
        "leakage",
        "pr_curve",
        "confusion",
        "unknown",
    ):
        lines.append(f"| {category} | {counts.get(category, 0)} |")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Figures",
        "",
        "| relative_path | category | extension | size_bytes | modified_time_utc | sha256_prefix |",
        "| --- | --- | --- | ---: | --- | --- |",
    ])
    for row in records:
        lines.append(
            "| {relative_path} | {guessed_category} | {extension} | {size_bytes} | "
            "{modified_time_utc} | {sha256} |".format(
                **{
                    **row,
                    "sha256": str(row["sha256"])[:16],
                }
            )
        )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a SHA-256 manifest for existing figure artifacts.",
    )
    parser.add_argument(
        "--json-output",
        default=os.path.join(REPO_ROOT, "gnnpaper", "figure_manifest.json"),
    )
    parser.add_argument(
        "--report-output",
        default=os.path.join(REPO_ROOT, "results", "reports", "figure_manifest.md"),
    )
    args = parser.parse_args()

    records, warnings = iter_figures(SCAN_DIRS)
    write_json(args.json_output, records, warnings)
    write_markdown(args.report_output, records, warnings)

    print("=== figure manifest (no training) ===")
    print(f"figures: {len(records)}")
    print(f"json:    {_rel(args.json_output)}")
    print(f"report:  {_rel(args.report_output)}")
    if warnings:
        print(f"warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
