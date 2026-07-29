#!/usr/bin/env python3
"""Generate and reconcile the exhaustive frozen-baseline visual inventory."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tkde_visual_rebuild.object_registry import ROOT, all_objects, rows


OUT = ROOT / "results" / "tkde_visual_rebuild"
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}")


def active_tex(entrypoint: Path) -> set[Path]:
    """Return the current active TeX closure for one independently built PDF."""

    compile_dir = entrypoint.parent.resolve()
    paper_root = (ROOT / "paper_tkde").resolve()
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


def write_csv(path: Path, data: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = fields or list(data[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def write_markdown(path: Path, data: list[dict[str, object]]) -> None:
    counts = Counter(str(row["object_type"]) for row in data)
    dispositions = Counter(str(row["final_disposition"]) for row in data)
    lines = [
        "# Visual object inventory",
        "",
        "This registry covers every active visual, table, display equation, and landscape block in the frozen 14-page main paper and 47-page supplement. Algorithms: 0. The inventory records the baseline object and its required final disposition; final-source reconciliation is performed separately after reconstruction.",
        "",
        "## Counts",
        "",
        f"- Total objects: {len(data)}",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    lines.extend(["", "## Dispositions", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(dispositions.items()))
    lines.extend(["", "## Object-level decisions", "", "| ID | Type | Document | Page | Label/title | Disposition | Destination | Planned replacement |", "| --- | --- | --- | ---: | --- | --- | --- | --- |"])
    for row in data:
        label = row["latex_label"] if not str(row["latex_label"]).startswith("unlabeled") else row["caption_title"]
        lines.append(f"| {row['object_id']} | {row['object_type']} | {row['document']} | {row['current_pdf_page']} | {label} | `{row['final_disposition']}` | {row['final_destination']} | {row['planned_replacement']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile(strict: bool) -> list[str]:
    problems: list[str] = []
    objects = all_objects()
    expected = Counter({"figure": 12, "table": 7, "equation": 25, "longtable": 23, "landscape_block": 5})
    actual = Counter(obj.object_type for obj in objects)
    if actual != expected:
        problems.append(f"type counts differ: {actual} != {expected}")
    for obj in objects:
        for token in (part.strip() for part in obj.source_data_files.split(";")):
            if token.startswith(("results/", "paper_tkde/", "manuscript_assets/")) and not (ROOT / token).exists():
                problems.append(f"source data absent: {obj.object_id} {token}")
        if not obj.planned_replacement.strip() or not obj.final_destination.strip():
            problems.append(f"unallocated baseline object: {obj.object_id}")

    paper_root = ROOT / "paper_tkde"
    active = active_tex(paper_root / "main.tex")
    active.update(active_tex(paper_root / "supplement" / "supplement.tex"))
    active_tables = {
        path.resolve()
        for path in active
        if path.parent.name == "tables" and path.suffix == ".tex"
    }
    main_manifest = list(
        csv.DictReader((OUT / "MAIN_TABLE_DATA_PROVENANCE.csv").open(encoding="utf-8"))
    )
    supplement_manifest = list(
        csv.DictReader((OUT / "CURATED_SUPPLEMENT_TABLE_MANIFEST.csv").open(encoding="utf-8"))
    )
    manifested_tables = {
        (ROOT / row["table_file"]).resolve() for row in main_manifest
    } | {
        (ROOT / row["table_fragment"]).resolve() for row in supplement_manifest
    }
    if active_tables != manifested_tables:
        missing = sorted(str(path.relative_to(ROOT)) for path in active_tables - manifested_tables)
        extra = sorted(str(path.relative_to(ROOT)) for path in manifested_tables - active_tables)
        problems.append(f"active/table-manifest mismatch: missing={missing}; extra={extra}")
    if len(main_manifest) != 8 or len(supplement_manifest) != 43:
        problems.append(
            f"final table counts differ: main={len(main_manifest)}, supplement={len(supplement_manifest)}"
        )

    figure_manifest = list(
        csv.DictReader((OUT / "FIGURE_DATA_PROVENANCE.csv").open(encoding="utf-8"))
    )
    manifested_figures = {(ROOT / row["figure_file"]).resolve() for row in figure_manifest}
    active_figures: set[Path] = set()
    for path in active:
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in GRAPHICS_RE.findall(text):
            requested = Path(target)
            if requested.suffix == "":
                requested = requested.with_suffix(".pdf")
            candidates = (path.parent / requested, path.parent / "figures" / requested.name, paper_root / "figures" / requested.name)
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                problems.append(f"active graphic absent: {path.relative_to(ROOT)} -> {target}")
            elif resolved.suffix.lower() == ".pdf":
                active_figures.add(resolved)
    if active_figures != manifested_figures:
        missing = sorted(str(path.relative_to(ROOT)) for path in active_figures - manifested_figures)
        extra = sorted(str(path.relative_to(ROOT)) for path in manifested_figures - active_figures)
        problems.append(f"active/figure-manifest mismatch: missing={missing}; extra={extra}")
    if len(figure_manifest) != 8:
        problems.append(f"final figure provenance count differs: {len(figure_manifest)} != 8")

    active_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in active)
    for token in (r"\begin{landscape}", r"\begin{sidewaystable}", r"\tiny", r"\scriptsize", r"\resizebox"):
        if token in active_text:
            problems.append(f"forbidden active layout token: {token}")
    if strict and problems:
        raise SystemExit("\n".join(problems))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    objects = all_objects()
    data = rows(objects)
    write_csv(OUT / "VISUAL_OBJECT_INVENTORY.csv", data)
    write_markdown(OUT / "VISUAL_OBJECT_INVENTORY.md", data)
    allocation_fields = ["object_id", "object_type", "latex_label", "caption_title", "document", "current_pdf_page", "final_disposition", "final_destination", "planned_replacement", "source_data_files", "provenance_checksum_record"]
    write_csv(OUT / "CONTENT_ALLOCATION_MAP.csv", data, allocation_fields)
    problems = reconcile(args.strict)
    print(f"objects={len(objects)} problems={len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
