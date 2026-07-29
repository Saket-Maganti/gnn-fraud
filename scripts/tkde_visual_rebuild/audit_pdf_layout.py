#!/usr/bin/env python3
"""Run strict Poppler, font, log, and page-layout audits on TKDE PDFs.

The checker uses only published PDF/LaTeX interfaces: ``pdfinfo`` for document
metadata, ``pdffonts`` for embedding/type checks, ``pdftotext -bbox-layout``
for print-scale text geometry, and ``pdftoppm`` for page rasterization.  It
does not claim that automated geometry replaces human visual review; instead,
it rejects mechanical failures and emits page-level measurements that make the
required human print/grayscale inspection auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = ROOT / "results" / "tkde_visual_rebuild" / "audits"
REQUIRED_COMMANDS = ("pdfinfo", "pdffonts", "pdftohtml", "pdftoppm")

LOG_FATAL_PATTERNS = (
    ("OVERFULL_BOX", re.compile(r"Overfull \\[hv]box")),
    ("UNDEFINED_CITATION", re.compile(r"Citation .+ undefined|There were undefined citations")),
    ("UNDEFINED_REFERENCE", re.compile(r"Reference .+ undefined|There were undefined references")),
    ("RERUN_REQUIRED", re.compile(r"Rerun to get cross-references right")),
    ("DUPLICATE_LABEL", re.compile(r"multiply defined|Label .+ multiply defined")),
    ("MISSING_CHARACTER", re.compile(r"Missing character:")),
    ("LATEX_FATAL", re.compile(r"^! |Fatal error occurred|Emergency stop", re.MULTILINE)),
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    document: str
    page: int | None
    message: str


@dataclass(frozen=True)
class FontRecord:
    name: str
    font_type: str
    encoding: str
    embedded: bool
    subset: bool
    unicode_map: bool


@dataclass(frozen=True)
class PageRecord:
    document: str
    page: int
    width_px: int
    height_px: int
    orientation: str
    ink_fraction: float
    content_bbox_fraction: float
    horizontal_coverage: float
    vertical_coverage: float
    border_ink_pixels: int
    extracted_words: int
    median_word_height_pt: float | None
    p10_word_height_pt: float | None
    layout_status: str


@dataclass(frozen=True)
class DocumentRecord:
    name: str
    pdf: str
    log: str
    pages: int
    page_width_pt: float
    page_height_pt: float
    encrypted: bool
    font_count: int
    embedded_font_count: int
    type3_font_count: int


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{detail}")
    return proc.stdout


def parse_pdfinfo(output: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def parse_page_size(value: str) -> tuple[float, float]:
    match = re.search(r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts", value)
    if not match:
        raise ValueError(f"Could not parse PDF page size: {value!r}")
    return float(match.group(1)), float(match.group(2))


def parse_pdffonts(output: str) -> list[FontRecord]:
    records: list[FontRecord] = []
    lines = output.splitlines()
    separator = next(
        (i for i, line in enumerate(lines) if "-" in line and re.fullmatch(r"[- ]+", line)),
        None,
    )
    if separator is None:
        return records
    # Poppler uses fixed columns but font types contain spaces.  The final six
    # fields are stable: encoding, emb, sub, uni, object number, generation.
    for line in lines[separator + 1 :]:
        if not line.strip():
            continue
        match = re.match(
            r"^(?P<name>.{1,37}?)\s{2,}(?P<type>.{1,18}?)\s{2,}"
            r"(?P<encoding>\S+)\s+(?P<emb>yes|no)\s+(?P<sub>yes|no)\s+"
            r"(?P<uni>yes|no)\s+\d+\s+\d+\s*$",
            line,
        )
        if not match:
            # Fallback split from the right for Poppler builds with different
            # fixed-column widths.
            parts = line.rsplit(maxsplit=7)
            if len(parts) != 8:
                continue
            left, encoding, emb, sub, uni, _obj, _gen = (
                " ".join(parts[:-6]),
                parts[-6],
                parts[-5],
                parts[-4],
                parts[-3],
                parts[-2],
                parts[-1],
            )
            left_parts = re.split(r"\s{2,}", left, maxsplit=1)
            if len(left_parts) != 2:
                continue
            name, font_type = left_parts
        else:
            name = match.group("name").strip()
            font_type = match.group("type").strip()
            encoding = match.group("encoding")
            emb = match.group("emb")
            sub = match.group("sub")
            uni = match.group("uni")
        records.append(
            FontRecord(
                name=name.strip(),
                font_type=font_type.strip(),
                encoding=encoding,
                embedded=emb == "yes",
                subset=sub == "yes",
                unicode_map=uni == "yes",
            )
        )
    return records


def _pgm_tokens(data: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    index = 0
    length = len(data)
    while len(tokens) < 4 and index < length:
        while index < length and chr(data[index]).isspace():
            index += 1
        if index < length and data[index] == ord("#"):
            while index < length and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < length and not chr(data[index]).isspace():
            index += 1
        if start < index:
            tokens.append(data[start:index])
    while index < length and chr(data[index]).isspace():
        index += 1
    return tokens, index


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    tokens, offset = _pgm_tokens(data)
    if len(tokens) != 4 or tokens[0] != b"P5":
        raise ValueError(f"Unsupported PGM format: {path}")
    width, height, maximum = map(int, tokens[1:])
    if maximum != 255:
        raise ValueError(f"Unsupported PGM maximum {maximum}: {path}")
    pixels = data[offset : offset + width * height]
    if len(pixels) != width * height:
        raise ValueError(f"Truncated PGM raster: {path}")
    return width, height, pixels


def raster_metrics(path: Path) -> tuple[int, int, float, float, float, float, int]:
    width, height, pixels = read_pgm(path)
    threshold = 245
    ink = 0
    min_x, min_y = width, height
    max_x = max_y = -1
    border = 0
    border_width = max(1, round(min(width, height) * 0.0025))
    for y in range(height):
        base = y * width
        for x in range(width):
            if pixels[base + x] >= threshold:
                continue
            ink += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            if x < border_width or x >= width - border_width or y < border_width or y >= height - border_width:
                border += 1
    ink_fraction = ink / (width * height)
    if ink == 0:
        return width, height, ink_fraction, 0.0, 0.0, 0.0, border
    horizontal = (max_x - min_x + 1) / width
    vertical = (max_y - min_y + 1) / height
    return width, height, ink_fraction, horizontal * vertical, horizontal, vertical, border


def bbox_word_metrics(
    pdf: Path,
    output_xml: Path,
    page_width_pt: float,
) -> dict[int, tuple[int, float | None, float | None]]:
    """Extract print-scale text geometry through Poppler's stable XML path."""

    run_command(["pdftohtml", "-xml", "-hidden", "-i", str(pdf), str(output_xml)])
    root = ET.parse(output_xml).getroot()
    result: dict[int, tuple[int, float | None, float | None]] = {}
    pages = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "page"]
    for index, page in enumerate(pages, start=1):
        heights: list[float] = []
        try:
            xml_width = float(page.attrib["width"])
        except (KeyError, ValueError):
            xml_width = page_width_pt
        scale = page_width_pt / xml_width if xml_width > 0 else 1.0
        for text in page.iter():
            if text.tag.rsplit("}", 1)[-1] != "text":
                continue
            try:
                height = float(text.attrib["height"]) * scale
            except (KeyError, ValueError):
                continue
            word_count = len(" ".join(text.itertext()).split())
            if height > 0 and word_count:
                heights.extend([height] * word_count)
        if heights:
            ordered = sorted(heights)
            p10 = ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)]
            result[index] = (len(ordered), statistics.median(ordered), p10)
        else:
            result[index] = (0, None, None)
    return result


def log_findings(name: str, log: Path, pdf: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not log.is_file():
        return [Finding("ERROR", "MISSING_LATEX_LOG", name, None, f"Missing strict-build log: {log.name}")]
    text = log.read_text(encoding="utf-8", errors="replace")
    for code, pattern in LOG_FATAL_PATTERNS:
        count = len(pattern.findall(text))
        if count:
            findings.append(Finding("ERROR", code, name, None, f"LaTeX log contains {count} matching warning(s)."))
    if not re.search(r"Output written on .+\.pdf \(\d+ page", text):
        findings.append(
            Finding("ERROR", "INCOMPLETE_LATEX_LOG", name, None, "Log lacks a successful PDF output record.")
        )
    if pdf.is_file() and log.stat().st_mtime + 1 < pdf.stat().st_mtime:
        findings.append(
            Finding("WARNING", "LOG_OLDER_THAN_PDF", name, None, "PDF is newer than its retained LaTeX log.")
        )
    return findings


def audit_document(
    name: str,
    pdf: Path,
    log: Path,
    root: Path,
    work_dir: Path,
    max_pages: int | None,
) -> tuple[DocumentRecord | None, list[FontRecord], list[PageRecord], list[Finding]]:
    findings: list[Finding] = []
    if not pdf.is_file():
        return None, [], [], [Finding("ERROR", "MISSING_PDF", name, None, f"Missing PDF: {_relative(pdf, root)}")]
    try:
        info = parse_pdfinfo(run_command(["pdfinfo", str(pdf)]))
        pages = int(info["Pages"])
        page_width, page_height = parse_page_size(info["Page size"])
    except (KeyError, ValueError, RuntimeError) as exc:
        return None, [], [], [Finding("ERROR", "PDFINFO_FAILURE", name, None, str(exc))]
    encrypted = info.get("Encrypted", "no").lower() != "no"
    if encrypted:
        findings.append(Finding("ERROR", "ENCRYPTED_PDF", name, None, "Reviewer PDF must not be encrypted."))
    if pages <= 0:
        findings.append(Finding("ERROR", "EMPTY_PDF", name, None, "PDF reports no pages."))
    if max_pages is not None and pages > max_pages:
        findings.append(
            Finding("ERROR", "PAGE_ENVELOPE_EXCEEDED", name, None, f"{pages} pages exceed the configured limit {max_pages}.")
        )

    try:
        fonts = parse_pdffonts(run_command(["pdffonts", str(pdf)]))
    except RuntimeError as exc:
        fonts = []
        findings.append(Finding("ERROR", "PDFFONTS_FAILURE", name, None, str(exc)))
    if not fonts:
        findings.append(Finding("ERROR", "NO_FONT_RECORDS", name, None, "pdffonts returned no parseable font records."))
    for font in fonts:
        if not font.embedded:
            findings.append(
                Finding("ERROR", "UNEMBEDDED_FONT", name, None, f"Font is not embedded: {font.name} ({font.font_type}).")
            )
        if font.font_type.lower().replace(" ", "") == "type3":
            findings.append(Finding("ERROR", "TYPE3_FONT", name, None, f"Type 3 font detected: {font.name}."))

    findings.extend(log_findings(name, log, pdf))
    prefix = work_dir / f"{name}_page"
    try:
        run_command(["pdftoppm", "-gray", "-r", "72", str(pdf), str(prefix)])
        word_metrics = bbox_word_metrics(pdf, work_dir / f"{name}_text.xml", page_width)
    except (RuntimeError, ET.ParseError) as exc:
        return None, fonts, [], findings + [Finding("ERROR", "PDF_RENDER_FAILURE", name, None, str(exc))]
    rasters = sorted(work_dir.glob(f"{name}_page-*.pgm"), key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    if len(rasters) != pages:
        findings.append(
            Finding("ERROR", "RENDERED_PAGE_COUNT_MISMATCH", name, None, f"Rendered {len(rasters)} of {pages} pages.")
        )
    page_records: list[PageRecord] = []
    for page_number, raster in enumerate(rasters, start=1):
        try:
            width, height, ink, bbox, horizontal, vertical, border = raster_metrics(raster)
        except ValueError as exc:
            findings.append(Finding("ERROR", "RASTER_PARSE_FAILURE", name, page_number, str(exc)))
            continue
        words, median_height, p10_height = word_metrics.get(page_number, (0, None, None))
        orientation = "landscape" if width > height else "portrait"
        status = "OK"
        if ink < 0.0008 or (words == 0 and ink < 0.002):
            status = "BLANK_OR_NEAR_BLANK"
            findings.append(
                Finding("ERROR", "BLANK_OR_NEAR_BLANK_PAGE", name, page_number, f"Ink fraction is only {ink:.5f}.")
            )
        elif orientation == "landscape" and bbox < 0.42:
            status = "SEVERELY_UNDERUTILIZED_LANDSCAPE"
            findings.append(
                Finding(
                    "ERROR",
                    "SEVERELY_UNDERUTILIZED_LANDSCAPE",
                    name,
                    page_number,
                    f"Landscape ink={ink:.4f}, content bbox={bbox:.3f}; redesign or document an exception.",
                )
            )
        elif ink < 0.007 and bbox < 0.48:
            status = "SPARSE"
            findings.append(
                Finding(
                    "WARNING",
                    "SPARSE_PAGE",
                    name,
                    page_number,
                    f"Ink={ink:.4f}, content bbox={bbox:.3f}; inspect float/page utilization.",
                )
            )
        if border:
            status = "CONTENT_AT_PAGE_EDGE" if status == "OK" else f"{status}+EDGE"
            findings.append(
                Finding(
                    "ERROR",
                    "CONTENT_AT_PAGE_EDGE",
                    name,
                    page_number,
                    f"{border} nonwhite pixels touch the outer 0.25% page border.",
                )
            )
        if words >= 60 and median_height is not None and median_height < 4.2:
            status = "MICROSCOPIC_TEXT" if status == "OK" else f"{status}+MICROTEXT"
            findings.append(
                Finding(
                    "ERROR",
                    "MICROSCOPIC_PAGE_TEXT",
                    name,
                    page_number,
                    f"Median extracted word height is {median_height:.2f} pt across {words} words.",
                )
            )
        page_records.append(
            PageRecord(
                document=name,
                page=page_number,
                width_px=width,
                height_px=height,
                orientation=orientation,
                ink_fraction=round(ink, 6),
                content_bbox_fraction=round(bbox, 6),
                horizontal_coverage=round(horizontal, 6),
                vertical_coverage=round(vertical, 6),
                border_ink_pixels=border,
                extracted_words=words,
                median_word_height_pt=round(median_height, 3) if median_height is not None else None,
                p10_word_height_pt=round(p10_height, 3) if p10_height is not None else None,
                layout_status=status,
            )
        )

    document = DocumentRecord(
        name=name,
        pdf=_relative(pdf, root),
        log=_relative(log, root),
        pages=pages,
        page_width_pt=page_width,
        page_height_pt=page_height,
        encrypted=encrypted,
        font_count=len(fonts),
        embedded_font_count=sum(font.embedded for font in fonts),
        type3_font_count=sum(font.font_type.lower().replace(" ", "") == "type3" for font in fonts),
    )
    return document, fonts, page_records, findings


def write_reports(
    report_dir: Path,
    documents: Sequence[DocumentRecord],
    fonts: dict[str, Sequence[FontRecord]],
    pages: Sequence[PageRecord],
    findings: Sequence[Finding],
    dependencies: dict[str, str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    payload = {
        "verdict": "PASS" if errors == 0 else "FAIL",
        "error_count": errors,
        "warning_count": warnings,
        "dependencies": dependencies,
        "documents": [asdict(item) for item in documents],
        "fonts": {name: [asdict(font) for font in records] for name, records in fonts.items()},
        "pages": [asdict(item) for item in pages],
        "findings": [asdict(item) for item in findings],
        "human_review_required": (
            "Inspect full-page, actual-print-scale, and grayscale renders; automated geometry does not "
            "establish scientific clarity or grayscale semantic equivalence."
        ),
    }
    (report_dir / "PDF_LAYOUT_AUDIT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PDF layout and preflight audit",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        f"- Errors: {errors}",
        f"- Warnings: {warnings}",
        "- Automated checks: Poppler readability, page count, rendering, fonts, LaTeX logs, blank/sparse pages, clipping proxy, and print-scale text geometry.",
        "- Human gate still required: full-page, 100% print-scale, grayscale semantic, and page-rhythm inspection.",
        "",
        "## Documents",
        "",
        "| Document | Pages | Size (pt) | Fonts | Embedded | Type 3 |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in documents:
        lines.append(
            f"| {item.name} | {item.pages} | {item.page_width_pt:g} x {item.page_height_pt:g} | "
            f"{item.font_count} | {item.embedded_font_count} | {item.type3_font_count} |"
        )
    lines.extend(
        [
            "",
            "## Page measurements",
            "",
            "| Document | Page | Orientation | Ink | BBox use | Words | Median word height | Status |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in pages:
        median = "n/a" if item.median_word_height_pt is None else f"{item.median_word_height_pt:.2f} pt"
        lines.append(
            f"| {item.document} | {item.page} | {item.orientation} | {item.ink_fraction:.4f} | "
            f"{item.content_bbox_fraction:.3f} | {item.extracted_words} | {median} | {item.layout_status} |"
        )
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("No mechanical PDF, font, log, clipping, blank-page, or severe utilization defect was detected.")
    else:
        for item in findings:
            page = f" page {item.page}" if item.page is not None else ""
            lines.append(f"- **{item.severity} {item.code}** {item.document}{page}: {item.message}")
    (report_dir / "PDF_LAYOUT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--main-pdf", type=Path)
    parser.add_argument("--supplement-pdf", type=Path)
    parser.add_argument("--main-log", type=Path)
    parser.add_argument("--supplement-log", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--max-main-pages", type=int, default=14)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--warnings-as-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    dependencies = {name: shutil.which(name) or "" for name in REQUIRED_COMMANDS}
    findings: list[Finding] = []
    for name, path in dependencies.items():
        if not path:
            findings.append(Finding("ERROR", "MISSING_DEPENDENCY", "toolchain", None, f"Required command is unavailable: {name}."))
    documents: list[DocumentRecord] = []
    all_pages: list[PageRecord] = []
    fonts: dict[str, Sequence[FontRecord]] = {}
    if not findings:
        specs = (
            (
                "main",
                (args.main_pdf or root / "paper_tkde" / "main.pdf").resolve(),
                (args.main_log or root / "paper_tkde" / "main.log").resolve(),
                args.max_main_pages,
            ),
            (
                "supplement",
                (args.supplement_pdf or root / "paper_tkde" / "supplement" / "supplement.pdf").resolve(),
                (args.supplement_log or root / "paper_tkde" / "supplement" / "supplement.log").resolve(),
                None,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="tkde_pdf_audit_") as directory:
            work_dir = Path(directory)
            for name, pdf, log, limit in specs:
                document, font_rows, page_rows, document_findings = audit_document(
                    name, pdf, log, root, work_dir, limit
                )
                if document is not None:
                    documents.append(document)
                fonts[name] = font_rows
                all_pages.extend(page_rows)
                findings.extend(document_findings)
    findings.sort(key=lambda item: (item.severity != "ERROR", item.document, item.page or 0, item.code))
    report_dir = (args.report_dir or root / "results" / "tkde_visual_rebuild" / "audits").resolve()
    write_reports(report_dir, documents, fonts, all_pages, findings, dependencies)
    errors = sum(item.severity == "ERROR" for item in findings)
    warnings = sum(item.severity == "WARNING" for item in findings)
    print(f"documents={len(documents)} pages={len(all_pages)} errors={errors} warnings={warnings}")
    if args.strict and (errors or (args.warnings_as_errors and warnings)):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
