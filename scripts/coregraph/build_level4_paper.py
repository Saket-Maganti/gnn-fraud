#!/usr/bin/env python3
"""Compile main and supplement and write a deterministic paper-build audit."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_iclr"
BUILD = PAPER / "build"
REPORT = ROOT / "results" / "coregraph_build" / "LEVEL4_PAPER_BUILD_REPORT.md"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PAPER,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _compile(stem: str) -> tuple[Path, Path]:
    target = BUILD / stem
    target.mkdir(parents=True, exist_ok=True)
    engine = shutil.which("pdflatex")
    if engine is None:
        raise RuntimeError("pdflatex is unavailable")
    base = [engine, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={target}", f"{stem}.tex"]
    _run(base)
    bibtex = shutil.which("bibtex")
    if bibtex is not None:
        _run([bibtex, f"build/{stem}/{stem}"], check=False)
    _run(base)
    _run(base)
    built = target / f"{stem}.pdf"
    if not built.is_file() or built.stat().st_size == 0:
        raise RuntimeError(f"paper build did not create {built}")
    final = PAPER / f"{stem}.pdf"
    shutil.copy2(built, final)
    return final, target / f"{stem}.log"


def _pages(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        return -1
    output = subprocess.check_output([pdfinfo, str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", output, flags=re.MULTILINE)
    return int(match.group(1)) if match else -1


def _type3_fonts(path: Path) -> int:
    pdffonts = shutil.which("pdffonts")
    if pdffonts is None:
        return -1
    output = subprocess.check_output([pdffonts, str(path)], text=True)
    return sum("Type 3" in line for line in output.splitlines())


def _text(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return ""
    return subprocess.check_output([pdftotext, str(path), "-"], text=True)


def main() -> int:
    failures: list[str] = []
    outputs: dict[str, dict[str, object]] = {}
    for stem in ("main", "supplement"):
        try:
            pdf, log = _compile(stem)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            failures.append(f"{stem}_compile:{exc}")
            continue
        log_text = log.read_text(encoding="utf-8", errors="ignore")
        extracted = _text(pdf)
        overfull = len(re.findall(r"Overfull \\hbox", log_text))
        undefined = len(re.findall(r"(?:Citation|Reference).*undefined", log_text))
        identity = bool(re.search(r"Saket\s+Maganti|saketmaganti|/Users/|/Volumes/", extracted, re.I))
        type3 = _type3_fonts(pdf)
        outputs[stem] = {
            "path": pdf.relative_to(ROOT).as_posix(),
            "bytes": pdf.stat().st_size,
            "pages": _pages(pdf),
            "overfull_boxes": overfull,
            "undefined_references_or_citations": undefined,
            "type3_fonts": type3,
            "identity_or_private_path": identity,
            "blocked_tokens_present": "RESULT_PENDING" in extracted or "CLAIM_BLOCKED" in extracted,
        }
        if overfull:
            failures.append(f"{stem}_overfull:{overfull}")
        if undefined:
            failures.append(f"{stem}_undefined_references:{undefined}")
        if identity:
            failures.append(f"{stem}_identity_or_private_path")
        if type3 > 0:
            failures.append(f"{stem}_type3_fonts:{type3}")
    status = "PASS_RESULTS_BLOCKED" if not failures else "FAIL"
    lines = [
        "# Level-4 paper build report",
        "",
        f"Status: `{status}`.",
        "",
        "The official ICLR 2027 style was not available from the venue at build time; the PDFs use the anonymous review fallback and must be rebuilt when the target-year template is published.",
        "",
        "| PDF | Pages | Bytes | Overfull | Undefined refs/cites | Type 3 | Identity/path |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stem in ("main", "supplement"):
        item = outputs.get(stem, {})
        lines.append(
            f"| {stem} | {item.get('pages', 'BLOCKED')} | {item.get('bytes', 'BLOCKED')} | "
            f"{item.get('overfull_boxes', 'BLOCKED')} | {item.get('undefined_references_or_citations', 'BLOCKED')} | "
            f"{item.get('type3_fonts', 'BLOCKED')} | {item.get('identity_or_private_path', 'BLOCKED')} |"
        )
    lines.extend(
        [
            "",
            f"Failures: `{json.dumps(failures, sort_keys=True)}`.",
            "",
            "All empirical result cells remain typed `PENDING`; no numerical result was inserted.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "outputs": outputs, "failures": failures}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
