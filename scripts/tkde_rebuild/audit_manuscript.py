#!/usr/bin/env python3
"""Deterministic source/PDF audit for the rebuilt TKDE manuscript.

This script performs no training and does not mutate canonical evidence.  It
checks the submission sources and, when available, the compiled PDFs, then
writes compact machine-readable and reviewer-readable audit reports.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_tkde"
OUT = ROOT / "results" / "tkde_rebuild"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def run(*args: str) -> tuple[int, str]:
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout + proc.stderr


def tex_tree() -> tuple[list[Path], list[Path]]:
    active = [
        "01_abstract.tex",
        "02_introduction.tex",
        "03_related_work.tex",
        "04_deployment_claim_contracts.tex",
        "05_benchmark_instantiation.tex",
        "06_experimental_design.tex",
        "07_results_protocol_architecture.tex",
        "08_results_construction_decisions.tex",
        "09_framework_validation_graphsafe.tex",
        "10_implications_validity_artifact.tex",
        "11_conclusion.tex",
    ]
    main_tables = [
        "table01_related_work.tex",
        "table02_dataset_tasks.tex",
        "table03_protocol_visibility.tex",
        "table04_rb09_protocol_effects.tex",
        "table05_ibm_baselines.tex",
        "table06_ibm_construction.tex",
        "table07_resource_boundaries.tex",
        "table08_graphsafe_case.tex",
    ]
    main = [PAPER / "main.tex"] + [PAPER / "sections" / name for name in active]
    main += [PAPER / "tables" / name for name in main_tables]
    supplement = sorted((PAPER / "supplement").rglob("*.tex"))
    return [p for p in main if p.exists()], supplement


def bibliography_keys() -> set[str]:
    return set(re.findall(r"^@[A-Za-z]+\{([^,]+),", read(PAPER / "references.bib"), re.M))


def citation_keys(text: str) -> list[str]:
    keys: list[str] = []
    for payload in re.findall(r"\\cite(?:t|p)?(?:\[[^\]]*\])?\{([^}]+)\}", text):
        keys.extend(k.strip() for k in payload.split(",") if k.strip())
    return keys


def source_audit(main: list[Path], supplement: list[Path]) -> dict[str, object]:
    bib = bibliography_keys()
    main_text = "\n".join(read(p) for p in main)
    supp_text = "\n".join(read(p) for p in supplement)
    submitted_text = main_text + "\n" + supp_text + "\n" + read(PAPER / "references.bib")
    cited_main = set(citation_keys(main_text))
    cited_supp = set(citation_keys(supp_text))
    labels = re.findall(r"\\label\{([^}]+)\}", submitted_text)
    refs = re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", submitted_text)
    duplicate_labels = sorted(k for k, n in Counter(labels).items() if n > 1)
    placeholders = []
    forbidden = {
        "TODO/FIXME": r"(?i)\b(?:TODO|FIXME|TBD)\b",
        "submission meta-language": r"(?i)(?:deliberately TKDE-first|submission-ready|for the reviewer)",
        "private absolute path": r"/(?:Users|home)/[^\s{}]+",
        "credential-like token": r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[^\s,;]+",
    }
    for label, pattern in forbidden.items():
        matches = re.findall(pattern, submitted_text)
        if matches:
            placeholders.append((label, len(matches)))
    return {
        "bib": bib,
        "cited_main": cited_main,
        "cited_supp": cited_supp,
        "missing_citations": sorted((cited_main | cited_supp) - bib),
        "unused_bib": sorted(bib - (cited_main | cited_supp)),
        "undefined_refs": sorted(set(refs) - set(labels)),
        "duplicate_labels": duplicate_labels,
        "forbidden": placeholders,
    }


def pdf_audit(pdf: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(pdf.relative_to(ROOT)), "exists": pdf.exists()}
    if not pdf.exists():
        return result
    rc, info = run("pdfinfo", str(pdf))
    page = re.search(r"^Pages:\s+(\d+)", info, re.M)
    result.update({"pdfinfo_rc": rc, "pages": int(page.group(1)) if page else None})
    rc, fonts = run("pdffonts", str(pdf))
    font_lines = [ln for ln in fonts.splitlines()[2:] if ln.strip()]
    result.update(
        {
            "pdffonts_rc": rc,
            "font_count": len(font_lines),
            "type3_fonts": [ln for ln in font_lines if "Type 3" in ln],
        }
    )
    rc, text = run("pdftotext", str(pdf), "-")
    result.update({"pdftotext_rc": rc, "word_count": len(re.findall(r"\b[\w'-]+\b", text))})
    return result


def latex_log_audit(log: Path) -> dict[str, object]:
    if not log.exists():
        return {"path": str(log.relative_to(ROOT)), "exists": False}
    text = read(log)
    overfull = re.findall(r"Overfull \\[hv]box \(([^)]+)\)", text)
    return {
        "path": str(log.relative_to(ROOT)),
        "exists": True,
        "undefined_citations": len(re.findall(r"Citation .* undefined", text)),
        "undefined_references": len(re.findall(r"Reference .* undefined", text)),
        "rerun_warnings": len(re.findall(r"Rerun to get cross-references right", text)),
        "overfull_boxes": overfull,
    }


def write_reports(source: dict[str, object], pdfs: list[dict[str, object]], logs: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cited_main = source["cited_main"]
    cited_supp = source["cited_supp"]
    bib = source["bib"]
    coverage = [
        "# Citation Coverage Audit",
        "",
        f"- Verified bibliography entries: **{len(bib)}**",
        f"- Distinct entries cited in the main paper: **{len(cited_main)}**",
        f"- Distinct entries cited in the supplement: **{len(cited_supp)}**",
        f"- Distinct entries cited anywhere: **{len(cited_main | cited_supp)}**",
        f"- Citation keys missing from the bibliography: **{len(source['missing_citations'])}**",
        f"- Verified entries unused by both documents: **{len(source['unused_bib'])}**",
        "",
        "## Missing keys",
        "",
        *(f"- `{key}`" for key in source["missing_citations"]),
        *( ["- None."] if not source["missing_citations"] else []),
        "",
        "## Unused verified entries",
        "",
        *(f"- `{key}`" for key in source["unused_bib"]),
        *( ["- None."] if not source["unused_bib"] else []),
        "",
        "## Cross-reference and source hygiene",
        "",
        f"- Undefined source-level references: {source['undefined_refs'] or 'none'}",
        f"- Duplicate labels: {source['duplicate_labels'] or 'none'}",
        f"- Forbidden-pattern hits: {source['forbidden'] or 'none'}",
    ]
    (OUT / "CITATION_COVERAGE_AUDIT.md").write_text("\n".join(coverage) + "\n", encoding="utf-8")

    rows: list[dict[str, object]] = []
    for item in pdfs:
        rows.append({"kind": "pdf", **item})
    for item in logs:
        rows.append({"kind": "log", **item})
    keys = sorted({key for row in rows for key in row})
    with (OUT / "MANUSCRIPT_MACHINE_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    main_tex, supp_tex = tex_tree()
    source = source_audit(main_tex, supp_tex)
    pdfs = [pdf_audit(PAPER / "main.pdf"), pdf_audit(PAPER / "supplement" / "supplement.pdf")]
    logs = [latex_log_audit(PAPER / "main.log"), latex_log_audit(PAPER / "supplement" / "supplement.log")]
    write_reports(source, pdfs, logs)
    fatal = bool(source["missing_citations"] or source["undefined_refs"] or source["duplicate_labels"] or source["forbidden"])
    print(
        f"bibliography={len(source['bib'])} cited={len(source['cited_main'] | source['cited_supp'])} "
        f"source_fatal={fatal}"
    )
    raise SystemExit(1 if fatal else 0)


if __name__ == "__main__":
    main()
