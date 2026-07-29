from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import zipfile

import pytest

from scripts.tkde_visual_rebuild import audit_pdf_layout as pdf_audit
from scripts.tkde_visual_rebuild import audit_table_readability as table_audit
from scripts.tkde_visual_rebuild import build_visual_release as release
from scripts.tkde_visual_rebuild import scientific_delta_gate as delta_gate


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_table_audit_accepts_readable_booktabs_table(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "paper_tkde" / "tables" / "good.tex",
        r"""
\begin{table}[t]
\caption{Matched effect summary.}
\label{tab:good}
\centering
\footnotesize
\begin{tabularx}{\columnwidth}{@{}lcr@{}}
\toprule
Model & Mean & Delta \\
\midrule
MLP & 0.20 & 0.00 \\
SAGE & 0.30 & 0.10 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )
    records, findings = table_audit.extract_tables(source, "main", tmp_path)
    assert len(records) == 1
    assert records[0].estimated_columns == 3
    assert records[0].font_command == "footnotesize"
    assert not [item for item in findings if item.severity == "ERROR"]


def test_table_audit_rejects_microscopic_raw_and_blocked_numeric_table(tmp_path: Path) -> None:
    rows = "\n".join(f"seed {index} & 0.123 \\\\" for index in range(90))
    source = _write(
        tmp_path / "paper_tkde" / "tables" / "bad.tex",
        rf"""
\begin{{table*}}[t]
\caption{{Complete raw seed row dump.}}
\label{{tab:bad}}
\tiny
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l|r}}
\toprule
Seed & AUPRC \\
\midrule
{rows}
resource-blocked & 0.456 \\
\bottomrule
\end{{tabular}}}}
\end{{table*}}
""",
    )
    _, findings = table_audit.extract_tables(source, "main", tmp_path)
    codes = {item.code for item in findings}
    assert {
        "TINY_TABLE",
        "RESIZEBOX_TABLE",
        "VERTICAL_TABLE_RULE",
        "RAW_ROW_DUMP",
        "BLOCKED_ROW_HAS_NUMERIC_PERFORMANCE",
    }.issubset(codes)


def test_active_tex_closure_fails_closed_on_missing_input(tmp_path: Path) -> None:
    entrypoint = _write(
        tmp_path / "paper_tkde" / "main.tex",
        r"\input{sections/missing}",
    )
    _, findings = table_audit.active_tex_closure(entrypoint, tmp_path)
    assert any(item.code == "MISSING_TEX_DEPENDENCY" for item in findings)


def test_pdffonts_parser_and_type3_detection_shape() -> None:
    output = """name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
AAAAAA+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes      4  0
BadBitmap                            Type 3            Custom           no  no  no       5  0
"""
    records = pdf_audit.parse_pdffonts(output)
    assert len(records) == 2
    assert records[0].embedded is True
    assert records[1].font_type == "Type 3"
    assert records[1].embedded is False


def test_pgm_layout_metrics_detect_border_ink(tmp_path: Path) -> None:
    width, height = 10, 8
    pixels = bytearray([255] * (width * height))
    pixels[0] = 0
    pixels[4 * width + 5] = 0
    pgm = tmp_path / "page.pgm"
    pgm.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))
    parsed = pdf_audit.raster_metrics(pgm)
    assert parsed[0:2] == (width, height)
    assert parsed[-1] == 1
    assert parsed[2] == pytest.approx(2 / (width * height))


def test_latex_log_scan_rejects_overfull_and_undefined(tmp_path: Path) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-fixture")
    log = _write(
        tmp_path / "main.log",
        "Overfull \\hbox\nCitation `x' undefined\nOutput written on main.pdf (1 page).\n",
    )
    codes = {item.code for item in pdf_audit.log_findings("main", log, pdf)}
    assert "OVERFULL_BOX" in codes
    assert "UNDEFINED_CITATION" in codes


def _freeze_ledger(path: Path, relpath: str, expected: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "artifact_type",
                "path",
                "sha256_before",
                "sha256_after",
                "identical",
                "command_exit",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "artifact_type": "generated_analysis",
                "path": relpath,
                "sha256_before": expected,
                "sha256_after": expected,
                "identical": "True",
                "command_exit": "0",
            }
        )
    return path


def test_frozen_hash_gate_rejects_scientific_change(tmp_path: Path) -> None:
    target = _write(tmp_path / "results" / "tkde_rebuild" / "SCIENCE.csv", "a\n2\n")
    expected = hashlib.sha256(b"a\n1\n").hexdigest()
    ledger = _freeze_ledger(tmp_path / "freeze.csv", "results/tkde_rebuild/SCIENCE.csv", expected)
    findings, _ = delta_gate.frozen_hash_findings(tmp_path, ledger)
    assert any(item.code == "FROZEN_HASH_CHANGED" for item in findings)
    assert target.is_file()


def test_presentation_provenance_hash_is_validated_semantically_not_frozen(tmp_path: Path) -> None:
    relpath = "results/tkde_rebuild/FIGURE_DATA_PROVENANCE.csv"
    _write(tmp_path / relpath, "figure_id\nF01\n")
    expected = hashlib.sha256(b"old presentation manifest").hexdigest()
    ledger = _freeze_ledger(tmp_path / "freeze.csv", relpath, expected)
    findings, _ = delta_gate.frozen_hash_findings(tmp_path, ledger)
    assert not any(item.code == "FROZEN_HASH_CHANGED" for item in findings)


def test_blocked_semantics_scan_rejects_numeric_metric(tmp_path: Path) -> None:
    _write(
        tmp_path / "results" / "tkde_rebuild" / "figure_data" / "blocked.csv",
        "status,auprc_mean,planned_results\nRESOURCE_BLOCKED,0.314,20\n",
    )
    findings, checked = delta_gate.blocked_numeric_findings(tmp_path)
    assert checked == 1
    assert any(item.code == "BLOCKED_CELL_HAS_NUMERIC_METRIC" for item in findings)


def test_deterministic_zip_has_fixed_order_timestamps_and_crc(tmp_path: Path) -> None:
    entries = [
        release.Entry("z.txt", b"z", "z.txt", "fixture"),
        release.Entry("a.txt", b"a", "a.txt", "fixture"),
    ]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_entries, first_hash = release.deterministic_zip(first, entries, tmp_path, "test")
    _, second_hash = release.deterministic_zip(second, entries, tmp_path, "test")
    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    release.validate_zip(first, first_entries)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(info.date_time == release.FIXED_ZIP_TIME for info in archive.infolist())
        assert archive.testzip() is None


def test_release_hygiene_rejects_private_raw_and_nested_zip_members() -> None:
    with pytest.raises(RuntimeError, match="Private absolute path"):
        release.validate_entry(
            release.Entry("note.txt", b"/" + b"Users/alice/project", "note", "fixture")
        )
    with pytest.raises(RuntimeError, match="Raw/prediction payload"):
        release.validate_entry(release.Entry("data/raw/x.csv", b"x", "x", "fixture"))
    with pytest.raises(RuntimeError, match="Nested ZIP"):
        release.validate_entry(release.Entry("old.zip", b"x", "old", "fixture"))
    identity_report = b'{"name_tokens": ["Sa' + b'ket", "Ma' + b'ganti"]}'
    with pytest.raises(RuntimeError, match="Identity-bearing content"):
        release.validate_entry(release.Entry("anonymization.json", identity_report, "report", "fixture"))


def test_release_source_closure_contains_generators_and_exhaustive_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = {release.relative(path, root) for path in release.generator_input_paths(root)}
    scripts = {release.relative(path, root) for path in release.generator_script_paths(root)}
    assert set(release.EXHAUSTIVE_MACHINE_SOURCES).issubset(inputs)
    assert set(release.REQUIRED_GENERATORS).issubset(scripts)
    assert "scripts/tkde_visual_rebuild/publication_style.py" in scripts


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError, match="Unsafe extraction member"):
        release.safe_extract(archive, tmp_path / "out")
