from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "audit_cross_paper_overlap.py"
    spec = importlib.util.spec_from_file_location("overlap_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlap_audit_detects_copied_prose_and_visual(tmp_path: Path) -> None:
    module = _module()
    tkde = tmp_path / "tkde"
    iclr = tmp_path / "iclr"
    tkde.mkdir()
    iclr.mkdir()
    copied = " ".join(f"distinctword{index}" for index in range(50)) + "."
    (tkde / "paper.tex").write_text(copied, encoding="utf-8")
    (iclr / "paper.tex").write_text(copied, encoding="utf-8")
    (tkde / "figure.png").write_bytes(b"same-visual")
    (iclr / "figure.png").write_bytes(b"same-visual")
    report = module.audit(tkde, iclr)
    assert report["status"] == "FAIL"
    assert report["measurements"]["byte_identical_visual_asset_count"] == 1


def test_overlap_audit_accepts_distinct_tiny_corpora(tmp_path: Path) -> None:
    module = _module()
    tkde = tmp_path / "tkde"
    iclr = tmp_path / "iclr"
    tkde.mkdir()
    iclr.mkdir()
    (tkde / "paper.tex").write_text("benchmark evidence protocol validity", encoding="utf-8")
    (iclr / "paper.tex").write_text("router contracts regret composition", encoding="utf-8")
    report = module.audit(tkde, iclr)
    assert report["status"] == "PASS"
