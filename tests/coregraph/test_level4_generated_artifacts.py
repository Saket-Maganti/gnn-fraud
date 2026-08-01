from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "results" / "coregraph_build"


def _count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_level4_builder_is_idempotent_and_counts_are_exact() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/coregraph/build_level4_artifacts.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["base_artifacts"] == 180
    assert _count(BUILD / "V5_BASE_ARTIFACTS.csv") == 180
    assert _count(BUILD / "V5_SCENARIOS.csv") == 60
    assert _count(BUILD / "V5_BINDINGS.csv") == 540
    assert _count(BUILD / "EVIDENCE_CACHE_MANIFEST.csv") == 366
    assert _count(BUILD / "LEVEL4_FULL_RUN_MATRIX.csv") > 1000


def test_generated_gates_are_truthful_and_private_path_free() -> None:
    leakage = json.loads((BUILD / "V5_LEAKAGE_AUDIT.json").read_text())
    assert leakage["structural_status"] == "PASS"
    assert leakage["row_scope_status"] == "PASS_20_DATASET_SEED_GROUPS"
    assert leakage["overall_status"] == "PASS_NO_TRAINING_BYTE_AND_STRUCTURE"
    validation = json.loads((BUILD / "ARCHIVE_MEMBER_VALIDATION.json").read_text())
    assert validation["archive_present"] == 6 and validation["fabricated_hashes"] == 0
    assert validation["member_checksum_verified"] == 180
    prereg = (BUILD / "LEVEL4_PREREGISTRATION_HASH.txt").read_text()
    assert len(prereg.splitlines()[0].split()[1]) == 64
    tracked_level4 = "\n".join(
        path.read_text(errors="ignore")
        for path in list((ROOT / "docs" / "coregraph").glob("LEVEL4*"))
        + list(BUILD.glob("LEVEL4*"))
        if path.is_file()
    )
    assert "/" + "Users/" not in tracked_level4
    assert "/" + "Volumes/" not in tracked_level4


def test_final_handoff_and_nine_prompts_are_generated_without_execution() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/coregraph/build_level4_handoff.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["prompts"] == 9
    assert payload["verdict"] == (
        "COREGRAPH_V5_EXECUTOR_IMPLEMENTED_REAL_PILOT_UNEXECUTED"
    )
    prompts = sorted((BUILD / "LEVEL4_NEXT_EXECUTION_PROMPTS").glob("*.md"))
    assert len(prompts) == 9
    for prompt in prompts:
        text = prompt.read_text(encoding="utf-8")
        compact = " ".join(text.split())
        assert "Never fabricate metrics" in compact
        assert "Do not use the SSD when the local cache" in compact
        assert "Never force-push or merge PR #2" in compact
    gate = json.loads((BUILD / "LEVEL4_FINAL_GATE_STATUS.json").read_text())
    assert gate["ready_for_saved_output_pilot"] is True
    assert gate["real_pilot_executed"] is False
    assert gate["v5"]["executor"] == "IMPLEMENTED_AND_SYNTHETICALLY_VALIDATED"
    assert gate["prohibited_actions"]["target_metric_computation"] == 0
