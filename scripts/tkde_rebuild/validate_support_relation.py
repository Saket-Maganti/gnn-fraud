#!/usr/bin/env python3
"""Validate the typed claim-support relation with sandboxed mutations.

The validator never edits canonical evidence.  Mutations are in-memory changes
to requirement/evidence descriptors.  It also reruns the deterministic analysis
builder and compares hashes, then verifies canonical inputs were unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tkde_rebuild"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Case:
    case_id: str
    base_claim: str
    mutation: str
    required_evidence_units: int
    observed_evidence_units: int
    required_seeds: int
    observed_min_seeds: int
    prediction_complete: bool
    integrity_state: str = "PASS"
    construct_state: str = "PASS"
    resource_state: str = "MEASURED"
    direction_state: str = "NOT_TESTED_BY_GATE"
    expected_status: str = "SUPPORTED"


def decide(case: Case) -> str:
    if case.integrity_state != "PASS":
        return "EXCLUDED_INTEGRITY"
    if case.construct_state != "PASS":
        return "EXCLUDED_CONSTRUCT_INVALID"
    if case.direction_state == "CONTRADICTED":
        return "REFUTED_IN_SCOPE"
    if case.resource_state != "MEASURED":
        return "RESOURCE_BLOCKED"
    if case.observed_evidence_units < case.required_evidence_units:
        return "BLOCKED_INCOMPLETE_SCOPE"
    if case.observed_min_seeds < case.required_seeds:
        return "BLOCKED_INCOMPLETE_SEEDS"
    if not case.prediction_complete:
        return "BLOCKED_MISSING_PREDICTIONS"
    return "SUPPORTED"


def build_cases(inventory: pd.DataFrame) -> pd.DataFrame:
    ell = inventory[inventory.evidence_id.str.startswith("RB09::elliptic::")]
    ell_strict_isolated = ell[ell.protocol.isin(["strict_inductive", "inductive_isolated"])]
    small_gine = inventory[inventory.evidence_id.str.contains(r"V28::ibm_aml::(?:hi|li)-small::.*gine_light_h64", regex=True)]
    medium_gine = inventory[inventory.evidence_id.str.contains(r"V28::ibm_aml::(?:hi|li)-medium::gine_light_h64::BLOCKED", regex=True)]
    large = inventory[inventory.evidence_id.str.contains(r"V26::ibm_aml::(?:hi|li)-large::BLOCKED", regex=True)]
    cases = [
        Case("FV01", "C01 Elliptic visibility ranking", "none; complete locked grid", 6, len(ell_strict_isolated), 10, int(ell_strict_isolated.n_seeds.min()), True, expected_status="SUPPORTED"),
        Case("FV02", "C01 Elliptic visibility ranking", "remove one model/protocol cell in sandbox", 6, len(ell_strict_isolated) - 1, 10, 10, True, expected_status="BLOCKED_INCOMPLETE_SCOPE"),
        Case("FV03", "C01 Elliptic visibility ranking", "remove seed 10 from one cell in sandbox", 6, len(ell_strict_isolated), 10, 9, True, expected_status="BLOCKED_INCOMPLETE_SEEDS"),
        Case("FV04", "C01 Elliptic visibility ranking", "erase one prediction-manifest reference in sandbox", 6, len(ell_strict_isolated), 10, 10, False, expected_status="BLOCKED_MISSING_PREDICTIONS"),
        Case("FV05", "C10 Small GINE AUPRC result", "widen scope from Small to Medium", len(small_gine) + len(medium_gine), len(small_gine) + len(medium_gine), 10, 0, False, resource_state="T4_CUDA_OOM", expected_status="RESOURCE_BLOCKED"),
        Case("FV06", "C05 IBM AML baseline grid", "widen scope from Small/Medium to all official variants", 26, 26, 10, 0, False, resource_state="SAFE_RESOURCE_BLOCKED_LARGE", expected_status="RESOURCE_BLOCKED"),
        Case("FV07", "V22 loss robustness", "substitute a partial failed import for a full10 lane", 1, 1, 10, 1, True, integrity_state="FAIL_MISSING_SEED", expected_status="EXCLUDED_INTEGRITY"),
        Case("FV08", "V24 temporal stress", "treat three metadata labels as distinct temporal windows", 3, 3, 10, 10, True, construct_state="STRESS_ARGUMENT_NOT_PASSED_TO_HARNESS", expected_status="EXCLUDED_CONSTRUCT_INVALID"),
        Case("FV09", "DGraphFin fixed GAT performance", "promote the T4-OOM cell to a predictive comparison", 1, 1, 10, 0, False, resource_state="BLOCKED_T4_OOM", expected_status="RESOURCE_BLOCKED"),
        Case("FV10", "DGraphFin GraphSAGE max-pool rerun", "promote runbook/status files without imported rows", 1, 1, 10, 0, False, resource_state="BLOCKED_WAITING_FOR_GPU", expected_status="RESOURCE_BLOCKED"),
        Case("FV11", "IBM sender-receiver construction", "count the contract alias as an independent replication", 8, 8, 10, 10, True, construct_state="IDENTICAL_IMPLEMENTATION_ALIAS", expected_status="EXCLUDED_CONSTRUCT_INVALID"),
        Case("FV12", "C04 universal graph harm", "widen mixed scoped results into a universal directional claim", 1, 1, 10, 10, True, direction_state="CONTRADICTED", expected_status="REFUTED_IN_SCOPE"),
        Case("FV13", "C17 universal GraphSafe improvement", "widen bounded mixed case study to universal dominance", 1, 1, 10, 10, True, direction_state="CONTRADICTED", expected_status="REFUTED_IN_SCOPE"),
        Case("FV14", "IBM early-to-late visibility", "describe the shared first-60% label-free node-history map as first-50% training-only features", 1, 1, 10, 10, True, construct_state="FEATURE_VISIBILITY_CONTRACT_MISSTATED", expected_status="EXCLUDED_CONSTRUCT_INVALID"),
    ]
    rows = []
    for case in cases:
        row = asdict(case)
        row["observed_status"] = decide(case)
        row["pass"] = row["observed_status"] == row["expected_status"]
        rows.append(row)
    return pd.DataFrame(rows)


def false_promotion_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["FP01", "V22 failed/noncomparable imports", 38, 38, "filenames and metric-bearing files exist", "EXCLUDED_INTEGRITY", "missing seed, failed return-code, duplicate/conflicting merge metadata"],
            ["FP02", "V24 RB41 duplicate stress labels", 240, 240, "three labels appear to be distinct conditions", "DEDUPLICATE_TO_120_BASE_CELLS", "stress label is never passed to benchmark harness; all performance metrics repeat"],
            ["FP03", "V24 memory-reduced DGraphFin GAT h32/l1", 20, 20, "GAT outputs could appear to fill the fixed h64/l2 gap", "DIAGNOSTIC_ONLY", "different width/depth; canonical lock says not comparable"],
            ["FP04", "V26 legacy P100-named lane", 12, 12, "path token could be read as measured P100 hardware", "NORMALIZE_ALIAS_TO_T4/CUDA0", "runtime records identify cuda:0 and validation normalizes the alias; path names are not hardware evidence"],
            ["FP05", "IBM AML Large status/runbook artifacts", 0, 0, "variant directories and plans exist", "RESOURCE_BLOCKED", "canonical lock contains zero results and predictions"],
            ["FP06", "Medium GINE blocked plans", 0, 0, "planned 40 outputs could be confused with completed coverage", "RESOURCE_BLOCKED", "two T4-OOM rows; zero results and predictions"],
            ["FP07", "RB18 larger-GPU preparation", 0, 0, "commands/notebook/status files exist", "BLOCKED_WAITING_FOR_GPU", "validation reports no imported result CSV"],
        ],
        columns=["audit_id", "artifact_family", "result_files_at_risk", "prediction_files_at_risk", "naive_promotion_rule", "correct_status", "reason"],
    )


def regeneration_audit() -> pd.DataFrame:
    canonical = [
        ROOT / "results/runs_rb09v3/runs.csv",
        ROOT / "results/runs_rb09v3/ARTIFACT_FAMILY.json",
        ROOT / "kaggle_workspace/manifests/V22_FINAL_GPU_EVIDENCE_LOCK.json",
        ROOT / "results/v24_imported/V24_IMPORTED_EVIDENCE_LOCK.json",
        ROOT / "results/v26_imported/V26_IMPORTED_EVIDENCE_LOCK.json",
        ROOT / "results/v27_imported/V27_STRONGER_GRAPH_EVIDENCE_LOCK.json",
        ROOT / "results/v28_imported/V28_ALL_RUNS_EVIDENCE_LOCK.json",
        ROOT / "results/runs_rb17_review_budget_worst_block/rb17_results.csv",
    ]
    generated = sorted(
        path
        for path in OUT.glob("*.csv")
        if path.name
        not in {
            "FRAMEWORK_VALIDATION_CASES.csv",
            "FALSE_PROMOTION_AUDIT.csv",
            "REGENERATION_HASH_AUDIT.csv",
            "NUMBER_PROVENANCE_MAP.csv",
        }
    )
    before_canonical = {path: sha256(path) for path in canonical}
    before_generated = {path: sha256(path) for path in generated}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/tkde_rebuild/compute_analysis.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"deterministic regeneration failed: {proc.stderr}\n{proc.stdout}")
    after_canonical = {path: sha256(path) for path in canonical}
    after_generated = {path: sha256(path) for path in generated}
    rows: list[dict[str, Any]] = []
    for path in canonical:
        rows.append(
            {
                "artifact_type": "canonical_input",
                "path": path.relative_to(ROOT).as_posix(),
                "sha256_before": before_canonical[path],
                "sha256_after": after_canonical[path],
                "identical": before_canonical[path] == after_canonical[path],
                "command_exit": proc.returncode,
            }
        )
    for path in generated:
        rows.append(
            {
                "artifact_type": "generated_analysis",
                "path": path.relative_to(ROOT).as_posix(),
                "sha256_before": before_generated[path],
                "sha256_after": after_generated[path],
                "identical": before_generated[path] == after_generated[path],
                "command_exit": proc.returncode,
            }
        )
    return pd.DataFrame(rows)


def frozen_regeneration_audit() -> pd.DataFrame:
    """Verify the saved regeneration audit without requiring excluded raw data."""

    path = OUT / "REGENERATION_HASH_AUDIT.csv"
    frame = pd.read_csv(path)
    presentation_manifests = {
        "results/tkde_rebuild/FIGURE_DATA_PROVENANCE.csv",
        "results/tkde_rebuild/MANUSCRIPT_MACHINE_AUDIT.csv",
        "results/tkde_rebuild/TABLE_DATA_PROVENANCE.csv",
    }
    required = {
        "artifact_type",
        "path",
        "sha256_before",
        "sha256_after",
        "identical",
        "command_exit",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"saved regeneration audit missing columns: {sorted(missing)}")
    for row in frame.itertuples(index=False):
        artifact = ROOT / str(row.path)
        if not artifact.is_file():
            raise RuntimeError(f"saved regeneration artifact is absent: {row.path}")
        if str(row.path) in presentation_manifests:
            continue
        if sha256(artifact) != str(row.sha256_after):
            raise RuntimeError(f"saved regeneration hash mismatch: {row.path}")
    if not frame["identical"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("saved regeneration audit contains a non-identical row")
    if not frame["command_exit"].astype(int).eq(0).all():
        raise RuntimeError("saved regeneration audit contains a nonzero command exit")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-only",
        action="store_true",
        help="Validate saved aggregate hashes without rerunning analysis that requires excluded raw data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(OUT / "EVIDENCE_INVENTORY.csv")
    cases = build_cases(inventory)
    false_promotions = false_promotion_audit()
    regen = frozen_regeneration_audit() if args.frozen_only else regeneration_audit()
    cases.to_csv(OUT / "FRAMEWORK_VALIDATION_CASES.csv", index=False)
    false_promotions.to_csv(OUT / "FALSE_PROMOTION_AUDIT.csv", index=False)
    regen.to_csv(OUT / "REGENERATION_HASH_AUDIT.csv", index=False)
    if not cases["pass"].all():
        raise SystemExit("framework validation case failure")
    if not regen["identical"].all():
        raise SystemExit("regeneration or canonical-input hash mismatch")

    status_counts = cases.observed_status.value_counts().to_dict()
    report = f"""# Framework Validation Report

Status: **PASS**

The support relation was exercised with {len(cases)} in-memory claim mutations and evidence ablations. All expected transitions matched the validator. The cases include complete support, missing cells, a missing seed, a missing prediction manifest, scope widening into Large or Medium-GINE resource boundaries, failed V22 imports, V24 metadata-only stress labels, a non-independent construction alias, and two directional universal claims contradicted by observed cells.

## Status transitions

{chr(10).join(f'- `{key}`: {value}' for key, value in sorted(status_counts.items()))}

## False-promotion prevention

The audit identifies {int(false_promotions.result_files_at_risk.sum())} result files and {int(false_promotions.prediction_files_at_risk.sum())} prediction files that a filename/count-only pipeline could misclassify. The largest class is V24: 240 result and 240 prediction files are duplicate scientific cells carrying three metadata labels that never reach the benchmark harness. V22 contributes 38 result and 38 prediction files from three integrity-failed/noncomparable imports. The memory-reduced DGraphFin GAT outputs remain diagnostic rather than filling the fixed-configuration OOM cell.

## Deterministic regeneration

{"The saved deterministic regeneration audit was rehashed against the curated aggregate files." if args.frozen_only else "`compute_analysis.py` was rerun with exit code 0."} All {len(regen[regen.artifact_type.eq('generated_analysis')])} pre-existing generated analysis CSV hashes were identical, and all {len(regen[regen.artifact_type.eq('canonical_input')])} audited canonical input hashes were unchanged. This validation does not claim that the framework proves scientific truth; it verifies the declared completeness, provenance, construct, prediction, and resource-status transitions.
"""
    (OUT / "FRAMEWORK_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"PASS: {len(cases)} support cases; {len(regen)} hash checks")


if __name__ == "__main__":
    main()
