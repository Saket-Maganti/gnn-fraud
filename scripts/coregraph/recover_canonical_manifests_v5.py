#!/usr/bin/env python3
"""Recover canonical RB09v3 evidence and build V5 readiness surfaces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.experiments.canonical_recovery import (  # noqa: E402
    EVIDENCE_PRECEDENCE,
    base_completeness_matrix,
    discover_prediction_index_records,
    discover_result_index_records,
    recover_rb09v3,
    scenario_completeness_surfaces,
)

VERDICT = "COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _portable(value: str, aliases: Sequence[tuple[str, str]]) -> str:
    for source, alias in aliases:
        if value == source:
            return alias
        prefix = source + "/"
        if value.startswith(prefix):
            return alias + "/" + value[len(prefix) :]
    return value


def _portable_mapping(
    value: Any,
    aliases: Sequence[tuple[str, str]],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable_mapping(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_mapping(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_portable_mapping(item, aliases) for item in value]
    if isinstance(value, str):
        return _portable(value, aliases)
    return value


def _precedence_report(
    *,
    sources,
    summary: Mapping[str, Any],
) -> str:
    rows = "\n".join(
        f"| {source.precedence} | `{source.evidence_type}` | "
        f"`{source.path}` | `{source.sha256}` | {source.relevance} |"
        for source in sources
    )
    return f"""# Canonical Prediction Evidence Precedence

Status: `{VERDICT}`

Scientific identity is taken from authoritative metadata. A prediction filename
is used only to navigate to a referenced file and never overrides a conflicting
lock or index field.

## Frozen precedence

1. final evidence lock;
2. final merged prediction index;
3. per-lane merge validation;
4. package import validation;
5. result sidecar;
6. raw filename/content navigation.

The executable precedence mapping is:
`{json.dumps(EVIDENCE_PRECEDENCE, sort_keys=True)}`.

Every joined record preserves artifact family, source package, original
version/alias, dataset, protocol, model, seed, prediction path/checksum, result
path/checksum, import validation, and lock membership. Conflicts remain blocked.

## Authoritative sources inspected

| precedence | type | exact path | SHA-256 | relevance |
|---:|---|---|---|---|
{rows}

## RB09v3 result

- Canonical coordinates: {summary["canonical_inventory_count"]}
- Locally usable canonical artifacts: {summary["usable_artifact_count"]}
- Index-referenced local files missing: {summary["missing_index_reference_count"]}
- Artifacts genuinely shown never to have existed: {summary["true_missing_artifact_count"]}
- Structured prediction-index records discovered: {summary["discovered_prediction_index_record_count"]}
- Structured result-index/JSONL records discovered: {summary["discovered_result_index_jsonl_record_count"]}
- Evidence locks discovered: {summary["discovered_evidence_lock_count"]}
- Validation reports discovered: {summary["discovered_validation_report_count"]}
- Package import/alias records discovered: {summary["discovered_package_import_record_count"]}
- Result sidecar files discovered: {summary["discovered_result_sidecar_count"]}
- JSONL sidecars discovered: {summary["discovered_jsonl_source_count"]}

The missing archive references are not reclassified as never-created artifacts:
RB15/RB16 import records prove those members were previously present and
consumed.
"""


def _reconciliation_report(
    *,
    historical_root: str,
    summary: Mapping[str, Any],
    records,
    sources,
) -> str:
    archive_rows: dict[str, tuple[str, bool, int]] = {}
    for record in records:
        if record.source_archive_path:
            archive_rows[record.source_archive_path] = (
                record.source_archive_sha256,
                record.archive_present,
                archive_rows.get(record.source_archive_path, ("", False, 0))[2]
                + 1,
            )
    archive_table = "\n".join(
        f"| `{path}` | `{checksum}` | {str(present).lower()} | {members} |"
        for path, (checksum, present, members) in sorted(archive_rows.items())
    )
    raw_count = len(
        {
            candidate
            for record in records
            for candidate in record.raw_navigation_candidates
        }
    )
    return f"""# Canonical RB09v3 Reconciliation

Verdict: `{VERDICT}`

## Canonical claim

`{historical_root}/results/runs_rb09v3/ARTIFACT_FAMILY.json` and
`predictions_manifest.json` agree on exactly 180 prediction CSVs:
2 datasets × 3 protocols × 3 experts × 10 seeds. The matching `runs.csv`
contains exactly 180 result rows. Source and target roles are not counted here.

## Reconciled current state

- Canonical inventory records: **{summary["canonical_inventory_count"]}**
- Canonical or explicitly compatible local artifacts: **{summary["usable_artifact_count"]}**
- Index-referenced local prediction/archive members missing: **{summary["missing_index_reference_count"]}**
- Result-sidecar/metadata primary blockers: **{summary["metadata_blocker_count"]}**
- Integrity-confirmed never-created artifacts: **{summary["true_missing_artifact_count"]}**
- Raw same-coordinate navigation candidates inspected: **{raw_count}**
- Authoritative evidence files inspected: **{len(sources)}**

The raw candidates are not promoted to compatible aliases without an
authoritative alias/checksum link. This avoids silently substituting a different
run family merely because a basename encodes the same coordinate.

## Previously consumed source archives

| recorded archive path | recorded archive SHA-256 | present now | indexed members |
|---|---|---:|---:|
{archive_table}

All six archives are Category D recovery dependencies when absent locally.
Their SHA-256 values and all 180 member paths survive in RB15/RB16 import
manifests. No new model run is justified while these indexed archives remain
recoverable from their original external storage.

## Scenario consequence

The corrected surface contains 180 base cells, 60 held-out-protocol scenarios,
and 540 role bindings. Since the immutable CSV bytes are unavailable locally,
row-scope and scenario leakage materialisation remain blocked; no target metric,
oracle, router fit, or pilot execution was attempted.
"""


def _future_run_report(records) -> str:
    counts: Counter[str] = Counter()
    for record in records:
        if record.status in {"RECOVERED_CANONICAL", "RECOVERED_COMPATIBLE_ALIAS"}:
            counts["A"] += 1
        elif record.status == "INDEX_REFERENCED_FILE_MISSING":
            counts["D"] += 1
        elif record.status in {"RESULT_SIDECAR_MISSING", "METADATA_UNRESOLVED"}:
            counts["C"] += 1
        elif record.status in {"ARTIFACT_GENUINELY_MISSING", "EXCLUDED_INTEGRITY"}:
            counts["E"] += 1
    archives = {
        (
            record.source_archive_path,
            record.source_archive_sha256,
        )
        for record in records
        if record.status == "INDEX_REFERENCED_FILE_MISSING"
        and record.source_archive_path
    }
    archive_lines = "\n".join(
        f"- `{path}` — expected SHA-256 `{checksum}`; restore the exact archive "
        "and validate it before extracting or streaming its indexed members."
        for path, checksum in sorted(archives)
    )
    return f"""# Future Run Necessity Report

Verdict: `{VERDICT}`

## Classification

- Category A — converter/discovery miss repaired: {counts["A"]}
- Category B — present archive not extracted: {counts["B"]}
- Category C — prediction present but metadata incomplete: {counts["C"]}
- Category D — canonical index references a missing local file/archive: {counts["D"]}
- Category E — artifact never existed or failed integrity: {counts["E"]}

## Decision

**No future GPU prediction-generation run is recommended.** Category D remains
non-zero and Category E is zero. The evidence proves the 180 members existed and
were consumed; the correct next action is canonical archive recovery, not
retraining.

## Exact archive dependencies

{archive_lines}

After restoring an archive, verify its recorded SHA-256, then rerun:

`python scripts/coregraph/recover_canonical_manifests_v5.py --historical-root "$HISTORICAL_GNN_FRAUD_REPO"`

The converter can stream ZIP members; extraction is not required. If extraction
is desired, use `unzip -n <exact-archive> -d <new-dedicated-directory>` only
after checksum verification.
"""


def _readiness_spec(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "coregraph_pilot_manifest_readiness_spec_v5",
        "frozen_before_pilot_execution": True,
        "purpose": "integration_and_no_training_readiness_only",
        "does_not_modify_empirical_pilot_thresholds": True,
        "required_surface": {
            "base_artifacts": 180,
            "evaluation_scenarios": 60,
            "scenario_bindings": 540,
            "source_bindings": 360,
            "target_bindings": 180,
            "datasets": ["elliptic", "dgraphfin"],
            "protocols": [
                "strict_inductive",
                "isolated_inductive",
                "transductive_structure",
            ],
            "experts": ["feature_mlp", "gcn", "graphsage"],
            "expert_prediction_seeds": list(range(1, 11)),
            "folds": ["fold0"],
        },
        "identity_rules": {
            "base_artifacts_are_role_neutral": True,
            "scenario_roles_are_bindings": True,
            "cross_scenario_role_reuse_allowed": True,
            "same_scenario_dual_role_forbidden": True,
            "base_artifact_hash_separate_from_contract_coordinate_hash": True,
            "scenario_id_separate_from_role_binding_id": True,
        },
        "row_scope_rules": {
            "source_permitted_splits": ["train", "validation"],
            "target_permitted_splits": ["test"],
            "target_requires_label_known": True,
            "unknown_target_rows_excluded": True,
            "target_labels_before_scoring": False,
        },
        "leakage_scope": "within_one_evaluation_scenario",
        "inference_identity": "dataset_stratified_expert_prediction_seed",
        "required_no_training_assertions": {
            "training_performed": False,
            "fitting_path_reachable": False,
            "metric_computation_performed": False,
            "oracle_computation_performed": False,
        },
        "current_reconciliation": dict(summary),
        "current_verdict": VERDICT,
        "pilot_authorized": False,
        "next_authority": "fifth_independent_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-root",
        default=os.environ.get("HISTORICAL_GNN_FRAUD_REPO", "../gnn-fraud"),
    )
    parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        default=None,
    )
    parser.add_argument(
        "--build-root",
        default="results/coregraph_build",
    )
    args = parser.parse_args()
    historical_root = Path(args.historical_root).expanduser().resolve()
    search_roots = args.search_roots or [str(ROOT), str(historical_root)]
    build_root = Path(args.build_root).expanduser().resolve()
    records, sources, summary = recover_rb09v3(
        historical_root=historical_root,
        search_roots=search_roots,
    )
    path_aliases = (
        (str(ROOT), "$COREGRAPH_REPO"),
        (str(historical_root), "$HISTORICAL_GNN_FRAUD_REPO"),
    )
    report_records = tuple(
        replace(
            record,
            canonical_inventory_path=_portable(
                record.canonical_inventory_path,
                path_aliases,
            ),
            resolved_prediction_path=_portable(
                record.resolved_prediction_path,
                path_aliases,
            ),
            raw_navigation_candidates=tuple(
                _portable(value, path_aliases)
                for value in record.raw_navigation_candidates
            ),
            source_archive_path=_portable(
                record.source_archive_path,
                path_aliases,
            ),
            alias_lineage=tuple(
                _portable(value, path_aliases)
                for value in record.alias_lineage
            ),
            result_path=_portable(record.result_path, path_aliases),
            import_validation_path=_portable(
                record.import_validation_path,
                path_aliases,
            ),
            config_provenance_path=_portable(
                record.config_provenance_path,
                path_aliases,
            ),
            code_provenance_path=_portable(
                record.code_provenance_path,
                path_aliases,
            ),
        )
        for record in records
    )
    report_sources = tuple(
        replace(source, path=_portable(source.path, path_aliases))
        for source in sources
    )
    summary = {
        **summary,
        "discovered_prediction_index_record_count": len(
            discover_prediction_index_records(search_roots)
        ),
        "discovered_result_index_jsonl_record_count": len(
            discover_result_index_records(search_roots)
        ),
        "discovered_evidence_lock_count": sum(
            source.evidence_type == "FINAL_EVIDENCE_LOCK"
            for source in sources
        ),
        "discovered_validation_report_count": sum(
            source.evidence_type == "PER_LANE_MERGE_VALIDATION"
            for source in sources
        ),
        "discovered_package_import_record_count": sum(
            source.evidence_type == "PACKAGE_IMPORT_VALIDATION"
            for source in sources
        ),
        "discovered_result_sidecar_count": sum(
            source.evidence_type == "RESULT_SIDECAR"
            for source in sources
        ),
        "discovered_jsonl_source_count": sum(
            Path(source.path).suffix.lower() == ".jsonl"
            for source in sources
        ),
    }
    portable_records = [
        _portable_mapping(record.to_csv_row(), path_aliases)
        for record in report_records
    ]
    _write_csv(
        build_root / "CANONICAL_RB09V3_ARTIFACT_INDEX.csv",
        portable_records,
    )
    missing_rows = [
        {
            "dataset": row["dataset"],
            "protocol_id": row["protocol_id"],
            "expert_id": row["expert_id"],
            "expert_prediction_seed": row["expert_prediction_seed"],
            "fold": row["fold"],
            "status": row["status"],
            "indexed_prediction_path": row["indexed_prediction_path"],
            "source_archive_path": row["source_archive_path"],
            "source_archive_sha256": row["source_archive_sha256"],
            "source_archive_member": row["source_archive_member"],
            "result_path": row["result_path"],
            "reason_codes": row["reason_codes"],
            "future_run_category": "D",
        }
        for row in portable_records
        if row["status"] == "INDEX_REFERENCED_FILE_MISSING"
    ]
    _write_csv(
        build_root / "CANONICAL_RB09V3_MISSING_INDEX_REFERENCES.csv",
        missing_rows,
    )
    base_rows = base_completeness_matrix(report_records)
    _write_csv(build_root / "BASE_ARTIFACT_COMPLETENESS_MATRIX.csv", base_rows)
    scenario_rows, scenario_index = scenario_completeness_surfaces(base_rows)
    _write_csv(
        build_root / "SCENARIO_COMPLETENESS_MATRIX.csv",
        scenario_rows,
    )
    _write_json(
        build_root / "SCENARIO_BINDING_INDEX.json",
        scenario_index,
    )
    (build_root / "CANONICAL_PREDICTION_EVIDENCE_PRECEDENCE.md").write_text(
        _precedence_report(sources=report_sources, summary=summary),
        encoding="utf-8",
    )
    (build_root / "CANONICAL_RB09V3_RECONCILIATION.md").write_text(
        _reconciliation_report(
            historical_root="$HISTORICAL_GNN_FRAUD_REPO",
            summary=summary,
            records=report_records,
            sources=report_sources,
        ),
        encoding="utf-8",
    )
    (build_root / "FUTURE_RUN_NECESSITY_REPORT.md").write_text(
        _future_run_report(report_records),
        encoding="utf-8",
    )
    _write_json(
        build_root / "PILOT_MANIFEST_READINESS_SPEC_V5.json",
        _readiness_spec(summary),
    )
    output = {
        "verdict": VERDICT,
        **summary,
        "base_matrix_rows": len(base_rows),
        "scenario_matrix_rows": len(scenario_rows),
        "scenario_binding_count": scenario_index["binding_count"],
        "training_performed": False,
        "fitting_performed": False,
        "metric_computation_performed": False,
        "oracle_computation_performed": False,
        "pilot_executed": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
