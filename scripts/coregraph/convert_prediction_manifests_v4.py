#!/usr/bin/env python3
"""Convert only fully evidenced historical predictions into V4 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.experiments.manifest_conversion import (  # noqa: E402
    audit_candidates,
    build_blocked_leakage_report,
    build_completeness_matrix,
    build_conversion_records,
    discover_historical_predictions,
    discover_validation_evidence,
    status_counts,
    validate_converted_bindings,
)
from coregraph.experiments.pilot import load_prediction_artifacts  # noqa: E402
from coregraph.experiments.protocol_registry import (  # noqa: E402
    load_protocol_registry,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _portable_path_aliases(
    roots: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    aliases = []
    for index, value in enumerate(roots, start=1):
        resolved = str(Path(value).expanduser().resolve())
        if Path(resolved) == ROOT:
            alias = "$COREGRAPH_REPO"
        elif Path(resolved).name == "gnn-fraud":
            alias = "$HISTORICAL_GNN_FRAUD_REPO"
        else:
            alias = f"$DISCOVERY_ROOT_{index}"
        aliases.append((resolved, alias))
    aliases.append((str(ROOT), "$COREGRAPH_REPO"))
    return tuple(sorted(set(aliases), key=lambda item: -len(item[0])))


def _portable_payload(
    value: Any,
    aliases: Sequence[tuple[str, str]],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable_payload(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_payload(item, aliases) for item in value]
    if isinstance(value, tuple):
        return [_portable_payload(item, aliases) for item in value]
    if isinstance(value, str):
        for root, alias in aliases:
            if value == root:
                return alias
            prefix = root + "/"
            if value.startswith(prefix):
                return alias + "/" + value[len(prefix):]
    return value


def _write_matrix(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "protocol_id",
        "expert_prediction_seed",
        "fold",
        "expert_id",
        "contract_role",
        "status",
        "reason_codes",
        "candidate_count",
        "unique_candidate_checksums",
        "original_path",
        "original_checksum",
        "contract_coordinate_hash",
        "contract_id",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_evidence_map(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("conversion evidence map must be a JSON object")
    return payload


def _verdict(matrix: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(row["status"]) for row in matrix}
    if "BLOCKED_METADATA_UNRESOLVED" in statuses:
        return "COREGRAPH_V4_MANIFEST_CONVERSION_BLOCKED_METADATA_UNRESOLVED"
    if any(status.startswith("BLOCKED_") for status in statuses):
        return "COREGRAPH_V4_MANIFEST_CONVERSION_BLOCKED_METADATA_UNRESOLVED"
    if "MISSING_ARTIFACT" in statuses:
        return "COREGRAPH_V4_MANIFEST_CONVERSION_PARTIAL_BLOCKED_MISSING_ARTIFACTS"
    return "COREGRAPH_V4_MANIFESTS_COMPLETE_READY_FOR_PILOT_EXECUTION_REVIEW"


def _report_markdown(
    *,
    roots: Sequence[str],
    candidate_count: int,
    validation_report_count: int,
    validated_candidate_count: int,
    structurally_usable_candidate_count: int,
    conversion_counts: Mapping[str, int],
    matrix_counts: Mapping[str, int],
    registry_audit: Mapping[str, Any],
    verdict: str,
) -> str:
    return f"""# V4 manifest conversion report

Status: `{verdict}`

This was a read-only historical-artifact conversion and no-training audit.
No router or learned baseline was fitted, no target metric or oracle was
computed, and no historical prediction file was modified.

## Discovery

- Search roots: {", ".join(f"`{value}`" for value in roots)}
- Requested-pattern candidates: {candidate_count}
- Prediction-validation reports inspected: {validation_report_count}
- Candidates with validation evidence: {validated_candidate_count}
- Structurally usable validated candidates: {structurally_usable_candidate_count}
- Conversion statuses: `{json.dumps(dict(conversion_counts), sort_keys=True)}`
- Expected-cell statuses: `{json.dumps(dict(matrix_counts), sort_keys=True)}`
- Frozen registry schema: `{registry_audit["registry_schema_status"]}`
- Complete contract binding audit: `{registry_audit["contract_binding_status"]}`

Candidates become loadable V4 manifests only when the original checksum,
full deployment contract, coordinate hash, complete contract ID, role,
config/code hashes, compute cost and its provenance are all evidenced.
Anything unresolved remains `BLOCKED_METADATA_UNRESOLVED`.

## Scope

The completeness matrix covers both artifact roles for every combination of
two datasets, three frozen protocol aliases, three experts, seeds 1--10 and
`fold0`. Split, `label_known`, score-domain, duplicate-ID and timestamp fields
were audited where present. Contract-registry and typed cross-role leakage
aliases were validated against the frozen registry; complete
coordinate/contract binding and typed cross-role leakage remain blocked for
cells without loadable V4 manifests.
No-training runner materialisation and gate completeness are likewise blocked
until the exact matrix is available.

The verdict is a manifest-conversion/readiness verdict only. It does not
authorize pilot execution.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=None,
        help="Read-only historical discovery root; repeatable.",
    )
    parser.add_argument(
        "--evidence-map",
        help="Optional explicit V4 metadata evidence map.",
    )
    parser.add_argument(
        "--output-root",
        default="results/coregraph_manifest_conversion_v4",
    )
    parser.add_argument(
        "--build-root",
        default="results/coregraph_build",
    )
    parser.add_argument(
        "--protocol-registry",
        default="results/coregraph_build/CONTRACT_PROTOCOL_REGISTRY_V4.json",
    )
    args = parser.parse_args()
    roots = args.roots or [
        str(ROOT),
        str(ROOT.parent / "gnn-fraud"),
    ]
    path_aliases = _portable_path_aliases(roots)
    output_root = Path(args.output_root).resolve()
    build_root = Path(args.build_root).resolve()
    registry_path = Path(args.protocol_registry).resolve()
    registry = load_protocol_registry(registry_path)
    candidates = discover_historical_predictions(roots)
    validation_reports = discover_validation_evidence(roots)
    audits = audit_candidates(candidates, validation_reports)
    evidence_map = _load_evidence_map(args.evidence_map)
    records = build_conversion_records(audits, evidence_map)
    matrix = build_completeness_matrix(records)
    leakage = build_blocked_leakage_report(matrix)
    manifests: list[str] = []
    manifest_root = output_root / "manifests"
    for record in records:
        manifest = record["manifest"]
        if manifest is None:
            continue
        name = (
            f"{manifest['dataset']}__{manifest['protocol_id']}__"
            f"{manifest['contract_role']}__{manifest['expert_id']}__"
            f"seed{manifest['expert_prediction_seed']}__"
            f"{manifest['fold']}_prediction_manifest.json"
        )
        path = manifest_root / name
        _write_json(path, manifest)
        manifests.append(str(path))
    registry_checksum = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    discovered_aliases = sorted({value.protocol_id for value in candidates})
    registry_aliases = sorted(
        str(value["protocol_id"]) for value in registry["protocols"]
    )
    registry_audit: dict[str, Any] = {
        "registry_path": str(registry_path),
        "registry_checksum": registry_checksum,
        "registry_schema_status": "PASS_FROZEN_V4_REGISTRY",
        "registry_aliases": registry_aliases,
        "discovered_aliases": discovered_aliases,
        "discovered_alias_status": (
            "PASS_ALL_DISCOVERED_ALIASES_FROZEN"
            if set(discovered_aliases).issubset(registry_aliases)
            else "FAIL_UNREGISTERED_ALIAS"
        ),
        "contract_binding_status": "BLOCKED_NO_LOADABLE_V4_MANIFESTS",
        "validated_bindings": [],
    }
    if manifests:
        loaded = load_prediction_artifacts(tuple(Path(value) for value in manifests))
        registry_audit["validated_bindings"] = validate_converted_bindings(
            loaded,
            registry,
        )
        registry_audit["contract_binding_status"] = (
            "PASS_CONVERTED_BINDINGS"
        )
    discovery_payload = {
        "schema_version": "coregraph_manifest_discovery_v4",
        "roots": [str(Path(value).expanduser().resolve()) for value in roots],
        "candidate_count": len(candidates),
        "validation_report_count": len(validation_reports),
        "candidates": [
            {
                "path": str(value.path),
                "dataset": value.dataset,
                "source_protocol": value.source_protocol,
                "protocol_id": value.protocol_id,
                "source_expert": value.source_expert,
                "expert_id": value.expert_id,
                "expert_prediction_seed": value.expert_prediction_seed,
                "fold": value.fold,
            }
            for value in candidates
        ],
    }
    conversion_counts = status_counts(records, "conversion_status")
    matrix_counts = status_counts(matrix, "status")
    verdict = _verdict(matrix)
    portable_discovery = _portable_payload(discovery_payload, path_aliases)
    portable_audits = _portable_payload(
        [value.to_dict() for value in audits],
        path_aliases,
    )
    portable_records = _portable_payload(list(records), path_aliases)
    portable_matrix = _portable_payload(list(matrix), path_aliases)
    portable_registry_audit = _portable_payload(
        registry_audit,
        path_aliases,
    )
    portable_roots = [
        str(_portable_payload(str(Path(value).expanduser().resolve()), path_aliases))
        for value in roots
    ]
    _write_json(output_root / "discovery.json", portable_discovery)
    _write_json(
        output_root / "candidate_audits.json",
        {
            "schema_version": "coregraph_candidate_audits_v4",
            "audits": portable_audits,
        },
    )
    _write_json(
        output_root / "conversion_records.json",
        {
            "schema_version": "coregraph_conversion_records_v4",
            "records": portable_records,
        },
    )
    _write_json(
        output_root / "no_training_audit_status.json",
        {
            "schema_version": "coregraph_no_training_audit_status_v4",
            "status": verdict,
            "training_performed": False,
            "metric_computation_performed": False,
            "target_oracle_measurement_performed": False,
            "converted_manifest_count": len(manifests),
            "converted_manifests": manifests,
            "conversion_status_counts": conversion_counts,
            "matrix_status_counts": matrix_counts,
            "split_audit_status": "PASS_AVAILABLE_ROWS_AUDITED_NO_METRICS",
            "label_known_audit_status": (
                "PASS_AVAILABLE_ROWS_AUDITED_NO_METRICS"
            ),
            "contract_registry_audit": portable_registry_audit,
            "cross_role_leakage_audit_status": (
                "BLOCKED_INCOMPLETE_OR_UNRESOLVED_MANIFESTS"
            ),
            "no_training_runner_status": (
                "BLOCKED_INCOMPLETE_OR_UNRESOLVED_MANIFESTS"
            ),
            "no_training_gate_completeness_status": (
                "BLOCKED_INCOMPLETE_OR_UNRESOLVED_MANIFESTS"
            ),
        },
    )
    _write_matrix(
        build_root / "MANIFEST_COMPLETENESS_MATRIX.csv",
        portable_matrix,
    )
    _write_json(build_root / "MANIFEST_LEAKAGE_AUDIT.json", leakage)
    (build_root / "MANIFEST_CONVERSION_REPORT.md").write_text(
        _report_markdown(
            roots=portable_roots,
            candidate_count=len(candidates),
            validation_report_count=len(validation_reports),
            validated_candidate_count=sum(
                value.validated_export for value in audits
            ),
            structurally_usable_candidate_count=sum(
                value.structurally_usable for value in audits
            ),
            conversion_counts=conversion_counts,
            matrix_counts=matrix_counts,
            registry_audit=registry_audit,
            verdict=verdict,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": verdict,
                "candidate_count": len(candidates),
                "converted_manifest_count": len(manifests),
                "matrix_status_counts": matrix_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
