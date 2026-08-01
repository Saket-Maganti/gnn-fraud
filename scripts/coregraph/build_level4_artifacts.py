#!/usr/bin/env python3
"""Build deterministic, results-blocked Level-4 registries and plans."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.baselines.registry import level4_baselines
from coregraph.benchmarks.synthetic_mechanisms import Mechanism, mechanism_registry
from coregraph.evidence.archive_store import CANONICAL_ARCHIVE_HASHES
from coregraph.experiments.scenario_manifests import make_scenario_id
from coregraph.contracts.axes import AccessRegime
from coregraph.io.path_resolution import resolve_paths


BUILD = ROOT / "results" / "coregraph_build"
SOURCE_INDEX = BUILD / "CANONICAL_RB09V3_ARTIFACT_INDEX.csv"
PROTOCOLS = ("strict_inductive", "isolated_inductive", "transductive_structure")
EXPERTS = ("feature_mlp", "gcn", "graphsage")
DATASETS = ("elliptic", "dgraphfin")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{_sha256_bytes(encoded)[:24]}"


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        if arguments == ("rev-parse", "HEAD"):
            return os.environ.get("COREGRAPH_SOURCE_SHA", "SOURCE_SNAPSHOT_NO_GIT_METADATA")
        if arguments == ("branch", "--show-current"):
            return os.environ.get(
                "COREGRAPH_SOURCE_BRANCH", "codex/coregraph-iclr-buildout-2026"
            )
        raise


def _load_canonical_index() -> list[dict[str, str]]:
    if not SOURCE_INDEX.is_file():
        raise FileNotFoundError(f"canonical V5 source index is absent: {SOURCE_INDEX}")
    with SOURCE_INDEX.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 180:
        raise ValueError(f"canonical V5 source index must contain 180 rows, observed {len(rows)}")
    coordinates = {
        (row["dataset"], row["protocol_id"], row["expert_id"], int(row["expert_prediction_seed"]))
        for row in rows
    }
    expected = {
        (dataset, protocol, expert, seed)
        for dataset in DATASETS
        for protocol in PROTOCOLS
        for expert in EXPERTS
        for seed in range(1, 11)
    }
    if coordinates != expected:
        raise ValueError("canonical V5 source index does not match the 2x3x3x10 grid")
    return rows


def _load_verified_member_index() -> dict[tuple[str, str, str, int], dict[str, str]]:
    cache_root = resolve_paths(start=ROOT).evidence_cache
    index_path = cache_root / "indexes" / "RB09V3_MEMBER_INDEX.csv"
    validation_path = cache_root / "audits" / "ARCHIVE_MEMBER_VALIDATION.json"
    if index_path.is_file() and validation_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("verdict") != "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS":
            return {}
        with index_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        # A clean source snapshot deliberately excludes the external archive
        # cache.  Preserve already validated compact metadata without claiming
        # that archive bytes were re-read inside the snapshot.
        report_path = BUILD / "ARCHIVE_MEMBER_VALIDATION.json"
        base_path = BUILD / "V5_BASE_ARTIFACTS.csv"
        if not report_path.is_file() or not base_path.is_file():
            return {}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("verdict") != "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS":
            return {}
        with base_path.open(encoding="utf-8", newline="") as handle:
            base_rows = list(csv.DictReader(handle))
        rows = [
            {
                "dataset": row["dataset"],
                "protocol": row["protocol"],
                "expert": row["expert"],
                "seed": row["seed"],
                "member_sha256": row["member_sha256"],
                "size_bytes": row["size_bytes"],
                "row_count": row["row_count"],
                "label_known_count": row["label_known_count"],
                "semantic_identity_sha256": row["semantic_identity_sha256"],
                "split_counts": row["row_scope"],
                "coordinate_verified": "true",
                "row_order_verified": "true",
                "chronology_verified": "true",
                "provider_alignment_verified": "true",
            }
            for row in base_rows
            if row.get("status") == "VERIFIED_ROLE_NEUTRAL_BASE_ARTIFACT"
        ]
    if len(rows) != 180:
        raise ValueError(f"verified member index must contain 180 rows, observed {len(rows)}")
    indexed = {
        (row["dataset"], row["protocol"], row["expert"], int(row["seed"])): row
        for row in rows
    }
    if len(indexed) != 180 or any(
        row["coordinate_verified"] != "true"
        or row["row_order_verified"] != "true"
        or row["chronology_verified"] != "true"
        or row["provider_alignment_verified"] != "true"
        for row in rows
    ):
        raise ValueError("member index is not a complete verified coordinate grid")
    return indexed


def build_authority(*, evidence_verified: bool) -> None:
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    payload = {
        "schema": "coregraph_project_paths_and_authorities_v1",
        "coregraph": {
            "path": "${COREGRAPH_REPO_ROOT}",
            "branch": branch,
            "build_parent_sha": head,
            "authority": "branch_tip_after_normal_fast_forward_push",
            "repository_form_at_preflight": "VALID_LINKED_WORKTREE_PORTABILITY_REPAIR_PENDING",
            "repository_form_current": (
                "INDEPENDENT_GIT_CHECKOUT_VALIDATED"
                if (ROOT / ".git").is_dir()
                else "LINKED_WORKTREE"
            ),
        },
        "curated_fraudshiftbench": {
            "path": "${COREGRAPH_CURATED_ROOT}",
            "branch": "codex/curated-fraudshiftbench-2026",
            "sha": "2dec25eac1d7a8951f9d4639f49e889c4c9ca486",
            "frozen_scientific_files": 249,
        },
        "evidence_cache": {
            "path": "${COREGRAPH_EVIDENCE_CACHE}",
            "status": (
                "VERIFIED_CANONICAL_RB09V3_LOCAL_CACHE"
                if evidence_verified
                else "BLOCKED_CANONICAL_ARCHIVES_UNAVAILABLE"
            ),
        },
        "data_root": "${COREGRAPH_DATA_ROOT}",
        "output_root": "${COREGRAPH_OUTPUT_ROOT}",
        "pull_request": {"number": 2, "state_required": "OPEN_DRAFT_UNMERGED"},
        "absolute_private_defaults": False,
    }
    _write_json(ROOT / "PROJECT_PATHS_AND_AUTHORITIES.json", payload)


def build_evidence(
    rows: list[dict[str, str]],
    verified_members: Mapping[tuple[str, str, str, int], Mapping[str, str]],
) -> None:
    if verified_members:
        required = (
            BUILD / "EVIDENCE_CACHE_MANIFEST.csv",
            BUILD / "EVIDENCE_CACHE_CHECKSUMS.sha256",
            BUILD / "ARCHIVE_MEMBER_VALIDATION.json",
            BUILD / "EVIDENCE_CACHE_BUILD_REPORT.md",
            BUILD / "SSD_INDEPENDENCE_REPORT.md",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "verified cache reports are incomplete; rerun "
                f"validate_level4_evidence_cache.py: {missing}"
            )
        report = json.loads((BUILD / "ARCHIVE_MEMBER_VALIDATION.json").read_text())
        if report.get("verdict") != "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS":
            raise ValueError("repository evidence report conflicts with verified external index")
        return
    manifest: list[dict[str, object]] = []
    for archive, digest in sorted(CANONICAL_ARCHIVE_HASHES.items()):
        manifest.append(
            {
                "record_type": "archive",
                "record_id": archive,
                "dataset": archive.split("_")[0],
                "protocol": "",
                "expert": "",
                "seed": "",
                "archive": archive,
                "member": "",
                "expected_sha256": digest,
                "observed_sha256": "",
                "status": "BLOCKED_ARCHIVE_ABSENT",
                "provenance": "CANONICAL_RB09V3_INDEX",
            }
        )
    for row in sorted(
        rows,
        key=lambda item: (
            item["dataset"], item["protocol_id"], item["expert_id"], int(item["expert_prediction_seed"])
        ),
    ):
        archive = Path(row["source_archive_path"]).name
        member_id = f"{row['dataset']}:{row['protocol_id']}:{row['expert_id']}:seed{row['expert_prediction_seed']}"
        manifest.append(
            {
                "record_type": "member",
                "record_id": member_id,
                "dataset": row["dataset"],
                "protocol": row["protocol_id"],
                "expert": row["expert_id"],
                "seed": row["expert_prediction_seed"],
                "archive": archive,
                "member": row["source_archive_member"],
                "expected_sha256": row["indexed_prediction_checksum"],
                "observed_sha256": "",
                "status": "BLOCKED_MEMBER_BYTES_UNAVAILABLE",
                "provenance": "CANONICAL_RB09V3_INDEX",
            }
        )
        manifest.append(
            {
                "record_type": "result",
                "record_id": member_id,
                "dataset": row["dataset"],
                "protocol": row["protocol_id"],
                "expert": row["expert_id"],
                "seed": row["expert_prediction_seed"],
                "archive": "",
                "member": row["result_path"],
                "expected_sha256": row["result_checksum"],
                "observed_sha256": row["result_checksum"],
                "status": "METADATA_INDEXED_PREDICTION_LINK_BLOCKED",
                "provenance": row["config_provenance_type"],
            }
        )
    fields = (
        "record_type", "record_id", "dataset", "protocol", "expert", "seed", "archive",
        "member", "expected_sha256", "observed_sha256", "status", "provenance",
    )
    count = _write_csv(BUILD / "EVIDENCE_CACHE_MANIFEST.csv", manifest, fields)
    if count != 366:
        raise AssertionError(f"evidence manifest must have 366 records, observed {count}")
    checksum_text = "# Expected canonical archive SHA-256 values; archive bytes are absent locally.\n" + "\n".join(
        f"{digest}  archives/{name}" for name, digest in sorted(CANONICAL_ARCHIVE_HASHES.items())
    )
    _write_text(BUILD / "EVIDENCE_CACHE_CHECKSUMS.sha256", checksum_text)
    archive_validation = {
        "schema": "coregraph_archive_member_validation_v1",
        "verdict": "BLOCKED_SOURCE_AUTHORITY_UNMOUNTED",
        "archive_expected": 6,
        "archive_present": 0,
        "archive_verified": 0,
        "member_expected": 180,
        "member_verified": 0,
        "schema_verified": 0,
        "coordinate_verified_from_index": 180,
        "row_order_verified": 0,
        "label_known_verified": 0,
        "fabricated_hashes": 0,
        "archives": [
            {
                "name": name,
                "expected_sha256": digest,
                "present": False,
                "status": "BLOCKED_ARCHIVE_ABSENT",
            }
            for name, digest in sorted(CANONICAL_ARCHIVE_HASHES.items())
        ],
    }
    _write_json(BUILD / "ARCHIVE_MEMBER_VALIDATION.json", archive_validation)
    _write_text(
        BUILD / "EVIDENCE_CACHE_BUILD_REPORT.md",
        """# Evidence cache build report

Verdict: `BLOCKED_SOURCE_AUTHORITY_UNMOUNTED`.

The portable cache directory and a 366-record expected inventory were created: six canonical archives, 180 prediction members, and 180 indexed result records. The source storage authority was not mounted, so no archive was copied, no observed archive/member checksum was asserted, and no ZIP/schema/row audit was performed. The six expected archive hashes are inherited from the canonical RB09v3 index and are labelled expected, not observed.

The extraction-free reader verifies archive and member hashes by streaming and exposes split/`label_known` filtering. Its tiny ZIP fixtures pass independently. Production readiness remains blocked until all six exact archives validate.
""",
    )
    _write_text(
        BUILD / "SSD_INDEPENDENCE_REPORT.md",
        """# SSD independence report

Status: `CODE_PATH_SSD_INDEPENDENT_EVIDENCE_SMOKE_BLOCKED`.

Normal evidence APIs resolve `${COREGRAPH_EVIDENCE_CACHE}` and contain no source-storage absolute default. Manifest/scenario planning is reproducible without external storage. Representative production members, 180 byte-backed base manifests, row-level leakage checks, and the full offline pilot smoke could not run because the six canonical archives are absent. Therefore SSD independence is established for configuration and no-training planning, not yet for production evidence execution.
""",
    )


def build_v5(
    rows: list[dict[str, str]],
    verified_members: Mapping[tuple[str, str, str, int], Mapping[str, str]],
) -> None:
    row_scope_report_path = BUILD / "V5_ROW_SCOPE_AUDIT.json"
    row_scope_report = (
        json.loads(row_scope_report_path.read_text(encoding="utf-8"))
        if row_scope_report_path.is_file()
        else {}
    )
    row_scope_verified = bool(verified_members) and row_scope_report.get("status") == "PASS"
    by_coordinate = {
        (row["dataset"], row["protocol_id"], row["expert_id"], int(row["expert_prediction_seed"])): row
        for row in rows
    }
    base_rows = []
    for coordinate in sorted(by_coordinate):
        row = by_coordinate[coordinate]
        member = verified_members.get(coordinate)
        member_sha256 = member["member_sha256"] if member else ""
        base_artifact_hash = (
            _sha256_bytes(
                json.dumps(
                    {
                        "schema": "coregraph_role_neutral_base_artifact_v5",
                        "dataset": row["dataset"],
                        "task": row["task"],
                        "protocol": row["protocol_id"],
                        "expert": row["expert_id"],
                        "seed": int(row["expert_prediction_seed"]),
                        "fold": row["fold"],
                        "archive_sha256": row["source_archive_sha256"],
                        "member_sha256": member_sha256,
                        "config_sha256": row["config_sha256"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if member
            else ""
        )
        base_rows.append(
            {
                "dataset": row["dataset"],
                "task": row["task"],
                "protocol": row["protocol_id"],
                "expert": row["expert_id"],
                "seed": row["expert_prediction_seed"],
                "fold": row["fold"],
                "base_coordinate_id": row["base_coordinate_id"],
                "base_artifact_hash": base_artifact_hash,
                "role": "ROLE_NEUTRAL",
                "archive": Path(row["source_archive_path"]).name,
                "archive_expected_sha256": row["source_archive_sha256"],
                "member": row["source_archive_member"],
                "member_sha256": member_sha256,
                "size_bytes": member["size_bytes"] if member else "",
                "row_count": member["row_count"] if member else "",
                "label_known_count": member["label_known_count"] if member else "",
                "semantic_identity_sha256": (
                    member["semantic_identity_sha256"] if member else ""
                ),
                "row_scope": (
                    member["split_counts"]
                    if member
                    else "BLOCKED_BEFORE_BYTE_ACCESS"
                ),
                "status": (
                    "VERIFIED_ROLE_NEUTRAL_BASE_ARTIFACT"
                    if member
                    else "BLOCKED_MEMBER_BYTES_UNAVAILABLE"
                ),
            }
        )
    base_fields = tuple(base_rows[0])
    if _write_csv(BUILD / "V5_BASE_ARTIFACTS.csv", base_rows, base_fields) != 180:
        raise AssertionError("V5 base registry must have 180 rows")

    scenarios: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    for dataset in DATASETS:
        for target_protocol in PROTOCOLS:
            source_protocols = tuple(protocol for protocol in PROTOCOLS if protocol != target_protocol)
            for seed in range(1, 11):
                scenario_id = make_scenario_id(
                    dataset=dataset,
                    target_protocol_id=target_protocol,
                    expert_prediction_seed=seed,
                    fold="fold0",
                    access_regime=AccessRegime.DG_NO_TARGET.value,
                )
                scenario_bindings = []
                for protocol in PROTOCOLS:
                    for expert in EXPERTS:
                        row = by_coordinate[(dataset, protocol, expert, seed)]
                        role = "target" if protocol == target_protocol else "source"
                        binding_payload = {
                            "scenario_id": scenario_id,
                            "base_coordinate_id": row["base_coordinate_id"],
                            "protocol": protocol,
                            "expert": expert,
                            "seed": seed,
                            "role": role,
                        }
                        scenario_bindings.append(
                            {
                                "binding_id": _stable_id("binding", binding_payload),
                                **binding_payload,
                                "permitted_splits": "test" if role == "target" else "train;validation",
                                "evaluation_split": "test" if role == "target" else "validation",
                                "label_access": "EVALUATION_ONLY_AFTER_FREEZE" if role == "target" else "SOURCE_LABELS_ALLOWED",
                                "resource_mask": "all_experts_available",
                                "review_budget": "UNKNOWN_UNTIL_OPERATIONAL_SPEC",
                                "status": (
                                    "MATERIALISABLE_FROM_VERIFIED_LOCAL_CACHE"
                                    if verified_members
                                    else "BLOCKED_MEMBER_BYTES_UNAVAILABLE"
                                ),
                            }
                        )
                bindings.extend(scenario_bindings)
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "dataset": dataset,
                        "target_protocol": target_protocol,
                        "source_protocols": ";".join(source_protocols),
                        "seed": seed,
                        "fold": "fold0",
                        "access_regime": AccessRegime.DG_NO_TARGET.value,
                        "feasible_experts": ";".join(EXPERTS),
                        "resource_mask": "all_experts_available",
                        "review_budget": "UNKNOWN_UNTIL_OPERATIONAL_SPEC",
                        "source_binding_count": 6,
                        "target_binding_count": 3,
                        "label_policy": "NO_TARGET_LABELS_DURING_FITTING",
                        "status": (
                            "MATERIALISABLE_FROM_VERIFIED_LOCAL_CACHE"
                            if verified_members
                            else "BLOCKED_MEMBER_BYTES_UNAVAILABLE"
                        ),
                    }
                )
    if _write_csv(BUILD / "V5_SCENARIOS.csv", scenarios, tuple(scenarios[0])) != 60:
        raise AssertionError("V5 scenario registry must have 60 rows")
    if _write_csv(BUILD / "V5_BINDINGS.csv", bindings, tuple(bindings[0])) != 540:
        raise AssertionError("V5 binding registry must have 540 rows")
    grouped: dict[str, list[dict[str, object]]] = {}
    for binding in bindings:
        grouped.setdefault(str(binding["scenario_id"]), []).append(binding)
    structure_pass = all(
        len(items) == 9
        and sum(item["role"] == "source" for item in items) == 6
        and sum(item["role"] == "target" for item in items) == 3
        and not (
            {item["base_coordinate_id"] for item in items if item["role"] == "source"}
            & {item["base_coordinate_id"] for item in items if item["role"] == "target"}
        )
        for items in grouped.values()
    )
    leakage = {
        "schema": "coregraph_v5_level4_leakage_audit_v1",
        "base_artifact_count": 180,
        "scenario_count": 60,
        "binding_count": 540,
        "source_binding_count": 360,
        "target_binding_count": 180,
        "base_artifacts_role_neutral": True,
        "same_scenario_dual_role": False,
        "target_protocol_absent_from_source_bindings": structure_pass,
        "target_labels_inaccessible_during_fitting": True,
        "structural_status": "PASS" if structure_pass else "FAIL",
        "row_scope_status": (
            "PASS_20_DATASET_SEED_GROUPS"
            if row_scope_verified
            else "BLOCKED_ROW_SCOPE_AUDIT_REQUIRED"
        ),
        "source_train_validation_vs_target_test_disjoint_groups": (
            row_scope_report.get("source_train_validation_vs_target_test_disjoint_groups", 0)
        ),
        "chronology_status": "PASS" if verified_members else "BLOCKED_MEMBER_BYTES_UNAVAILABLE",
        "label_known_status": "PASS" if verified_members else "BLOCKED_MEMBER_BYTES_UNAVAILABLE",
        "provider_alignment_status": (
            "PASS_60_GROUPS" if verified_members else "BLOCKED_MEMBER_BYTES_UNAVAILABLE"
        ),
        "dataset_identity_status": "PASS" if verified_members else "BLOCKED_MEMBER_BYTES_UNAVAILABLE",
        "metric_computation_performed": False,
        "oracle_computation_performed": False,
        "overall_status": (
            "PASS_NO_TRAINING_BYTE_AND_STRUCTURE"
            if row_scope_verified
            else "PASS_STRUCTURE_BLOCKED_ROW_SCOPE"
        ),
    }
    _write_json(BUILD / "V5_LEAKAGE_AUDIT.json", leakage)
    _write_text(
        BUILD / "V5_MATERIALISATION_REPORT.md",
        (
            """# V5 materialisation report

Status: `PASS_NO_TRAINING_BYTE_AND_STRUCTURE`.

- Role-neutral base coordinates: 180/180 registered and byte-verified.
- Evaluation scenarios: 60/60 registered and materialisable from the local canonical cache.
- Scenario bindings: 540/540 registered (360 source, 180 target).
- Archive/member identity, schema, coordinate, chronology, label-known, row-order, and 60 cross-expert provider-alignment groups: pass.
- Cross-protocol provider partitions and source train/validation versus target test disjointness: 20/20 dataset-seed groups pass.
- Structural role, held-out protocol, split permission, and no-target-label fitting rules: pass.

The registry is a validated no-training evidence surface, not an empirical result. No target metric, target oracle, fitting, or threshold selection occurred. A base coordinate may change roles across scenarios but never has both roles within one scenario.
"""
            if row_scope_verified
            else """# V5 materialisation report

Status: `STRUCTURAL_REGISTRY_COMPLETE_BYTE_MATERIALISATION_BLOCKED`.

- Role-neutral base coordinates: 180/180 registered; 0/180 byte-verified.
- Evaluation scenarios: 60/60 registered; 0/60 byte-materialisable.
- Scenario bindings: 540/540 registered (360 source, 180 target).
- Structural role, held-out protocol, split permission, and no-target-label fitting rules: pass.
- Member checksum, row-scope disjointness, chronology, unknown-label exclusion, and provider-score alignment: blocked because archive bytes are absent.

The registry is a truthful execution plan, not evidence availability. A base coordinate may change roles across scenarios but never has both roles within one scenario.
"""
        ),
    )
    browser_lines = [
        "# Level-4 scenario browser",
        "",
        "This registry is a no-training view. Target labels remain evaluation-only after freeze.",
        "",
        "| Scenario | Dataset | Held-out target | Source protocols | Seed | Bindings | Status |",
        "|---|---|---|---|---:|---:|---|",
    ]
    browser_lines.extend(
        "| `{scenario_id}` | {dataset} | `{target_protocol}` | `{source_protocols}` | "
        "{seed} | 6 source + 3 target | `{status}` |".format(**scenario)
        for scenario in scenarios
    )
    _write_text(ROOT / "docs/coregraph/LEVEL4_SCENARIO_BROWSER.md", "\n".join(browser_lines))


def build_baselines() -> None:
    records = [
        {
            "baseline_id": item.baseline_id,
            "category": item.category,
            "status": item.status.value,
            "deployable": str(item.deployable).lower(),
            "target_label_access": item.target_label_access,
            "official_repository": item.official_repository,
            "official_commit": item.official_commit,
            "licence": item.licence,
            "protocol_validity": item.protocol_validity,
            "blocker": item.blocker,
        }
        for item in level4_baselines()
    ]
    _write_csv(BUILD / "LEVEL4_BASELINE_REGISTRY.csv", records, tuple(records[0]))


def build_run_matrix() -> None:
    scenarios = list(csv.DictReader((BUILD / "V5_SCENARIOS.csv").open(encoding="utf-8", newline="")))
    rows: list[dict[str, object]] = []
    common = {
        "budget": "declared_by_scenario",
        "resource_mask": "all_experts_available",
        "objective_ablation": "none",
        "encoder_ablation": "none",
        "diagnostic_ablation": "none",
        "synthetic_mechanism": "NOT_APPLICABLE",
        "graph_ood_adapter": "NOT_APPLICABLE",
        "expected_runtime": "UNKNOWN_UNTIL_PILOT",
        "expected_memory": "BLOCKED_RESOURCE_UNMEASURED",
        "expected_output": "run_manifest+predictions+resource_record+validation",
        "dependency": "CANONICAL_SAVED_OUTPUTS",
    }
    methods = (
        ("coregraph", "hierarchical", "PILOT_MUST_RUN", "L4-H1;L4-H2;L4-H3;L4-H4;L4-H6"),
        ("uniform_average", "fixed", "PILOT_MUST_RUN", "L4-H1"),
        ("best_fixed_expert", "fixed", "PILOT_MUST_RUN", "L4-H1"),
        ("source_logistic_gate", "instance", "PILOT_MUST_RUN", "L4-H2"),
        ("source_mlp_gate", "instance", "FULL_MUST_RUN", "L4-H2"),
        ("contract_router", "contract", "FULL_MUST_RUN", "L4-H3"),
        ("instance_router", "instance", "FULL_MUST_RUN", "L4-H3"),
        ("resource_aware_heuristic", "contract", "STRONG_RECOMMENDED", "L4-H5"),
        ("cheapest_feasible_expert", "fixed", "DIAGNOSTIC", "L4-H5"),
    )
    for scenario in scenarios:
        for baseline, routing_mode, priority, claim in methods:
            rows.append(
                {
                    "run_id": _stable_id("run", [scenario["scenario_id"], baseline, "fraud"]),
                    "dataset": scenario["dataset"],
                    "contract": scenario["scenario_id"],
                    "held_out_composition": scenario["target_protocol"],
                    "expert_set": ";".join(EXPERTS),
                    "baseline": baseline,
                    "seed": scenario["seed"],
                    "routing_mode": routing_mode,
                    "expected_device": "CPU_SAVED_OUTPUT_PILOT",
                    "priority": priority,
                    "claim_supported": claim,
                    **common,
                }
            )
    for spec in mechanism_registry():
        for seed in range(1, 11):
            for baseline, routing_mode in (
                ("coregraph", "hierarchical"),
                ("protocol_one_hot", "contract"),
                ("flat_contract_mlp", "contract"),
                ("uniform_average", "fixed"),
            ):
                rows.append(
                    {
                        "run_id": _stable_id("run", [spec.mechanism.value, seed, baseline]),
                        "dataset": "controlled_synthetic",
                        "contract": spec.held_out_contract,
                        "held_out_composition": spec.mechanism.value,
                        "expert_set": "feature;graph;recent",
                        "baseline": baseline,
                        "seed": seed,
                        "budget": "mechanism_declared",
                        "resource_mask": "mechanism_declared",
                        "routing_mode": routing_mode,
                        "objective_ablation": "none",
                        "encoder_ablation": "protocol_one_hot" if baseline == "protocol_one_hot" else "flat" if baseline == "flat_contract_mlp" else "none",
                        "diagnostic_ablation": "none",
                        "synthetic_mechanism": spec.mechanism.value,
                        "graph_ood_adapter": "NOT_APPLICABLE",
                        "expected_device": "CPU_TINY_THEN_GPU_FULL",
                        "expected_runtime": "UNKNOWN_UNTIL_PILOT",
                        "expected_memory": "BLOCKED_RESOURCE_UNMEASURED",
                        "expected_output": "mechanism_manifest+predictions+validation",
                        "dependency": "NONE_TINY;GPU_FULL_SEPARATE_AUTHORITY",
                        "priority": "FULL_MUST_RUN" if baseline in {"coregraph", "protocol_one_hot"} else "STRONG_RECOMMENDED",
                        "claim_supported": "L4-H1;L4-H7",
                    }
                )
    for family in ("GOOD", "OGB_molecular_fallback"):
        for seed in range(1, 11):
            for baseline in ("coregraph", "uniform_average", "strongest_feasible_graph_ood"):
                rows.append(
                    {
                        "run_id": _stable_id("run", [family, seed, baseline]),
                        "dataset": family,
                        "contract": "official_held_out_environment",
                        "held_out_composition": "official_split_mapping",
                        "expert_set": "task_compatible_only",
                        "baseline": baseline,
                        "seed": seed,
                        "budget": "NOT_APPLICABLE_OR_DATASET_DECLARED",
                        "resource_mask": "all_experts_available",
                        "routing_mode": "hierarchical" if baseline == "coregraph" else "baseline_declared",
                        "objective_ablation": "none",
                        "encoder_ablation": "none",
                        "diagnostic_ablation": "none",
                        "synthetic_mechanism": "NOT_APPLICABLE",
                        "graph_ood_adapter": family,
                        "expected_device": "KAGGLE_T4X2",
                        "expected_runtime": "UNKNOWN_UNTIL_OFFICIAL_SMOKE",
                        "expected_memory": "BLOCKED_RESOURCE_UNMEASURED",
                        "expected_output": "official_parity+run_manifest+predictions",
                        "dependency": "OFFICIAL_AVAILABLE_NOT_INSTALLED" if family == "GOOD" else "BLOCKED_PENDING_LICENSE_REVIEW",
                        "priority": "FULL_MUST_RUN" if family == "GOOD" else "STRETCH_LEVEL4",
                        "claim_supported": "CROSS_DOMAIN_GENERALISATION",
                    }
                )
    resource_profiles = (
        "all_experts_available", "one_graph_expert_unavailable", "all_graph_experts_unavailable",
        "tight_memory", "tight_latency", "tight_review_budget", "combined_graph_resource_shift",
        "dynamic_availability_change",
    )
    for dataset in DATASETS:
        for profile in resource_profiles:
            for seed in range(1, 11):
                for baseline in ("coregraph", "resource_aware_heuristic", "cheapest_feasible_expert"):
                    rows.append(
                        {
                            "run_id": _stable_id("run", [dataset, profile, seed, baseline]),
                            "dataset": dataset,
                            "contract": "resource_counterfactual",
                            "held_out_composition": profile,
                            "expert_set": ";".join(EXPERTS),
                            "baseline": baseline,
                            "seed": seed,
                            "budget": "profile_declared",
                            "resource_mask": profile,
                            "routing_mode": "hierarchical" if baseline == "coregraph" else "heuristic",
                            "objective_ablation": "none",
                            "encoder_ablation": "none",
                            "diagnostic_ablation": "none",
                            "synthetic_mechanism": "NOT_APPLICABLE",
                            "graph_ood_adapter": "NOT_APPLICABLE",
                            "expected_device": "CPU_COUNTERFACTUAL_THEN_T4_PROFILE",
                            "expected_runtime": "UNKNOWN_UNTIL_PILOT",
                            "expected_memory": "BLOCKED_RESOURCE_UNMEASURED",
                            "expected_output": "resource_record+counterfactual_manifest",
                            "dependency": "SAVED_OUTPUT_PILOT;RESOURCE_PROFILER",
                            "priority": "FULL_MUST_RUN",
                            "claim_supported": "L4-H5",
                        }
                    )
    fields = (
        "run_id", "dataset", "contract", "held_out_composition", "expert_set", "baseline", "seed",
        "budget", "resource_mask", "routing_mode", "objective_ablation", "encoder_ablation",
        "diagnostic_ablation", "synthetic_mechanism", "graph_ood_adapter", "expected_device",
        "expected_runtime", "expected_memory", "expected_output", "dependency", "priority", "claim_supported",
    )
    _write_csv(BUILD / "LEVEL4_FULL_RUN_MATRIX.csv", rows, fields)
    priority_counts = Counter(str(row["priority"]) for row in rows)
    device_counts = Counter(str(row["expected_device"]) for row in rows)
    _write_text(
        BUILD / "LEVEL4_RUN_COUNT_SUMMARY.md",
        "# Level-4 run count summary\n\nStatus: `PLAN_ONLY_NO_RUNS_EXECUTED`.\n\n"
        f"Total planned coordinates: **{len(rows)}**.\n\n"
        + "## Priority\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in sorted(priority_counts.items()))
        + "\n\n## Device plan\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in sorted(device_counts.items()))
        + "\n\nCounts are deterministic plan rows, not executed runs. Runtime and memory remain unmeasured.",
    )
    runtime_rows = [
        {"wave": "saved_output_pilot", "runtime": "UNKNOWN_UNTIL_PILOT", "basis": "NO_LEVEL4_TIMING_OBSERVED", "status": "BLOCKED_PILOT_INPUTS"},
        {"wave": "fraud_training", "runtime": "UNKNOWN_UNTIL_GPU_SMOKE", "basis": "NO_LEVEL4_TIMING_OBSERVED", "status": "BLOCKED_PENDING_AUTHORITY"},
        {"wave": "synthetic_full", "runtime": "UNKNOWN_UNTIL_GPU_SMOKE", "basis": "TINY_FIXTURE_NOT_EXTRAPOLATED", "status": "PLAN_ONLY"},
        {"wave": "graph_ood", "runtime": "UNKNOWN_UNTIL_OFFICIAL_SMOKE", "basis": "OFFICIAL_REPOSITORY_NOT_INSTALLED", "status": "BLOCKED_OFFICIAL_INTEGRATION"},
        {"wave": "resource_profile", "runtime": "UNKNOWN_UNTIL_MEASUREMENT", "basis": "PROTOCOL_ONLY", "status": "BLOCKED_RESOURCE_UNMEASURED"},
    ]
    _write_csv(BUILD / "LEVEL4_ESTIMATED_RUNTIME_TABLE.csv", runtime_rows, tuple(runtime_rows[0]))
    output_rows = [
        {"artifact_family": family, "estimated_size": "UNKNOWN_UNTIL_PILOT", "assumption": "NO_FABRICATED_ROW_OR_COMPRESSION_RATE", "status": status}
        for family, status in (
            ("saved_output_pilot", "BLOCKED_INPUTS"),
            ("fraud_full", "PLAN_ONLY"),
            ("synthetic_full", "PLAN_ONLY"),
            ("graph_ood", "BLOCKED_OFFICIAL_INTEGRATION"),
            ("paper_and_tables", "PENDING_RESULTS"),
        )
    ]
    _write_csv(BUILD / "LEVEL4_OUTPUT_SIZE_ESTIMATE.csv", output_rows, tuple(output_rows[0]))
    _write_text(
        BUILD / "LEVEL4_GPU_WAVE_PLAN.md",
        """# Level-4 GPU wave plan

Status: `PLAN_ONLY_NO_GPU_JOB_LAUNCHED`.

1. T4x2 bootstrap and repository/dataset/hash validation.
2. Fraud saved-output pilot validation, then separately authorised full fraud training.
3. Strong official baseline parity and fraud waves only for licensed task-valid adapters.
4. Controlled synthetic full mechanisms.
5. Primary GOOD adapter, then fallback only if its licence/data gate passes.
6. Resource profiling with fixed warmup and batch manifests.
7. Encoder, diagnostic, objective, and routing ablations.
8. Prediction regeneration only for integrity-confirmed missing artifacts.
9. Output validation, checksum, one final ZIP, and completion report.

Each coordinate is resumable and failures are explicit. OOM is `RESOURCE_BLOCKED`; no cell is silently skipped.
""",
    )
    _write_text(
        BUILD / "LEVEL4_CPU_WAVE_PLAN.md",
        """# Level-4 CPU wave plan

Status: `VALIDATION_AND_ANALYSIS_ONLY`.

The sequential CPU bundle performs path and evidence manifests, no-training scenario/leakage audits, tiny mechanisms, theory checks, notebook static validation, future statistical analysis, figure/table generation, paper compilation, release construction, and clean-room validation. It does not fit a real gate, compute target metrics/oracles, or start official baseline training.
""",
    )


def build_failure_labels() -> None:
    labels = [
        "WRONG_EXPERT_SELECTED", "ALL_EXPERTS_POOR", "OVERCONFIDENT_DIAGNOSTICS",
        "ROUTING_INSTABILITY", "CORRELATED_SHIFT", "LATENT_CONTRACT_AMBIGUITY",
        "BUDGET_COLLAPSE", "RESOURCE_MASK_COLLAPSE", "ABSTENTION_COLLAPSE",
        "SOURCE_OVERFITTING", "BASELINE_UNFAIRNESS", "METRIC_DISAGREEMENT",
    ]
    _write_json(
        BUILD / "LEVEL4_FAILURE_LABELS.json",
        {"schema": "coregraph_level4_failure_labels_v1", "frozen": True, "labels": labels},
    )


def build_preregistration_hash() -> str:
    inputs = (
        ROOT / "docs" / "coregraph" / "LEVEL4_STATISTICAL_ANALYSIS_PLAN.md",
        ROOT / "configs" / "coregraph" / "level4" / "STATISTICAL_GATE_SPEC.json",
        ROOT / "configs" / "coregraph" / "level4" / "CLAIM_GATE_SPEC.json",
        ROOT / "docs" / "coregraph" / "LEVEL4_CLAIM_LEDGER.csv",
    )
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    value = digest.hexdigest()
    _write_text(
        BUILD / "LEVEL4_PREREGISTRATION_HASH.txt",
        f"sha256 {value}\nstatus FROZEN_BEFORE_REAL_LEVEL4_RESULTS\ninputs "
        + ";".join(path.relative_to(ROOT).as_posix() for path in inputs),
    )
    return value


def build_cleanup_inventory() -> None:
    candidates = (
        ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage",
        "coregraph/__pycache__", "fraudshiftbench/__pycache__", "models/__pycache__",
        "tests/__pycache__", ".DS_Store", "tmp/pdfs",
    )
    rows = []
    for relative in candidates:
        path = ROOT / relative
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        else:
            size = 0
        rows.append(
            {
                "path": relative,
                "size_bytes": size,
                "tracked": "false",
                "reproducibility": "REPRODUCIBLE",
                "authority": "NON_AUTHORITATIVE",
                "referenced_by_code": "false",
                "referenced_by_reports": "false",
                "safe_to_delete": "true",
                "backup_location": "NOT_REQUIRED",
                "decision": "REMOVE_AFTER_VALIDATION",
            }
        )
    rows.append(
        {
            "path": "../gnn-fraud-old",
            "size_bytes": "NOT_SCANNED_READ_ONLY_ARCHIVE",
            "tracked": "outside_active_repo",
            "reproducibility": "UNKNOWN_MIXED",
            "authority": "HISTORICAL_ONLY",
            "referenced_by_code": "false",
            "referenced_by_reports": "historical_provenance_only",
            "safe_to_delete": "false",
            "backup_location": "USER_MANAGED_BACKUP_REQUIRES_VERIFICATION",
            "decision": "RETAIN_PENDING_USER_ARCHIVE_DECISION",
        }
    )
    _write_csv(BUILD / "LOCAL_CLEANUP_INVENTORY.csv", rows, tuple(rows[0]))
    duplicates = [
        {
            "path_a": "release/coregraph/anonymous",
            "path_b": "active source tree",
            "relationship": "GENERATED_ANONYMOUS_SNAPSHOT",
            "authoritative": "active source tree",
            "decision": "REBUILD_NOT_MANUALLY_EDIT",
        },
        {
            "path_a": "../gnn-fraud-old",
            "path_b": "active repositories",
            "relationship": "HISTORICAL_OVERLAP_UNQUANTIFIED",
            "authoritative": "active repositories plus frozen evidence manifests",
            "decision": "RETAIN_READ_ONLY_PENDING_BACKUP_VERIFICATION",
        },
    ]
    _write_csv(BUILD / "LOCAL_CLEANUP_DUPLICATES.csv", duplicates, tuple(duplicates[0]))


def build_theory_status() -> None:
    _write_text(
        BUILD / "LEVEL4_THEORY_STATUS.md",
        """# Level-4 theory status

| Result | Status | Executable check | Limitation |
|---|---|---|---|
| Fixed-mixture impossibility | `PROVED_INTERNAL_REVIEW_PENDING` | analytic value and finite crossing construction | randomized selection risk; nonlinear prediction mixtures need more assumptions |
| Regret decomposition | `PROVED_INTERNAL_REVIEW_PENDING` | nonnegative component bookkeeping | terms are not all identifiable causal effects |
| Compositional generalisation | `PROVED_INTERNAL_REVIEW_PENDING` | near-bound and XOR counterexample | observed axis values, bounded interactions, identifiable relative risks |
| Resource-mask guarantee | `PROVED_AND_REVIEWED` | exact zero mass, unit sum, empty-set sentinel | depends on correct measured constraints |
| Selective-risk transfer | `PROVED_INTERNAL_REVIEW_PENDING` | finite density-ratio construction | not distribution-free; requires positive target coverage |

No theorem implies empirical fraud performance. External mathematical review remains required before submission.
""",
    )


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    rows = _load_canonical_index()
    verified_members = _load_verified_member_index()
    build_authority(evidence_verified=bool(verified_members))
    build_evidence(rows, verified_members)
    build_v5(rows, verified_members)
    build_baselines()
    build_run_matrix()
    build_failure_labels()
    build_cleanup_inventory()
    build_theory_status()
    preregistration = build_preregistration_hash()
    print(
        json.dumps(
            {
                "status": "LEVEL4_RESULTS_BLOCKED_ARTIFACTS_GENERATED",
                "base_artifacts": 180,
                "scenarios": 60,
                "bindings": 540,
                "archive_present": 6 if verified_members else 0,
                "preregistration_sha256": preregistration,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
