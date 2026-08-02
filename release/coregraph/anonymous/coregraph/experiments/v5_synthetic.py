"""Deterministic tiny 180-member ZIP fixture for the complete V5 path."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from coregraph.contracts.serialization import stable_sha256
from coregraph.experiments.scenario_manifests import make_scenario_id
from coregraph.experiments.v5_pilot_types import EXPERT_ORDER
from coregraph.experiments.v5_scenario_loader import V5PilotConfig


DATASETS = ("elliptic", "dgraphfin")
PROTOCOLS = ("strict_inductive", "isolated_inductive", "transductive_structure")
ARCHIVES = {
    ("elliptic", "strict_inductive"): "elliptic_10seed_strict_inductive.zip",
    ("elliptic", "isolated_inductive"): "elliptic_10seed_inductive_isolated.zip",
    ("elliptic", "transductive_structure"): "elliptic_10seed_transductive.zip",
    ("dgraphfin", "strict_inductive"): "dgraphfin_10seed_strict_inductive.zip",
    ("dgraphfin", "isolated_inductive"): "dgraphfin_10seed_inductive_isolated.zip",
    ("dgraphfin", "transductive_structure"): "dgraphfin_10seed_transductive.zip",
}
PROVIDER_PROTOCOL = {
    "strict_inductive": "strict_inductive",
    "isolated_inductive": "inductive_isolated",
    "transductive_structure": "transductive",
}
PROVIDER_EXPERT = {"feature_mlp": "mlp", "gcn": "gcn", "graphsage": "graphsage"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _payload(dataset: str, protocol: str, expert: str, seed: int) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "dataset",
        "protocol",
        "model",
        "seed",
        "split",
        "node_id",
        "timestep",
        "y_true",
        "score",
        "label_known",
        "artifact_source",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    winner = PROTOCOLS.index(protocol)
    expert_index = EXPERT_ORDER.index(expert)
    row_index = 0
    for split, count, timestep in (("train", 8, 1), ("validation", 8, 2), ("test", 8, 3)):
        for local in range(count):
            label = int((local + seed) % 3 == 0)
            quality = 0.90 if expert_index == winner else 0.62 - 0.04 * expert_index
            score = quality if label else 1.0 - quality
            score = min(0.999, max(0.001, score + (local % 2) * 0.005))
            writer.writerow(
                {
                    "dataset": dataset,
                    "protocol": PROVIDER_PROTOCOL[protocol],
                    "model": PROVIDER_EXPERT[expert],
                    "seed": seed,
                    "split": "val" if split == "validation" else split,
                    "node_id": row_index,
                    "timestep": timestep,
                    "y_true": label,
                    "score": f"{score:.6f}",
                    "label_known": "True",
                    "artifact_source": "synthetic_fixture",
                }
            )
            row_index += 1
    writer.writerow(
        {
            "dataset": dataset,
            "protocol": PROVIDER_PROTOCOL[protocol],
            "model": PROVIDER_EXPERT[expert],
            "seed": seed,
            "split": "test",
            "node_id": row_index,
            "timestep": 3,
            "y_true": 0,
            "score": "0.5",
            "label_known": "False",
            "artifact_source": "synthetic_fixture",
        }
    )
    return output.getvalue().encode("utf-8")


def build_synthetic_fixture(root: Path, config: V5PilotConfig) -> tuple[Path, Path]:
    fixture = root.resolve()
    evidence = fixture / "evidence"
    archives_root = evidence / "archives"
    indexes_root = evidence / "indexes"
    registries = fixture / "registries"
    archives_root.mkdir(parents=True, exist_ok=True)
    indexes_root.mkdir(parents=True, exist_ok=True)
    registries.mkdir(parents=True, exist_ok=True)
    member_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    archive_hashes: dict[str, str] = {}
    base_by_key: dict[tuple[str, str, str, int], str] = {}
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            archive_name = ARCHIVES[(dataset, protocol)]
            archive_path = archives_root / archive_name
            pending: list[tuple[str, bytes, str, int, str, int]] = []
            for seed in range(1, 11):
                for expert in EXPERT_ORDER:
                    payload = _payload(dataset, protocol, expert, seed)
                    member_name = (
                        f"predictions/{dataset}__{PROVIDER_PROTOCOL[protocol]}__"
                        f"{PROVIDER_EXPERT[expert]}__seed{seed}.csv"
                    )
                    member_hash = hashlib.sha256(payload).hexdigest()
                    coordinate = stable_sha256(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "expert": expert,
                            "seed": seed,
                            "fold": "fold0",
                        }
                    )
                    base_by_key[(dataset, protocol, expert, seed)] = coordinate
                    pending.append((member_name, payload, expert, seed, coordinate, len(payload)))
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for member_name, payload, *_ in pending:
                    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, payload)
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            archive_hashes[archive_name] = archive_hash
            for member_name, payload, expert, seed, coordinate, size_bytes in pending:
                member_hash = hashlib.sha256(payload).hexdigest()
                base_hash = stable_sha256(
                    {
                        "schema": "coregraph_role_neutral_base_artifact_v5",
                        "dataset": dataset,
                        "protocol": protocol,
                        "expert": expert,
                        "seed": seed,
                        "archive_sha256": archive_hash,
                        "member_sha256": member_hash,
                    }
                )
                base_rows.append(
                    {
                        "dataset": dataset,
                        "task": "node_classification",
                        "protocol": protocol,
                        "expert": expert,
                        "seed": seed,
                        "fold": "fold0",
                        "base_coordinate_id": coordinate,
                        "base_artifact_hash": base_hash,
                        "role": "ROLE_NEUTRAL",
                        "archive": archive_name,
                        "archive_expected_sha256": archive_hash,
                        "member": member_name,
                        "member_sha256": member_hash,
                        "size_bytes": size_bytes,
                        "row_count": 25,
                        "label_known_count": 24,
                        "semantic_identity_sha256": stable_sha256(
                            {"coordinate": coordinate, "rows": 25}
                        ),
                        "row_scope": json.dumps({"train": 8, "validation": 8, "test": 9}),
                        "status": "VERIFIED_ROLE_NEUTRAL_BASE_ARTIFACT",
                    }
                )
                member_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "expert": expert,
                        "seed": seed,
                        "archive_name": archive_name,
                        "member_name": member_name,
                        "member_sha256": member_hash,
                        "size_bytes": size_bytes,
                        "row_count": 25,
                        "label_known_count": 24,
                        "label_unknown_count": 1,
                        "split_counts": json.dumps({"train": 8, "validation": 8, "test": 9}),
                        "label_known_by_split": json.dumps({"train": 8, "validation": 8, "test": 8}),
                        "timestamp_ranges": json.dumps(
                            {"train": [1, 1], "validation": [2, 2], "test": [3, 3]}
                        ),
                        "semantic_identity_sha256": stable_sha256(
                            {"coordinate": coordinate, "rows": 25}
                        ),
                        "duplicate_identifier_count": 0,
                        "schema_version": "RB09V3_PREDICTION_CSV_V1",
                        "coordinate_verified": "true",
                        "row_order_verified": "true",
                        "chronology_verified": "true",
                        "provider_alignment_verified": "true",
                    }
                )
    scenario_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for target in PROTOCOLS:
            sources = tuple(protocol for protocol in PROTOCOLS if protocol != target)
            for seed in range(1, 11):
                scenario_id = make_scenario_id(
                    dataset=dataset,
                    target_protocol_id=target,
                    expert_prediction_seed=seed,
                    fold="fold0",
                    access_regime="DG_NO_TARGET",
                )
                scenario_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "dataset": dataset,
                        "target_protocol": target,
                        "source_protocols": ";".join(sources),
                        "seed": seed,
                        "fold": "fold0",
                        "access_regime": "DG_NO_TARGET",
                        "feasible_experts": ";".join(EXPERT_ORDER),
                        "resource_mask": "all_experts_available",
                        "review_budget": "0.01",
                        "source_binding_count": 6,
                        "target_binding_count": 3,
                        "label_policy": "NO_TARGET_LABELS_DURING_FITTING",
                        "status": "SYNTHETIC_MATERIALISABLE",
                    }
                )
                for protocol in PROTOCOLS:
                    role = "target" if protocol == target else "source"
                    for expert in EXPERT_ORDER:
                        coordinate = base_by_key[(dataset, protocol, expert, seed)]
                        binding_rows.append(
                            {
                                "binding_id": "binding-" + stable_sha256(
                                    {
                                        "scenario": scenario_id,
                                        "coordinate": coordinate,
                                        "role": role,
                                    }
                                )[:24],
                                "scenario_id": scenario_id,
                                "base_coordinate_id": coordinate,
                                "protocol": protocol,
                                "expert": expert,
                                "seed": seed,
                                "role": role,
                                "permitted_splits": "test" if role == "target" else "train;validation",
                                "evaluation_split": "test" if role == "target" else "validation",
                                "label_access": "EVALUATION_ONLY_AFTER_FREEZE" if role == "target" else "SOURCE_LABELS_ALLOWED",
                                "resource_mask": "all_experts_available",
                                "review_budget": "0.01",
                                "status": "SYNTHETIC_MATERIALISABLE",
                            }
                        )
    _write_csv(registries / "V5_BASE_ARTIFACTS.csv", base_rows)
    _write_csv(registries / "V5_SCENARIOS.csv", scenario_rows)
    _write_csv(registries / "V5_BINDINGS.csv", binding_rows)
    _write_csv(indexes_root / "RB09V3_MEMBER_INDEX.csv", member_rows)
    config_payload = copy.deepcopy(dict(config.payload))
    config_payload["preregistration_path"] = str(
        config.resolve("preregistration_path")
    )
    config_payload["base_artifact_registry"] = str(
        registries / "V5_BASE_ARTIFACTS.csv"
    )
    config_payload["scenario_registry"] = str(registries / "V5_SCENARIOS.csv")
    config_payload["binding_registry"] = str(registries / "V5_BINDINGS.csv")
    config_payload["member_index"] = "indexes/RB09V3_MEMBER_INDEX.csv"
    config_payload["archive_hashes"] = archive_hashes
    config_payload["optimization"]["coregraph_steps"] = 3
    config_payload["streaming"]["source_rows_per_split_per_environment"] = 8
    synthetic_config = fixture / "saved_output_v5.synthetic.yaml"
    synthetic_config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    return synthetic_config, evidence
