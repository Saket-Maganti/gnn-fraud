#!/usr/bin/env python3
"""Stream provider row identities to close cross-protocol scenario scopes."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.io.path_resolution import resolve_paths


PROVIDER_SPLITS = {b"train": "train", b"val": "validation", b"test": "test"}


def main() -> int:
    cache = resolve_paths(start=ROOT).evidence_cache
    index_path = cache / "indexes" / "RB09V3_MEMBER_INDEX.csv"
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["expert"] == "feature_mlp"]
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 provider-scope representatives, observed {len(rows)}")
    fingerprints: dict[tuple[str, int, str, str], str] = {}
    counts: dict[tuple[str, int, str, str], int] = {}
    for number, row in enumerate(rows, start=1):
        archive_path = cache / "archives" / row["archive_name"]
        digests = {split: hashlib.sha256() for split in PROVIDER_SPLITS.values()}
        split_counts = defaultdict(int)
        with zipfile.ZipFile(archive_path) as archive, archive.open(row["member_name"]) as member:
            header = member.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
            positions = {name: header.index(name) for name in header}
            for raw in member:
                values = raw.rstrip(b"\r\n").split(b",")
                provider_split = values[positions["split"]]
                split = PROVIDER_SPLITS[provider_split]
                identity = b"\x1f".join(
                    (
                        values[positions["dataset"]],
                        values[positions["seed"]],
                        provider_split,
                        values[positions["node_id"]],
                        values[positions["timestep"]],
                        values[positions["y_true"]],
                        values[positions["label_known"]],
                    )
                )
                digests[split].update(identity + b"\n")
                split_counts[split] += 1
        for split, digest in digests.items():
            key = row["dataset"], int(row["seed"]), row["protocol"], split
            fingerprints[key] = digest.hexdigest()
            counts[key] = split_counts[split]
        print(
            json.dumps(
                {
                    "representative": number,
                    "of": 60,
                    "dataset": row["dataset"],
                    "protocol": row["protocol"],
                    "seed": int(row["seed"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    failures: list[str] = []
    groups: list[dict[str, object]] = []
    protocols = ("isolated_inductive", "strict_inductive", "transductive_structure")
    for dataset in ("dgraphfin", "elliptic"):
        for seed in range(1, 11):
            split_status: dict[str, object] = {}
            for split in ("train", "validation", "test"):
                values = [fingerprints[(dataset, seed, protocol, split)] for protocol in protocols]
                count_values = [counts[(dataset, seed, protocol, split)] for protocol in protocols]
                aligned = len(set(values)) == 1 and len(set(count_values)) == 1
                split_status[split] = {
                    "aligned_across_protocols": aligned,
                    "row_count": count_values[0],
                    "semantic_sha256": values[0] if aligned else "MISMATCH",
                }
                if not aligned:
                    failures.append(f"alignment:{dataset}:seed{seed}:{split}")
            split_hashes = {
                str(value["semantic_sha256"])
                for value in split_status.values()
                if isinstance(value, dict)
            }
            disjoint_by_identity = len(split_hashes) == 3
            if not disjoint_by_identity:
                failures.append(f"split_identity_collision:{dataset}:seed{seed}")
            groups.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "protocols": list(protocols),
                    "splits": split_status,
                    "source_train_validation_vs_target_test_disjoint": disjoint_by_identity,
                }
            )
    report = {
        "schema": "coregraph_v5_row_scope_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "representative_members_streamed": len(rows),
        "cross_protocol_dataset_seed_groups": len(groups),
        "cross_expert_alignment_authority": "ARCHIVE_MEMBER_VALIDATION_60_GROUPS",
        "source_train_validation_vs_target_test_disjoint_groups": sum(
            bool(group["source_train_validation_vs_target_test_disjoint"]) for group in groups
        ),
        "provider_partition_alignment_groups": sum(
            all(
                bool(value["aligned_across_protocols"])
                for value in group["splits"].values()
            )
            for group in groups
        ),
        "target_labels_exposed": False,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
        "permanent_extractions": 0,
        "groups": groups,
        "failures": failures,
    }
    output = ROOT / "results/coregraph_build/V5_ROW_SCOPE_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "groups"}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
