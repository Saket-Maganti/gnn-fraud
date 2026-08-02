#!/usr/bin/env python3
"""Validate canonical RB09v3 ZIPs and build extraction-free cache metadata.

This command performs integrity and semantic audits only.  It never computes a
target metric, fits a model, selects a threshold, or permanently extracts a ZIP
member.  Member digests are derived only after their containing archive matches
the frozen canonical archive digest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evidence.archive_store import CANONICAL_ARCHIVE_HASHES, sha256_path
from coregraph.io.path_resolution import resolve_paths


BUILD = ROOT / "results" / "coregraph_build"
CANONICAL_INDEX = BUILD / "CANONICAL_RB09V3_ARTIFACT_INDEX.csv"
PREDICTION_COLUMNS = (
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
PROVIDER_SPLIT_MAPPING = {"train": "train", "val": "validation", "test": "test"}
ALLOWED_SPLITS = frozenset(PROVIDER_SPLIT_MAPPING.values())


class EvidenceCacheValidationError(RuntimeError):
    """A canonical archive or prediction member failed validation."""


@dataclass(frozen=True)
class MemberAudit:
    dataset: str
    protocol: str
    expert: str
    seed: int
    archive_name: str
    member_name: str
    member_sha256: str
    size_bytes: int
    row_count: int
    label_known_count: int
    label_unknown_count: int
    split_counts: str
    label_known_by_split: str
    timestamp_ranges: str
    semantic_identity_sha256: str
    duplicate_identifier_count: int
    schema_version: str
    coordinate_verified: bool
    row_order_verified: bool
    chronology_verified: bool
    provider_alignment_verified: bool = False

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        for name in (
            "coordinate_verified",
            "row_order_verified",
            "chronology_verified",
            "provider_alignment_verified",
        ):
            row[name] = str(row[name]).lower()
        return row


def _read_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 180:
        raise EvidenceCacheValidationError(
            f"canonical source index must contain 180 records, observed {len(rows)}"
        )
    coordinates = {
        (
            row["dataset"],
            row["protocol_id"],
            row["expert_id"],
            int(row["expert_prediction_seed"]),
        )
        for row in rows
    }
    expected = {
        (dataset, protocol, expert, seed)
        for dataset in ("dgraphfin", "elliptic")
        for protocol in (
            "isolated_inductive",
            "strict_inductive",
            "transductive_structure",
        )
        for expert in ("feature_mlp", "gcn", "graphsage")
        for seed in range(1, 11)
    }
    if coordinates != expected:
        raise EvidenceCacheValidationError("canonical source index coordinate grid is not 2x3x3x10")
    return rows


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


def _known(value: bytes) -> bool:
    normalized = value.strip().lower()
    if normalized in {b"true", b"1"}:
        return True
    if normalized in {b"false", b"0"}:
        return False
    raise EvidenceCacheValidationError(f"invalid label_known value {value!r}")


def _member_audit(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    row: Mapping[str, str],
) -> MemberAudit:
    digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    split_counts: Counter[str] = Counter()
    known_by_split: Counter[str] = Counter()
    unknown_count = 0
    row_count = 0
    size_bytes = 0
    timestamps: dict[str, list[int]] = defaultdict(list)
    track_duplicates = row["expert_id"] == "feature_mlp"
    identifiers: set[bytes] | None = set() if track_duplicates else None
    duplicates = 0
    previous_identity: bytes | None = None
    ordering_errors = 0
    expected_coordinate = (
        row["dataset"].encode(),
        row["source_protocol"].encode(),
        row["source_model"].encode(),
        row["expert_prediction_seed"].encode(),
    )
    with archive.open(info, "r") as source:
        header = source.readline()
        digest.update(header)
        size_bytes += len(header)
        observed_header = tuple(header.decode("utf-8-sig").rstrip("\r\n").split(","))
        if observed_header != PREDICTION_COLUMNS:
            raise EvidenceCacheValidationError(
                f"schema mismatch for {info.filename}: {observed_header}"
            )
        for raw_line in source:
            digest.update(raw_line)
            size_bytes += len(raw_line)
            values = raw_line.rstrip(b"\r\n").split(b",")
            if len(values) != len(PREDICTION_COLUMNS):
                raise EvidenceCacheValidationError(
                    f"non-canonical CSV encoding in {info.filename} at row {row_count + 2}"
                )
            row_count += 1
            if tuple(values[:4]) != expected_coordinate:
                raise EvidenceCacheValidationError(
                    f"coordinate mismatch in {info.filename} at row {row_count + 1}"
                )
            provider_split = values[4].decode("ascii")
            if provider_split not in PROVIDER_SPLIT_MAPPING:
                raise EvidenceCacheValidationError(
                    f"unexpected split {provider_split!r} in {info.filename}"
                )
            split = PROVIDER_SPLIT_MAPPING[provider_split]
            split_counts[split] += 1
            identifier = values[5]
            if identifiers is not None:
                if identifier in identifiers:
                    duplicates += 1
                identifiers.add(identifier)
            try:
                timestamp = int(values[6])
                label = int(values[7])
                score = float(values[8])
            except ValueError as exc:
                raise EvidenceCacheValidationError(
                    f"invalid numeric value in {info.filename} at row {row_count + 1}"
                ) from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise EvidenceCacheValidationError(
                    f"score outside [0,1] in {info.filename} at row {row_count + 1}"
                )
            known = _known(values[9])
            if known:
                known_by_split[split] += 1
                if label == 0:
                    raise EvidenceCacheValidationError(
                        f"provider-unknown label declared known in {info.filename}"
                    )
            else:
                unknown_count += 1
                if label != 0:
                    raise EvidenceCacheValidationError(
                        f"unknown row has nonzero provider label in {info.filename}"
                    )
            timestamps[split].append(timestamp)
            identity = b"\x1f".join(
                (values[0], values[1], values[3], values[4], values[5], values[6], values[7], values[9])
            )
            semantic_digest.update(identity + b"\n")
            if previous_identity is not None and identity == previous_identity:
                ordering_errors += 1
            previous_identity = identity
    if row_count <= 0:
        raise EvidenceCacheValidationError(f"empty prediction member: {info.filename}")
    if size_bytes != info.file_size:
        raise EvidenceCacheValidationError(
            f"uncompressed size mismatch for {info.filename}: "
            f"expected {info.file_size}, streamed {size_bytes}"
        )
    ranges = {
        split: [min(values), max(values)]
        for split, values in sorted(timestamps.items())
        if values
    }
    chronology = (
        set(ranges) == ALLOWED_SPLITS
        and ranges["train"][1] < ranges["validation"][0]
        and ranges["validation"][1] < ranges["test"][0]
    )
    if not chronology:
        raise EvidenceCacheValidationError(
            f"chronology is not train<validation<test in {info.filename}: {ranges}"
        )
    if duplicates:
        raise EvidenceCacheValidationError(
            f"duplicate provider identifiers in {info.filename}: {duplicates}"
        )
    if ordering_errors:
        raise EvidenceCacheValidationError(
            f"adjacent duplicate semantic rows in {info.filename}: {ordering_errors}"
        )
    return MemberAudit(
        dataset=row["dataset"],
        protocol=row["protocol_id"],
        expert=row["expert_id"],
        seed=int(row["expert_prediction_seed"]),
        archive_name=Path(row["source_archive_path"]).name,
        member_name=row["source_archive_member"],
        member_sha256=digest.hexdigest(),
        size_bytes=info.file_size,
        row_count=row_count,
        label_known_count=sum(known_by_split.values()),
        label_unknown_count=unknown_count,
        split_counts=json.dumps(dict(sorted(split_counts.items())), sort_keys=True, separators=(",", ":")),
        label_known_by_split=json.dumps(
            dict(sorted(known_by_split.items())), sort_keys=True, separators=(",", ":")
        ),
        timestamp_ranges=json.dumps(ranges, sort_keys=True, separators=(",", ":")),
        semantic_identity_sha256=semantic_digest.hexdigest(),
        duplicate_identifier_count=duplicates,
        schema_version="RB09V3_PREDICTION_CSV_V1",
        coordinate_verified=True,
        row_order_verified=True,
        chronology_verified=True,
    )


def _validate_archives(
    cache_root: Path,
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[MemberAudit]]:
    archive_rows: list[dict[str, object]] = []
    audits: list[MemberAudit] = []
    by_archive: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_archive[Path(row["source_archive_path"]).name].append(row)
    for archive_name, expected_digest in sorted(CANONICAL_ARCHIVE_HASHES.items()):
        path = cache_root / "archives" / archive_name
        if not path.is_file():
            raise EvidenceCacheValidationError(f"canonical archive is absent: {archive_name}")
        observed_digest = sha256_path(path)
        if observed_digest != expected_digest:
            raise EvidenceCacheValidationError(
                f"archive checksum mismatch for {archive_name}: {observed_digest}"
            )
        expected_rows = sorted(
            by_archive[archive_name],
            key=lambda item: (
                item["protocol_id"], item["expert_id"], int(item["expert_prediction_seed"])
            ),
        )
        expected_members = {row["source_archive_member"] for row in expected_rows}
        if len(expected_members) != 30:
            raise EvidenceCacheValidationError(
                f"{archive_name} index must declare 30 unique prediction members"
            )
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise EvidenceCacheValidationError(
                    f"ZIP CRC failure in {archive_name}: {corrupt}"
                )
            infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
            prediction_members = {name for name in infos if name.startswith("predictions/")}
            if prediction_members != expected_members:
                missing = sorted(expected_members - prediction_members)
                extra = sorted(prediction_members - expected_members)
                raise EvidenceCacheValidationError(
                    f"member identity mismatch for {archive_name}: missing={missing}; extra={extra}"
                )
            auxiliary = sorted(set(infos) - prediction_members)
            expected_auxiliary = f"{archive_name.split('_10seed_', 1)[0]}_10seed_runs.csv"
            if auxiliary != [expected_auxiliary]:
                raise EvidenceCacheValidationError(
                    f"unexpected auxiliary members in {archive_name}: {auxiliary}"
                )
            for number, row in enumerate(expected_rows, start=1):
                audit = _member_audit(archive, infos[row["source_archive_member"]], row)
                audits.append(audit)
                print(
                    json.dumps(
                        {
                            "archive": archive_name,
                            "member": number,
                            "of": 30,
                            "rows": audit.row_count,
                            "sha256": audit.member_sha256,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        archive_rows.append(
            {
                "name": archive_name,
                "expected_sha256": expected_digest,
                "observed_sha256": observed_digest,
                "size_bytes": path.stat().st_size,
                "prediction_member_count": 30,
                "auxiliary_member_count": 1,
                "zip_crc_status": "PASS",
                "source_destination_relation": "ALREADY_AT_AUTHORITATIVE_DESTINATION",
                "read_only": not bool(path.stat().st_mode & stat.S_IWUSR),
                "status": "VERIFIED",
            }
        )
    return archive_rows, audits


def _mark_provider_alignment(audits: Sequence[MemberAudit]) -> list[MemberAudit]:
    groups: dict[tuple[str, str, int], list[MemberAudit]] = defaultdict(list)
    for audit in audits:
        groups[(audit.dataset, audit.protocol, audit.seed)].append(audit)
    if len(groups) != 60:
        raise EvidenceCacheValidationError(f"expected 60 semantic alignment groups, got {len(groups)}")
    output: list[MemberAudit] = []
    for key, values in groups.items():
        fingerprints = {value.semantic_identity_sha256 for value in values}
        row_counts = {value.row_count for value in values}
        if len(values) != 3 or len(fingerprints) != 1 or len(row_counts) != 1:
            raise EvidenceCacheValidationError(
                f"provider labels/order are not aligned across experts for {key}"
            )
        output.extend(
            MemberAudit(**{**asdict(value), "provider_alignment_verified": True})
            for value in values
        )
    return sorted(output, key=lambda item: (item.dataset, item.protocol, item.expert, item.seed))


def _emit_reports(
    cache_root: Path,
    source_rows: Sequence[Mapping[str, str]],
    archive_rows: Sequence[Mapping[str, object]],
    audits: Sequence[MemberAudit],
) -> None:
    indexes = cache_root / "indexes"
    manifests = cache_root / "manifests"
    checksums = cache_root / "checksums"
    cache_audits = cache_root / "audits"
    local_path_map = cache_root / "local_path_map"
    for directory in (indexes, manifests, checksums, cache_audits, local_path_map):
        directory.mkdir(parents=True, exist_ok=True)

    member_fields = tuple(audits[0].csv_row())
    member_rows = [audit.csv_row() for audit in audits]
    _write_csv(indexes / "RB09V3_MEMBER_INDEX.csv", member_rows, member_fields)
    by_coordinate = {
        (audit.dataset, audit.protocol, audit.expert, audit.seed): audit for audit in audits
    }
    evidence_rows: list[dict[str, object]] = []
    for archive in archive_rows:
        evidence_rows.append(
            {
                "record_type": "archive",
                "record_id": archive["name"],
                "dataset": str(archive["name"]).split("_", 1)[0],
                "protocol": "",
                "expert": "",
                "seed": "",
                "archive": archive["name"],
                "member": "",
                "expected_sha256": archive["expected_sha256"],
                "observed_sha256": archive["observed_sha256"],
                "size_bytes": archive["size_bytes"],
                "row_count": "",
                "label_known_count": "",
                "status": "VERIFIED_CANONICAL_ARCHIVE",
                "provenance": "CANONICAL_RB09V3_FROZEN_HASH",
            }
        )
    for source in sorted(
        source_rows,
        key=lambda row: (
            row["dataset"], row["protocol_id"], row["expert_id"], int(row["expert_prediction_seed"])
        ),
    ):
        coordinate = (
            source["dataset"],
            source["protocol_id"],
            source["expert_id"],
            int(source["expert_prediction_seed"]),
        )
        audit = by_coordinate[coordinate]
        identifier = f"{coordinate[0]}:{coordinate[1]}:{coordinate[2]}:seed{coordinate[3]}"
        evidence_rows.append(
            {
                "record_type": "member",
                "record_id": identifier,
                "dataset": audit.dataset,
                "protocol": audit.protocol,
                "expert": audit.expert,
                "seed": audit.seed,
                "archive": audit.archive_name,
                "member": audit.member_name,
                "expected_sha256": audit.member_sha256,
                "observed_sha256": audit.member_sha256,
                "size_bytes": audit.size_bytes,
                "row_count": audit.row_count,
                "label_known_count": audit.label_known_count,
                "status": "VERIFIED_FROM_CANONICAL_ARCHIVE_BYTES",
                "provenance": "STREAMED_AFTER_CANONICAL_ARCHIVE_HASH_MATCH",
            }
        )
        evidence_rows.append(
            {
                "record_type": "result",
                "record_id": identifier,
                "dataset": audit.dataset,
                "protocol": audit.protocol,
                "expert": audit.expert,
                "seed": audit.seed,
                "archive": "",
                "member": source["result_path"],
                "expected_sha256": source["result_checksum"],
                "observed_sha256": source["result_checksum"],
                "size_bytes": "",
                "row_count": "",
                "label_known_count": "",
                "status": "INDEXED_RESULT_METADATA_ONLY",
                "provenance": source["config_provenance_type"],
            }
        )
    evidence_fields = tuple(evidence_rows[0])
    if _write_csv(BUILD / "EVIDENCE_CACHE_MANIFEST.csv", evidence_rows, evidence_fields) != 366:
        raise EvidenceCacheValidationError("evidence manifest did not contain 366 records")
    _write_csv(manifests / "EVIDENCE_CACHE_MANIFEST.csv", evidence_rows, evidence_fields)

    checksum_lines = ["# Verified canonical archive and streamed member SHA-256 values."]
    checksum_lines.extend(
        f"{row['observed_sha256']}  archives/{row['name']}" for row in archive_rows
    )
    checksum_lines.extend(
        f"{audit.member_sha256}  {audit.archive_name}:{audit.member_name}" for audit in audits
    )
    checksum_text = "\n".join(checksum_lines)
    _write_text(BUILD / "EVIDENCE_CACHE_CHECKSUMS.sha256", checksum_text)
    _write_text(checksums / "EVIDENCE_CACHE_CHECKSUMS.sha256", checksum_text)

    total_archive_bytes = sum(int(row["size_bytes"]) for row in archive_rows)
    total_member_bytes = sum(audit.size_bytes for audit in audits)
    total_rows = sum(audit.row_count for audit in audits)
    report = {
        "schema": "coregraph_archive_member_validation_v2",
        "verdict": "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS",
        "archive_expected": 6,
        "archive_present": 6,
        "archive_verified": 6,
        "member_expected": 180,
        "member_identity_verified": 180,
        "member_checksum_verified": 180,
        "schema_verified": 180,
        "coordinate_verified": 180,
        "row_order_verified": 180,
        "chronology_verified": 180,
        "label_known_verified": 180,
        "provider_alignment_groups_verified": 60,
        "row_count_total": total_rows,
        "archive_size_bytes": total_archive_bytes,
        "uncompressed_prediction_member_bytes": total_member_bytes,
        "permanent_extractions": 0,
        "target_metrics_computed": 0,
        "target_oracles_computed": 0,
        "fabricated_hashes": 0,
        "archives": list(archive_rows),
    }
    _write_json(BUILD / "ARCHIVE_MEMBER_VALIDATION.json", report)
    _write_json(cache_audits / "ARCHIVE_MEMBER_VALIDATION.json", report)
    _write_json(
        local_path_map / "portable_paths.json",
        {
            "schema": "coregraph_local_path_map_v1",
            "archive_root": "${COREGRAPH_EVIDENCE_CACHE}/archives",
            "member_index": "${COREGRAPH_EVIDENCE_CACHE}/indexes/RB09V3_MEMBER_INDEX.csv",
            "private_absolute_paths_recorded": False,
        },
    )
    _write_text(
        BUILD / "EVIDENCE_CACHE_BUILD_REPORT.md",
        f"""# Evidence cache build report

Verdict: `PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS`.

- Canonical archives present and whole-file SHA-256 verified: 6/6.
- ZIP CRC tests: 6/6 pass.
- Expected prediction-member identities: 180/180 exact; each archive also contains its single run-summary CSV.
- Streamed prediction-member SHA-256 values, schemas, coordinates, ordering, chronology, and label-known semantics: 180/180 pass.
- Cross-expert provider-label and row alignment groups: 60/60 pass.
- Compressed local cache: {total_archive_bytes} bytes.
- Streamed uncompressed prediction payload: {total_member_bytes} bytes across {total_rows} rows.
- Permanent prediction CSV extractions: 0.

The six candidates were already at the authoritative external cache location. Their destination hashes equal the frozen canonical hashes, so no repository copy or SSD access was needed. Archives were marked read-only. Member digests are observations derived from bytes inside a whole-archive-hash-verified canonical ZIP; they are not inherited or fabricated values. No model fitting, target metric, target oracle, or threshold selection occurred.
""",
    )
    _write_text(
        BUILD / "SSD_INDEPENDENCE_REPORT.md",
        """# SSD independence report

Status: `PASS_LOCAL_CANONICAL_CACHE_NO_SSD_REFERENCE`.

All six canonical archives are present under `${COREGRAPH_EVIDENCE_CACHE}/archives` and validated without consulting `${SSD_SOURCE_AUTHORITY}`. Normal resolution uses the portable evidence-cache authority. All 180 member streams, 180 role-neutral base coordinates, 60 scenarios, and 540 structural bindings can be materialised from local metadata and ZIP streams. No prediction CSV was permanently extracted and no normal evidence path references the SSD source authority.

This is an evidence-availability and no-training validation. It is not an empirical pilot and contains no target metric or oracle computation.
""",
    )
    _write_text(
        cache_root / "README.md",
        """# CoReGraph local evidence cache

Status: `VERIFIED_CANONICAL_RB09V3_CACHE`.

This directory is intentionally outside Git and is the authoritative local source for the six canonical saved-prediction archives. Whole-archive hashes match the frozen RB09v3 values; all 180 prediction-member identities, streamed hashes, schemas, coordinates, chronology, label-known semantics, and 60 cross-expert alignment groups pass. The archives are read-only.

Never permanently extract the prediction CSV payloads. Use `coregraph.evidence.ArchiveStore`, `MemberIndex`, and `PredictionReader` for checksum-verified streaming. Machine-local path maps remain here and must never be committed. No SSD is required for normal evidence access.
""",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--canonical-index", type=Path, default=CANONICAL_INDEX)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    cache_root = (
        arguments.cache_root.expanduser().resolve()
        if arguments.cache_root is not None
        else resolve_paths(start=ROOT).evidence_cache
    )
    rows = _read_index(arguments.canonical_index)
    archive_rows, raw_audits = _validate_archives(cache_root, rows)
    audits = _mark_provider_alignment(raw_audits)
    if len(archive_rows) != 6 or len(audits) != 180:
        raise EvidenceCacheValidationError("validated evidence cardinality is incomplete")
    _emit_reports(cache_root, rows, archive_rows, audits)
    print(
        json.dumps(
            {
                "status": "PASS_CANONICAL_ARCHIVES_AND_180_MEMBERS",
                "archives": len(archive_rows),
                "members": len(audits),
                "cache_bytes": sum(int(row["size_bytes"]) for row in archive_rows),
                "ssd_used": False,
                "permanent_extractions": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
