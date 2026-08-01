"""Typed coordinate index for immutable archive members."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MemberRecord:
    dataset: str
    protocol: str
    expert: str
    seed: int
    archive_name: str
    member_name: str
    member_sha256: str
    size_bytes: int | None = None
    row_count: int | None = None
    label_known_count: int | None = None

    @property
    def coordinate(self) -> tuple[str, str, str, int]:
        return self.dataset, self.protocol, self.expert, self.seed


class MemberIndex:
    def __init__(self, records: Iterable[MemberRecord]) -> None:
        self.records = tuple(records)
        by_coordinate: dict[tuple[str, str, str, int], MemberRecord] = {}
        by_member: dict[tuple[str, str], MemberRecord] = {}
        for record in self.records:
            if record.coordinate in by_coordinate:
                raise ValueError(f"duplicate member coordinate: {record.coordinate}")
            member_key = (record.archive_name, record.member_name)
            if member_key in by_member:
                raise ValueError(f"duplicate archive member: {member_key}")
            by_coordinate[record.coordinate] = record
            by_member[member_key] = record
        self._by_coordinate = by_coordinate

    def locate(self, dataset: str, protocol: str, expert: str, seed: int) -> MemberRecord:
        coordinate = dataset, protocol, expert, seed
        try:
            return self._by_coordinate[coordinate]
        except KeyError as exc:
            raise KeyError(f"archive member coordinate is absent: {coordinate}") from exc

    def validate_canonical_grid(self) -> None:
        datasets = {"dgraphfin", "elliptic"}
        protocols = {"isolated_inductive", "strict_inductive", "transductive_structure"}
        experts = {"feature_mlp", "gcn", "graphsage"}
        expected = {
            (dataset, protocol, expert, seed)
            for dataset in datasets
            for protocol in protocols
            for expert in experts
            for seed in range(1, 11)
        }
        observed = set(self._by_coordinate)
        if observed != expected:
            missing = sorted(expected - observed)[:5]
            extra = sorted(observed - expected)[:5]
            raise ValueError(
                f"canonical member grid mismatch: {len(observed)}/180; "
                f"missing={missing}; extra={extra}"
            )

    @classmethod
    def from_csv(cls, path: Path) -> "MemberIndex":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "dataset",
            "protocol",
            "expert",
            "seed",
            "archive_name",
            "member_name",
            "member_sha256",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"member index schema is incomplete: {path}")

        def optional_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            return int(value)

        return cls(
            MemberRecord(
                dataset=row["dataset"],
                protocol=row["protocol"],
                expert=row["expert"],
                seed=int(row["seed"]),
                archive_name=row["archive_name"],
                member_name=row["member_name"],
                member_sha256=row["member_sha256"],
                size_bytes=optional_int(row.get("size_bytes")),
                row_count=optional_int(row.get("row_count")),
                label_known_count=optional_int(row.get("label_known_count")),
            )
            for row in rows
        )
