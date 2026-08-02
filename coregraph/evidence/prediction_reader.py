"""Schema-aware streaming facade for prediction CSV archive members."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from coregraph.evidence.archive_store import ArchiveIntegrityError, ArchiveStore
from coregraph.evidence.member_index import MemberIndex, MemberRecord


REQUIRED_COLUMNS = frozenset({"score", "y_true", "split", "label_known"})


def _known(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", "", "nan", "unknown"}:
        return False
    raise ArchiveIntegrityError(f"invalid label_known value: {value!r}")


@dataclass(frozen=True)
class PredictionChunk:
    record: MemberRecord
    rows: tuple[Mapping[str, str], ...]
    first_source_row: int


class PredictionReader:
    def __init__(self, store: ArchiveStore, index: MemberIndex) -> None:
        self.store = store
        self.index = index

    def iter_chunks(
        self,
        *,
        dataset: str,
        protocol: str,
        expert: str,
        seed: int,
        splits: Sequence[str] = ("test",),
        require_label_known: bool = True,
        chunk_size: int = 50_000,
        id_column: str | None = None,
        require_monotonic_ids: bool = False,
    ) -> Iterator[PredictionChunk]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        record = self.index.locate(dataset, protocol, expert, seed)
        allowed_splits = set(splits)
        if not allowed_splits:
            raise ValueError("at least one split must be selected")
        with self.store.open_member(
            record.archive_name,
            record.member_name,
            expected_sha256=record.member_sha256,
        ) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            fieldnames = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ArchiveIntegrityError(
                    f"prediction schema missing columns {sorted(missing)} in {record.member_name}"
                )
            if id_column is not None and id_column not in fieldnames:
                raise ArchiveIntegrityError(f"prediction ID column is absent: {id_column}")
            rows: list[Mapping[str, str]] = []
            first_source_row = 0
            previous_id: str | None = None
            for source_row, row in enumerate(reader, start=2):
                if row["split"] not in allowed_splits:
                    continue
                if require_label_known and not _known(row["label_known"]):
                    continue
                if require_monotonic_ids and id_column is not None:
                    current = row[id_column]
                    if previous_id is not None and current < previous_id:
                        raise ArchiveIntegrityError(
                            f"non-deterministic row order in {record.member_name}: "
                            f"{current!r} follows {previous_id!r}"
                        )
                    previous_id = current
                if not rows:
                    first_source_row = source_row
                rows.append(dict(row))
                if len(rows) == chunk_size:
                    yield PredictionChunk(record, tuple(rows), first_source_row)
                    rows = []
            if rows:
                yield PredictionChunk(record, tuple(rows), first_source_row)
