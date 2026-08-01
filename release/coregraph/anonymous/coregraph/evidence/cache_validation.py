"""Offline evidence-cache inventory and fail-closed validation summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from coregraph.evidence.archive_store import (
    CANONICAL_ARCHIVE_HASHES,
    ArchiveIntegrityError,
    ArchiveStore,
)


@dataclass(frozen=True)
class ArchiveValidation:
    archive_name: str
    expected_sha256: str
    present: bool
    verified: bool
    member_count: int | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_cache(
    cache_root: Path,
    *,
    expected: Mapping[str, str] | None = None,
) -> tuple[ArchiveValidation, ...]:
    records = dict(expected or CANONICAL_ARCHIVE_HASHES)
    store = ArchiveStore(cache_root, records)
    results: list[ArchiveValidation] = []
    for name, digest in sorted(records.items()):
        path = store.archive_path(name)
        if not path.is_file():
            results.append(
                ArchiveValidation(name, digest, False, False, None, "BLOCKED_ARCHIVE_ABSENT")
            )
            continue
        try:
            members = store.list_members(name)
        except ArchiveIntegrityError as exc:
            results.append(
                ArchiveValidation(name, digest, True, False, None, f"INVALID:{exc}")
            )
        else:
            results.append(
                ArchiveValidation(name, digest, True, True, len(members), "VERIFIED")
            )
    return tuple(results)
