"""Pinned named adapters for official external implementations."""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from coregraph.experts.base import OfficialStatus
from coregraph.experts.official_adapters.process import OfficialProcessAdapter


@dataclass(frozen=True)
class ParityResult:
    adapter: str
    status: str
    rows: int
    output_sha256: str
    errors: tuple[str, ...] = ()


def load_official_adapters(registry_path: str | Path) -> dict[str, OfficialProcessAdapter]:
    payload = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    adapters: dict[str, OfficialProcessAdapter] = {}
    for name, record in payload["baselines"].items():
        adapters[name] = OfficialProcessAdapter(
            name=name,
            repository=str(record["repository"]),
            commit=str(record["commit"]),
            licence=str(record["licence"]),
            checkout_env=str(record["checkout_env"]),
            entrypoint=str(record["entrypoint"]),
            status=OfficialStatus(record["status"]),
        )
    return adapters


def checkout_from_environment(adapter: OfficialProcessAdapter) -> Path | None:
    value = os.environ.get(adapter.checkout_env)
    return Path(value) if value else None


def validate_prediction_parity(
    adapter: OfficialProcessAdapter,
    output_path: str | Path,
    *,
    expected_ids: Sequence[str],
    id_column: str,
) -> ParityResult:
    """Validate schema/alignment; numeric parity tolerances belong to method fixtures."""

    path = Path(output_path)
    if not path.exists():
        return ParityResult(adapter.name, "PENDING_INTEGRATION", 0, "", ("output_missing",))
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[Mapping[str, str]] = list(csv.DictReader(handle))
    errors: list[str] = []
    required = {id_column, "score", "expert_id", "official_status"}
    if any(required - set(row) for row in rows):
        errors.append("schema_missing")
    ids = [row.get(id_column, "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_ids")
    if set(ids) != set(expected_ids):
        errors.append("identifier_mismatch")
    for row in rows:
        try:
            score = float(row["score"])
        except (KeyError, ValueError):
            errors.append("invalid_score")
            break
        if not 0 <= score <= 1:
            errors.append("score_out_of_range")
            break
    return ParityResult(
        adapter=adapter.name,
        status="PARITY_SCHEMA_PASS" if not errors else "PARITY_FAILED",
        rows=len(rows),
        output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        errors=tuple(sorted(set(errors))),
    )
