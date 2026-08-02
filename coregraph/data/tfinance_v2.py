"""T-Finance V2 timestamp and provenance guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class TimestampQuality(str, Enum):
    PROVIDER_GENUINE = "provider_genuine"
    VERIFIED_DERIVED = "verified_derived"
    MISSING = "missing"
    EDGE_ORDER_PROXY = "edge_order_proxy"


@dataclass(frozen=True)
class TFinanceSourceAudit:
    source_url: str
    licence_status: str
    timestamp_quality: TimestampQuality
    checksum: str
    temporal_claim_allowed: bool


def require_temporal_timestamps(
    edge_timestamps: Optional[np.ndarray],
    *,
    quality: TimestampQuality,
    temporal_claim: bool,
    edge_count: int,
) -> np.ndarray:
    """Reject edge-order pseudo-time for every temporal claim."""

    if edge_timestamps is None:
        if temporal_claim:
            raise ValueError(
                "T-Finance temporal_claim=True requires genuine provider timestamps; "
                "edge-order fallback is prohibited"
            )
        return np.arange(edge_count, dtype=int)
    times = np.asarray(edge_timestamps)
    if len(times) != edge_count:
        raise ValueError("T-Finance timestamps must align to edges")
    if temporal_claim and quality not in {
        TimestampQuality.PROVIDER_GENUINE,
        TimestampQuality.VERIFIED_DERIVED,
    }:
        raise ValueError(
            f"timestamp quality {quality.value} is inadmissible for temporal claims"
        )
    return times


def validate_tfinance_archive(
    path: str | Path,
    *,
    temporal_claim: bool,
    audit: TFinanceSourceAudit,
) -> dict[str, int | str | bool]:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"T-Finance archive not found: {archive_path}. Acquire it manually "
            "after the source/licence audit; no download or synthetic fallback runs here."
        )
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {"x", "y", "edge_index"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"T-Finance archive missing keys: {missing}")
        edges = np.asarray(archive["edge_index"])
        if edges.shape[0] != 2:
            edges = edges.T
        times = archive["edge_timestamp"] if "edge_timestamp" in archive.files else None
        require_temporal_timestamps(
            times,
            quality=audit.timestamp_quality,
            temporal_claim=temporal_claim,
            edge_count=edges.shape[1],
        )
        return {
            "nodes": int(len(archive["x"])),
            "edges": int(edges.shape[1]),
            "timestamp_quality": audit.timestamp_quality.value,
            "temporal_claim_allowed": bool(
                temporal_claim and audit.temporal_claim_allowed
            ),
        }
