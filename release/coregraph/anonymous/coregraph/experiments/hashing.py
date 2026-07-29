"""Stable run and artifact hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from coregraph.contracts.serialization import canonical_json
from coregraph.experiments.config import RunConfig


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: RunConfig) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def stable_run_id(
    config: RunConfig,
    *,
    code_commit: str | None = None,
    dataset_manifest_hash: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "canonical_config": config.to_dict(),
        "code_commit": code_commit or config.code_commit,
        "dataset_manifest": dataset_manifest_hash or config.dataset_manifest,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
