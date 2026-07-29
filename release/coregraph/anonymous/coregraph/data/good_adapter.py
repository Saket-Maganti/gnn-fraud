"""Metadata-only GOOD adapter and tiny no-download fixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GOODDatasetSpec:
    name: str
    task: str
    domain: str
    shift: str
    acquisition: str
    checksum: str = "CHECK_AT_ACQUISITION"
    official_baseline_compatible: bool = True


SELECTED_GOOD_SETTINGS: tuple[GOODDatasetSpec, ...] = (
    GOODDatasetSpec(
        name="GOODCora",
        task="node_classification",
        domain="degree",
        shift="covariate",
        acquisition="Install pinned GOOD package and use its official dataset cache.",
    ),
    GOODDatasetSpec(
        name="GOODArxiv",
        task="node_classification",
        domain="time",
        shift="covariate",
        acquisition="Install pinned GOOD package and use its official dataset cache.",
    ),
    GOODDatasetSpec(
        name="GOODTwitch",
        task="node_classification",
        domain="language",
        shift="concept",
        acquisition="Install pinned GOOD package and use its official dataset cache.",
    ),
)


def validate_good_record(record: Mapping[str, object], spec: GOODDatasetSpec) -> None:
    required = {"x", "y", "edge_index", "domain_id", "split"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{spec.name} record missing keys: {missing}")
    edge_index = np.asarray(record["edge_index"])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("GOOD edge_index must have shape [2,E]")
    if len(np.asarray(record["x"])) != len(np.asarray(record["y"])):
        raise ValueError("GOOD features and labels must align")


def tiny_good_fixture() -> dict[str, np.ndarray]:
    return {
        "x": np.asarray([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]]),
        "y": np.asarray([0, 0, 1, 1]),
        "edge_index": np.asarray([[0, 1, 2], [1, 2, 3]]),
        "domain_id": np.asarray([0, 0, 1, 1]),
        "split": np.asarray(["train", "validation", "test", "test"]),
    }
