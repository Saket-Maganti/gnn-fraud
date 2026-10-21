"""Deterministic scalable batching primitives used by graph expert adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SamplingPlan:
    batch_size: int
    fanouts: tuple[int, ...] = (15, 10)
    gradient_accumulation: int = 1
    mixed_precision: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("batch size and accumulation must be positive")
        if not self.fanouts or any(fanout < 1 for fanout in self.fanouts):
            raise ValueError("fanouts must be positive")


def deterministic_batches(
    identifiers: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Iterator[np.ndarray]:
    identifiers = np.asarray(identifiers)
    order = np.arange(len(identifiers))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield identifiers[order[start : start + batch_size]]


def temporal_event_batches(
    event_ids: np.ndarray,
    timestamps: np.ndarray,
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield stable chronological event batches."""

    event_ids = np.asarray(event_ids)
    timestamps = np.asarray(timestamps)
    if len(event_ids) != len(timestamps):
        raise ValueError("event ids and timestamps must align")
    order = np.lexsort((event_ids.astype(str), timestamps))
    for start in range(0, len(order), batch_size):
        yield event_ids[order[start : start + batch_size]]


def sample_one_hop(
    edge_index: np.ndarray,
    seeds: np.ndarray,
    *,
    fanout: int,
    rng_seed: int,
) -> np.ndarray:
    """Reference deterministic directed neighbor sampler for adapter smoke tests."""

    edges = np.asarray(edge_index, dtype=int)
    seeds = np.asarray(seeds, dtype=int)
    rng = np.random.default_rng(rng_seed)
    selected: list[int] = []
    for node in seeds:
        incident = np.flatnonzero(edges[1] == node)
        if len(incident) > fanout:
            incident = np.sort(rng.choice(incident, size=fanout, replace=False))
        selected.extend(int(index) for index in incident)
    return np.asarray(sorted(set(selected)), dtype=int)
