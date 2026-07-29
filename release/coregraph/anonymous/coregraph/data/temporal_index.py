"""Vectorised first-observation and temporal-window indexing."""

from __future__ import annotations

import numpy as np


def first_incident_timestamp(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_timestamps: np.ndarray,
    *,
    isolated_value: int | float | None = None,
) -> np.ndarray:
    """Return each node's first incident event without future lifecycle access."""

    edges = np.asarray(edge_index, dtype=int)
    times = np.asarray(edge_timestamps)
    if edges.shape != (2, len(times)):
        raise ValueError("edge_index and timestamps must align")
    if num_nodes < 0 or np.any(edges < 0) or np.any(edges >= num_nodes):
        raise ValueError("edge endpoints outside node range")
    if len(times) == 0:
        fill = 0 if isolated_value is None else isolated_value
        return np.full(num_nodes, fill)
    if not np.issubdtype(times.dtype, np.number):
        raise ValueError("edge timestamps must be numeric")
    first = np.full(num_nodes, np.inf, dtype=float)
    np.minimum.at(first, edges[0], times)
    np.minimum.at(first, edges[1], times)
    fill = float(np.min(times)) if isolated_value is None else float(isolated_value)
    first[np.isinf(first)] = fill
    if np.issubdtype(times.dtype, np.integer) and float(fill).is_integer():
        return first.astype(times.dtype)
    return first


def quantile_buckets(values: np.ndarray, n_buckets: int) -> np.ndarray:
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    values = np.asarray(values)
    if values.size == 0:
        return np.asarray([], dtype=int)
    boundaries = np.quantile(values, np.linspace(0, 1, n_buckets + 1))
    boundaries[0] = np.nextafter(boundaries[0], -np.inf)
    boundaries[-1] = np.nextafter(boundaries[-1], np.inf)
    return np.digitize(values, boundaries[1:-1], right=False).astype(int) + 1


def historical_edge_mask(
    edge_timestamps: np.ndarray,
    *,
    cutoff: float,
    window: float | None = None,
) -> np.ndarray:
    times = np.asarray(edge_timestamps)
    keep = times <= cutoff
    if window is not None:
        if window <= 0:
            raise ValueError("historical window must be positive")
        keep &= times > cutoff - window
    return keep
