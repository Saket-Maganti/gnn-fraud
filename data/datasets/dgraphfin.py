"""
data/datasets/dgraphfin.py

DGraphFin loader (Huang et al., NeurIPS 2022; https://dgraph.xinye.com/dataset).

DGraphFin is a large-scale real-world financial fraud graph:
  * ~3.7M nodes (registered users)
  * ~4.3M directed edges (contacts with timestamps)
  * 17-dim anonymised node features
  * Binary labels with an "unlabeled" class
  * Timestamps on edges (we bucketise to per-node times)

The dataset is distributed as a single ``dgraphfin.npz`` archive. Download:

    # Visit https://dgraph.xinye.com/ to request access, then place the file at:
    #   data/raw/dgraphfin.npz

The NPZ exposes:
    x           (N, 17)         node features
    y           (N,)             {0: normal, 1: fraud, 2: background, 3: background}
    edge_index  (2, E)           directed edges
    edge_type   (E,)             edge type (ignored by default)
    edge_timestamp (E,)          integer timestamp

We follow the paper's conventions:
  * Keep only classes 0 (normal) and 1 (fraud) as labeled (classes 2/3 = unknown).
  * Remap labels: {0: 2 (licit, using Elliptic conventions), 1: 1 (illicit), else: 0}.
  * Bucketise timestamps into ``n_buckets`` equal-count bins to build time_step.
  * Chronological split on those buckets: 70 / 10 / 20 by time.

This layout keeps the metrics code (`y == 1` is fraud) identical to Elliptic.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from torch_geometric.utils import to_undirected

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.datasets.base import FraudDataset, SplitBoundaries  # noqa: E402
from data.feature_scaling import resolve_scaler_mode, scale_features  # noqa: E402


RAW_PATH_DEFAULT = os.path.join(REPO_ROOT, "data", "raw", "dgraphfin.npz")


def _assign_node_timestamps(
    num_nodes:       int,
    edge_index:      np.ndarray,
    edge_timestamp:  np.ndarray,
) -> np.ndarray:
    """Assign each node the median timestamp of its incident edges.

    Nodes with no incident edges fall back to the global median so no node
    is orphaned from a time bucket. Using the median (not the mean) keeps
    the per-node time stable against bursty outlier edges.
    """
    ts_by_node = [list() for _ in range(num_nodes)]
    src = edge_index[0]; dst = edge_index[1]
    for i in range(edge_timestamp.shape[0]):
        ts_by_node[int(src[i])].append(int(edge_timestamp[i]))
        ts_by_node[int(dst[i])].append(int(edge_timestamp[i]))
    global_median = int(np.median(edge_timestamp))
    out = np.empty(num_nodes, dtype=np.int64)
    for i, lst in enumerate(ts_by_node):
        out[i] = int(np.median(lst)) if lst else global_median
    return out


def _bucketise(ts: np.ndarray, n_buckets: int) -> np.ndarray:
    """Equal-count quantile bins, 1-indexed so t=0 is reserved."""
    quantiles = np.quantile(ts, np.linspace(0, 1, n_buckets + 1))
    quantiles[0]  -= 1          # ensure min value falls into bucket 1
    quantiles[-1] += 1          # ensure max value falls into bucket n_buckets
    bins = np.digitize(ts, quantiles[1:-1], right=False) + 1
    return bins.astype(np.int64)


def load_dgraphfin_dataset(
    raw_path:   Optional[str] = None,
    n_buckets:  int = 20,
    split:      Optional[SplitBoundaries] = None,
    normalize:  Optional[bool] = None,
    scaler_mode: Optional[str] = "train_only",
) -> FraudDataset:
    mode = resolve_scaler_mode(
        scaler_mode,
        normalize=normalize,
        default="train_only",
    )
    raw_path = raw_path or RAW_PATH_DEFAULT
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"DGraphFin archive not found: {raw_path}\n"
            "Download from https://dgraph.xinye.com/ and place the "
            "'dgraphfin.npz' file at data/raw/dgraphfin.npz."
        )

    arch = np.load(raw_path)
    x_raw          = arch["x"].astype(np.float32)
    y_raw          = arch["y"].astype(np.int64)
    edge_index_np  = arch["edge_index"].astype(np.int64)
    edge_timestamp = arch["edge_timestamp"].astype(np.int64)

    if edge_index_np.shape[0] != 2:
        edge_index_np = edge_index_np.T

    num_nodes = x_raw.shape[0]

    # Label remapping: 1 (fraud) → 1, 0 (normal) → 2, everything else → 0
    # keeps the Elliptic invariant that ``y == 1`` is the positive class.
    y = np.zeros(num_nodes, dtype=np.int64)
    y[y_raw == 1] = 1
    y[y_raw == 0] = 2
    # classes 2 and 3 in DGraphFin are background users without ground truth,
    # so they stay at 0 (unknown) and never appear in any mask.

    # Bucketise edge timestamps to per-node times, then split by time.
    node_ts   = _assign_node_timestamps(num_nodes, edge_index_np, edge_timestamp)
    time_step = _bucketise(node_ts, n_buckets=n_buckets)

    if split is None:
        # 70 / 10 / 20 chronological split on buckets.
        split = SplitBoundaries(
            train_max = int(round(0.70 * n_buckets)),
            val_max   = int(round(0.80 * n_buckets)),
            test_max  = n_buckets,
        )

    t = torch.from_numpy(time_step)
    labeled = torch.from_numpy(y != 0)
    train_mask = labeled & (t <= split.train_max)
    val_mask   = labeled & (t > split.train_max) & (t <= split.val_max)
    test_mask  = labeled & (t > split.val_max)   & (t <= split.test_max)

    x_raw, mode = scale_features(x_raw, train_mask.numpy(), mode)

    edge_index = torch.from_numpy(edge_index_np)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    meta = {
        "source":      "dgraphfin",
        "url":         "https://dgraph.xinye.com/",
        "n_buckets":   n_buckets,
        "split":       split.describe(),
        "scaler_mode": mode,
    }
    return FraudDataset.build(
        x          = torch.from_numpy(x_raw),
        y          = torch.from_numpy(y),
        edge_index = edge_index,
        time_step  = t,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        name       = "dgraphfin",
        meta       = meta,
    )
