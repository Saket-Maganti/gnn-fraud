"""
data/datasets/tfinance.py

T-Finance loader (Tang et al., KDD 2022; "Rethinking GNN-based Fraud Detection").

T-Finance is a smaller, temporal transaction graph:
  * ~39K nodes (accounts)
  * ~21M edges (transactions, with timestamps)
  * 10-dim hand-crafted node features
  * Binary labels (anomaly / normal), ~4.6% anomaly rate
  * Edge timestamps we bucketise to per-node times (same recipe as DGraphFin)

Distribution varies by source; the most convenient is the DGL-bundled copy
released with the paper. We accept two equivalent layouts:

  (a) A single ``tfinance.npz`` archive with keys
        x, y, edge_index, edge_timestamp
      (this is what scripts/download_tfinance.py writes out).

  (b) A DGL ``.bin`` graph dumped by ``dgl.data.utils.save_graphs``, with
      node features ``feature`` and labels ``label`` and a 1-D edge tensor
      ``timestamp``. Set ``raw_path`` to the .bin file to use this path.

Either way, place the archive at ``data/raw/tfinance.npz`` (or .bin) before
running any experiment on this dataset.

Label semantics (remap to the shared schema):
    raw 1 (anomaly) → 1 (illicit)
    raw 0 (normal)  → 2 (licit)
    missing labels  → 0 (unknown, not used anywhere)

We reuse the same quantile bucketisation + chronological split as DGraphFin
so the "evaluation protocol reverses rankings" narrative translates 1:1
across the three datasets.
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
from data.datasets.dgraphfin import _assign_node_timestamps, _bucketise  # noqa: E402
from data.feature_scaling import resolve_scaler_mode, scale_features  # noqa: E402


RAW_PATH_DEFAULT = os.path.join(REPO_ROOT, "data", "raw", "tfinance.npz")


def _load_dgl_bin(path: str):
    """Optional path: read a DGL-bundled .bin file if that's what you have."""
    try:
        import dgl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Reading a DGL .bin T-Finance archive requires the `dgl` "
            "package. Either install dgl or convert to the NPZ layout "
            "(see scripts/download_tfinance.py)."
        ) from exc
    from dgl.data.utils import load_graphs
    graphs, _ = load_graphs(path)
    g = graphs[0]
    x = g.ndata["feature"].numpy().astype(np.float32)
    y = g.ndata["label"].numpy().astype(np.int64)
    src, dst = g.edges()
    edge_index = np.stack([src.numpy(), dst.numpy()], axis=0).astype(np.int64)
    if "timestamp" in g.edata:
        edge_timestamp = g.edata["timestamp"].numpy().astype(np.int64)
    else:
        edge_timestamp = np.arange(edge_index.shape[1], dtype=np.int64)
    return x, y, edge_index, edge_timestamp


def load_tfinance_dataset(
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
            f"T-Finance archive not found: {raw_path}\n"
            "Place the 'tfinance.npz' (or .bin) file under data/raw/. The "
            "dataset ships with 'Rethinking GNN-based Fraud Detection' "
            "(Tang et al., KDD 2022) and is available from the authors' "
            "repo: https://github.com/YingtongDou/CARE-GNN (mirrored "
            "copies exist on DGL and the BGNN benchmark)."
        )

    if raw_path.endswith(".bin"):
        x_raw, y_raw, edge_index_np, edge_timestamp = _load_dgl_bin(raw_path)
    else:
        arch = np.load(raw_path)
        x_raw          = arch["x"].astype(np.float32)
        y_raw          = arch["y"].astype(np.int64)
        edge_index_np  = arch["edge_index"].astype(np.int64)
        if edge_index_np.shape[0] != 2:
            edge_index_np = edge_index_np.T
        if "edge_timestamp" in arch.files:
            edge_timestamp = arch["edge_timestamp"].astype(np.int64)
        else:
            # fall back to edge order as timestamp (monotone proxy)
            edge_timestamp = np.arange(edge_index_np.shape[1], dtype=np.int64)

    num_nodes = x_raw.shape[0]

    # Remap to shared {0: unknown, 1: illicit, 2: licit} schema.
    y = np.zeros(num_nodes, dtype=np.int64)
    y[y_raw == 1] = 1
    y[y_raw == 0] = 2

    node_ts   = _assign_node_timestamps(num_nodes, edge_index_np, edge_timestamp)
    time_step = _bucketise(node_ts, n_buckets=n_buckets)

    if split is None:
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
        "source":      "tfinance",
        "url":         "https://github.com/YingtongDou/CARE-GNN",
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
        name       = "tfinance",
        meta       = meta,
    )
