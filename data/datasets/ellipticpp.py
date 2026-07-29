"""
data/datasets/ellipticpp.py

Elliptic++ ACTORS loader (Youssef et al., 2023; ``git-disl/EllipticPlusPlus``;
arXiv:2306.06108).

This is the wallet-address ("actors") graph of Elliptic++: address nodes with
56 time-dependent features, illicit / licit / unknown labels, and an
address-to-address (``AddrAddr``) graph derived from Bitcoin input/output
relationships. It shares the 49 time-step temporal structure and the
illicit-node classification task with Elliptic.

POSITIONING (important — keep this honest):
    Elliptic++ is Bitcoin-derived, *like Elliptic*. It is used here as
    **temporal-robustness corroboration** (a correlated-domain second temporal
    graph), **NOT** as independent domain generalization. ``dgraphfin`` remains
    the independent second-dataset evidence. Do not cite Elliptic++ as a
    separate domain or as evidence of cross-domain generalization.

Canonical construction (matches the authors' Actors classification task):
  * Node = one ``(address, time step)`` OCCURRENCE — 1,268,260 occurrences over
    822,942 unique wallets. The 56 features are time-dependent per occurrence,
    so the same wallet active in several time steps is several nodes (exactly
    like an Elliptic transaction belongs to one time step).
  * Edges = ``AddrAddr`` edges materialised WITHIN a single time step: an edge
    ``(u, v)`` connects the occurrence of ``u`` and the occurrence of ``v`` in
    the same time step (the paper states AddrAddr edges inherit the time step
    of their source transaction).
  * Labels are per-address (time-invariant): raw ``1 = illicit``,
    ``2 = licit``, ``3 = unknown``. We remap ``3 -> 0`` so the schema matches
    the repo invariant ``{0: unknown, 1: illicit, 2: licit}`` and metrics stay
    on ``y == 1``.
  * Temporal split mirrors the published Elliptic split (test = t35..t49); we
    shave t31..t34 as a held-out validation window — the same recipe as
    ``elliptic.py`` — so TPC/TTA and early stopping never peek at test.

Download (manual, like DGraphFin — no auto-download, fail loud if absent):
    The actors CSVs ship via the Google Drive linked from
    https://github.com/git-disl/EllipticPlusPlus . Place the three files under
    ``data/raw/ellipticpp/``::

        data/raw/ellipticpp/wallets_features.csv
        data/raw/ellipticpp/wallets_classes.csv
        data/raw/ellipticpp/AddrAddr_edgelist.csv
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_undirected

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.datasets.base import FraudDataset, SplitBoundaries  # noqa: E402
from data.feature_scaling import resolve_scaler_mode, scale_features  # noqa: E402


RAW_DIR_DEFAULT = os.path.join(REPO_ROOT, "data", "raw", "ellipticpp")
_FEATURES_FILE = "wallets_features.csv"
_CLASSES_FILE = "wallets_classes.csv"
_EDGES_FILE = "AddrAddr_edgelist.csv"

# Mirror the published Elliptic split: train t1..t30, val t31..t34, test t35..t49.
# (Authors use train t1..t34 / test t35..t49; we carve t31..t34 as the val window.)
_DEFAULT_SPLIT = SplitBoundaries(train_max=30, val_max=34, test_max=49)


def _require_files(raw_dir: str) -> None:
    missing = [f for f in (_FEATURES_FILE, _CLASSES_FILE, _EDGES_FILE)
               if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Elliptic++ actors files not found in {raw_dir}: missing {missing}.\n"
            "Download the Actors dataset CSVs from the Google Drive linked at "
            "https://github.com/git-disl/EllipticPlusPlus and place "
            f"{_FEATURES_FILE}, {_CLASSES_FILE}, {_EDGES_FILE} under "
            "data/raw/ellipticpp/. No synthetic fallback is provided by design."
        )


def _find_time_col(columns) -> Optional[str]:
    for c in columns:
        cl = str(c).strip().lower().replace("_", " ")
        if cl in ("time step", "timestep", "time") or cl.startswith("time"):
            return c
    return None


def load_ellipticpp_dataset(
    raw_dir:    Optional[str] = None,
    split:      Optional[SplitBoundaries] = None,
    normalize:  Optional[bool] = None,
    scaler_mode: Optional[str] = "train_only",
) -> FraudDataset:
    mode = resolve_scaler_mode(scaler_mode, normalize=normalize, default="train_only")
    raw_dir = raw_dir or RAW_DIR_DEFAULT
    _require_files(raw_dir)
    split = split or _DEFAULT_SPLIT

    # ── Node features: one row per (address, time step) occurrence ───────────
    feats = pd.read_csv(os.path.join(raw_dir, _FEATURES_FILE))
    addr_col = feats.columns[0]
    time_col = _find_time_col(feats.columns)
    if time_col is None:
        raise ValueError(
            f"Could not find a 'Time step' column in {_FEATURES_FILE}; "
            f"columns were {list(feats.columns)}. Share the header so the "
            "loader can be aligned."
        )
    feature_cols = [c for c in feats.columns if c not in (addr_col, time_col)]
    if not feature_cols:
        raise ValueError(f"No feature columns left in {_FEATURES_FILE} after "
                         f"dropping address '{addr_col}' and time '{time_col}'.")

    addr = feats[addr_col].to_numpy()
    time_step = feats[time_col].to_numpy().astype(np.int64)
    x_raw = feats[feature_cols].to_numpy().astype(np.float32)
    num_nodes = x_raw.shape[0]
    node_ids = np.arange(num_nodes, dtype=np.int64)

    # ── Labels: per-address, remap raw {1 illicit, 2 licit, 3 unknown} ──────
    cls = pd.read_csv(os.path.join(raw_dir, _CLASSES_FILE))
    cls_addr_col = cls.columns[0]
    cls_label_col = cls.columns[1] if len(cls.columns) > 1 else cls.columns[-1]
    raw_label = pd.to_numeric(cls[cls_label_col], errors="coerce")
    addr_to_label = dict(zip(cls[cls_addr_col].to_numpy(), raw_label.to_numpy()))

    y = np.zeros(num_nodes, dtype=np.int64)  # default 0 = unknown
    mapped = np.array([addr_to_label.get(a, np.nan) for a in addr], dtype=float)
    y[mapped == 1] = 1   # illicit  -> 1 (positive class)
    y[mapped == 2] = 2   # licit    -> 2
    # raw 3 (unknown) and any address absent from classes stay 0 (unknown).

    # ── Edges: AddrAddr materialised within a single time step ──────────────
    edges = pd.read_csv(os.path.join(raw_dir, _EDGES_FILE))
    src_col, dst_col = edges.columns[0], edges.columns[1]
    edge_time_col = _find_time_col(edges.columns)

    occ = pd.DataFrame({"_addr": addr, "_t": time_step, "_node": node_ids})

    e = edges.rename(columns={src_col: "_src_addr", dst_col: "_dst_addr"})
    occ_src = occ.rename(columns={"_addr": "_src_addr", "_node": "_src_node"})
    occ_dst = occ.rename(columns={"_addr": "_dst_addr", "_node": "_dst_node"})

    if edge_time_col is not None:
        # Edgelist already carries a time step: join each endpoint at that step.
        e = e.rename(columns={edge_time_col: "_t"})
        e["_t"] = pd.to_numeric(e["_t"], errors="coerce").astype("Int64")
        m = e.merge(occ_src, on=["_src_addr", "_t"]).merge(occ_dst, on=["_dst_addr", "_t"])
    else:
        # No per-edge time: connect endpoints that co-occur in the same step.
        m = (e.merge(occ_src, on="_src_addr")
               .merge(occ_dst, on=["_dst_addr", "_t"]))

    if len(m) == 0:
        raise ValueError(
            "Elliptic++ edge construction produced 0 edges. This usually means "
            "the address id space in AddrAddr_edgelist.csv does not match "
            "wallets_features.csv (e.g. raw addresses vs integer indices). "
            "Share the file headers so the loader can be aligned."
        )

    edge_index_np = np.stack([m["_src_node"].to_numpy(), m["_dst_node"].to_numpy()]).astype(np.int64)

    # ── Scale, mask, assemble ───────────────────────────────────────────────
    t = torch.from_numpy(time_step)
    labeled = torch.from_numpy(y != 0)
    train_mask = labeled & (t <= split.train_max)
    val_mask   = labeled & (t > split.train_max) & (t <= split.val_max)
    test_mask  = labeled & (t > split.val_max)   & (t <= split.test_max)

    x_raw, mode = scale_features(x_raw, train_mask.numpy(), mode)

    edge_index = torch.from_numpy(edge_index_np)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    meta = {
        "source":      "ellipticpp_actors",
        "url":         "https://github.com/git-disl/EllipticPlusPlus",
        "positioning": "temporal_robustness_correlated_domain",  # NOT domain generalization
        "n_timesteps": int(t.max().item()),
        "unique_wallets": int(pd.unique(addr).shape[0]),
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
        name       = "ellipticpp",
        meta       = meta,
    )
