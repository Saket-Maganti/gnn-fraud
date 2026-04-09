"""
data/dataset.py
Elliptic Bitcoin Dataset loader.
Supports node features, edge features, temporal splits, and per-timestep slicing.

Download from Kaggle:
    kaggle datasets download ellipticco/elliptic-data-set -p data/raw/ --unzip

Files expected in data/raw/:
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional

RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


# ─────────────────────────────────────────────────────────────────────────────
# Raw loading
# ─────────────────────────────────────────────────────────────────────────────

def load_elliptic_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "features": os.path.join(RAW_DIR, "elliptic_txs_features.csv"),
        "classes":  os.path.join(RAW_DIR, "elliptic_txs_classes.csv"),
        "edges":    os.path.join(RAW_DIR, "elliptic_txs_edgelist.csv"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {name}: {p}\n"
                "Download: kaggle datasets download ellipticco/elliptic-data-set "
                "-p data/raw/ --unzip"
            )
    features = pd.read_csv(paths["features"], header=None)
    classes  = pd.read_csv(paths["classes"])
    edges    = pd.read_csv(paths["edges"])
    return features, classes, edges


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    features:   pd.DataFrame,
    classes:    pd.DataFrame,
    edges:      pd.DataFrame,
    normalize:  bool = True,
    use_edge_features: bool = False,
) -> Data:
    """
    Build a PyG Data object.

    Node feature layout (after col 0 = txId stripped):
        col 1      : time_step  (1-49)
        cols 2-95  : 94 local features
        cols 96-166: 72 aggregated neighbourhood features

    Returns Data with:
        x, y, edge_index, edge_attr (if use_edge_features)
        node_ids, time_step, train_mask, test_mask
    """
    node_ids  = features.iloc[:, 0].values
    time_step = features.iloc[:, 1].values.astype(int)
    feat_mat  = features.iloc[:, 2:].values.astype(np.float32)

    if normalize:
        feat_mat = StandardScaler().fit_transform(feat_mat)

    # Labels: unknown=0, illicit=1, licit=2
    id2label  = dict(zip(classes["txId"], classes["class"].astype(str)))
    label_map = {"unknown": 0, "1": 1, "2": 2}
    y = np.array(
        [label_map.get(id2label.get(txid, "unknown"), 0) for txid in node_ids],
        dtype=np.int64
    )

    # Edges
    id2idx    = {txid: idx for idx, txid in enumerate(node_ids)}
    valid     = edges["txId1"].isin(id2idx) & edges["txId2"].isin(id2idx)
    valid_edges = edges[valid].copy()
    src = valid_edges["txId1"].map(id2idx).values
    dst = valid_edges["txId2"].map(id2idx).values
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_index = to_undirected(edge_index)

    # Edge features (L3+): derive from node features of src/dst
    edge_attr = None
    if use_edge_features:
        edge_attr = _compute_edge_features(feat_mat, edge_index)

    # Temporal train/test split (Pareja et al. 2020)
    labeled    = y != 0
    train_mask = labeled & (time_step <= 34)
    test_mask  = labeled & (time_step > 34)

    data = Data(
        x          = torch.tensor(feat_mat,   dtype=torch.float),
        y          = torch.tensor(y,          dtype=torch.long),
        edge_index = edge_index,
        edge_attr  = edge_attr,
    )
    data.node_ids   = torch.tensor(node_ids,  dtype=torch.long)
    data.time_step  = torch.tensor(time_step, dtype=torch.long)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.test_mask  = torch.tensor(test_mask,  dtype=torch.bool)
    return data


def _compute_edge_features(feat_mat: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Derive 6-dim edge features from node features.
    Features: [src_local_mean, dst_local_mean, diff_abs_mean,
               src_agg_mean, dst_agg_mean, agg_diff_abs_mean]
    These proxy tx amount, fee, and neighbourhood activity differences.
    """
    src_idx = edge_index[0].numpy()
    dst_idx = edge_index[1].numpy()

    src_local = feat_mat[src_idx, :94].mean(axis=1, keepdims=True)
    dst_local = feat_mat[dst_idx, :94].mean(axis=1, keepdims=True)
    diff_loc  = np.abs(src_local - dst_local)
    src_agg   = feat_mat[src_idx, 94:].mean(axis=1, keepdims=True)
    dst_agg   = feat_mat[dst_idx, 94:].mean(axis=1, keepdims=True)
    diff_agg  = np.abs(src_agg - dst_agg)

    edge_feat = np.concatenate(
        [src_local, dst_local, diff_loc, src_agg, dst_agg, diff_agg], axis=1
    ).astype(np.float32)
    return torch.tensor(edge_feat, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(data: Data) -> torch.Tensor:
    train_y   = data.y[data.train_mask]
    n_illicit = (train_y == 1).sum().item()
    n_licit   = (train_y == 2).sum().item()
    n_total   = n_illicit + n_licit
    w_illicit = n_total / (2 * max(n_illicit, 1))
    w_licit   = n_total / (2 * max(n_licit, 1))
    # Cap weights at 5.0 to prevent collapse to minority class
    # sqrt scaling reduces extremity of weights
    import math
    w_illicit = math.sqrt(w_illicit)
    w_licit   = math.sqrt(w_licit)
    return torch.tensor([0.0, w_illicit, w_licit], dtype=torch.float)


def get_timestep_masks(data: Data) -> Dict[int, torch.Tensor]:
    """Return a mask dict {timestep: bool_mask} for test time steps 35-49."""
    masks = {}
    for t in range(35, 50):
        mask = data.test_mask & (data.time_step == t)
        if mask.sum().item() >= 5:
            masks[t] = mask
    return masks


def print_stats(data: Data):
    print("=" * 52)
    print("ELLIPTIC DATASET STATS")
    print("=" * 52)
    y = data.y
    print(f"  Nodes        : {data.num_nodes:,}")
    print(f"  Edges        : {data.num_edges:,}")
    print(f"  Features     : {data.num_node_features}")
    if data.edge_attr is not None:
        print(f"  Edge features: {data.edge_attr.shape[1]}")
    print(f"  Illicit (1)  : {(y==1).sum().item():,}  ({100*(y==1).float().mean():.1f}%)")
    print(f"  Licit   (2)  : {(y==2).sum().item():,}  ({100*(y==2).float().mean():.1f}%)")
    print(f"  Unknown (0)  : {(y==0).sum().item():,}  ({100*(y==0).float().mean():.1f}%)")
    print(f"  Train nodes  : {data.train_mask.sum().item():,}")
    print(f"  Test  nodes  : {data.test_mask.sum().item():,}")
    n_ill = (y[data.train_mask]==1).sum().item()
    n_lic = (y[data.train_mask]==2).sum().item()
    print(f"  Imbalance    : {n_lic/max(n_ill,1):.1f}:1 (licit:illicit)")
    print("=" * 52)