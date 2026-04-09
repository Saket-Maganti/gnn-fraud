"""
data/graph_builder.py
Multiple graph construction strategies for the Elliptic dataset.

Strategies
----------
  original     : Raw transaction edges (no change)
  similarity   : Within-timestep cosine-similarity edges above threshold
  feature_knn  : Within-timestep k-NN in feature space
  temporal     : Cross-timestep k-NN edges bridging adjacent time slices
  augmented    : Original + similarity edges combined

All return a modified copy of the PyG Data object with a new edge_index.
Edge attributes are NOT recomputed for constructed graphs (set to None).

Design choice: within-timestep construction keeps the problem tractable.
Each timestep has ~4 K nodes, so pairwise ops stay under 16 M pairs.
"""

import numpy as np
import torch
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _edges_from_pairs(src_global: np.ndarray, dst_global: np.ndarray) -> torch.Tensor:
    edge_index = torch.tensor(np.stack([src_global, dst_global]), dtype=torch.long)
    return to_undirected(edge_index)


def _timestep_groups(data: Data):
    """Return dict {t: node_indices_array} for all timesteps."""
    ts = data.time_step.numpy()
    groups = {}
    for t in np.unique(ts):
        groups[int(t)] = np.where(ts == t)[0]
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 – cosine similarity within each timestep
# ─────────────────────────────────────────────────────────────────────────────

def build_similarity_graph(data: Data, threshold: float = 0.92) -> Data:
    """
    Add edges between nodes in the same timestep whose cosine similarity
    exceeds `threshold`.  Returns a new Data object with updated edge_index.
    """
    feat = data.x.numpy()
    groups = _timestep_groups(data)

    all_src, all_dst = [], []
    for t, idx in tqdm(groups.items(), desc="similarity-graph", leave=False):
        if len(idx) < 2:
            continue
        sims = cosine_similarity(feat[idx])           # (n_t, n_t)
        np.fill_diagonal(sims, 0.0)
        src_t, dst_t = np.where(sims >= threshold)
        # keep src < dst to avoid duplicates before to_undirected
        mask = src_t < dst_t
        all_src.append(idx[src_t[mask]])
        all_dst.append(idx[dst_t[mask]])

    if all_src:
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        new_ei = _edges_from_pairs(src, dst)
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)

    out = deepcopy(data)
    out.edge_index = new_ei
    out.edge_attr  = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 – k-NN in feature space within each timestep
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_knn_graph(data: Data, k: int = 5) -> Data:
    """
    Connect each node to its k nearest neighbours (L2 distance) in the same
    timestep.  Returns a new Data object.
    """
    feat = data.x.numpy()
    groups = _timestep_groups(data)

    all_src, all_dst = [], []
    for t, idx in tqdm(groups.items(), desc="knn-graph", leave=False):
        n = len(idx)
        if n < 2:
            continue
        kk = min(k + 1, n)                             # +1: first hit is self
        nbrs = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(feat[idx])
        _, knn_idx = nbrs.kneighbors(feat[idx])        # (n, k+1)
        for local_i in range(n):
            for local_j in knn_idx[local_i, 1:]:       # skip self (col 0)
                if local_i < local_j:                  # deduplicate
                    all_src.append(idx[local_i])
                    all_dst.append(idx[local_j])

    if all_src:
        src = np.array(all_src, dtype=np.int64)
        dst = np.array(all_dst, dtype=np.int64)
        new_ei = _edges_from_pairs(src, dst)
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)

    out = deepcopy(data)
    out.edge_index = new_ei
    out.edge_attr  = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 – cross-timestep temporal edges
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_graph(data: Data, k: int = 3, window: int = 1) -> Data:
    """
    For each timestep t, connect every node in t to its k nearest neighbours
    in timestep t+1 (and t+2 if window=2, etc.) using L2 feature distance.

    This captures temporal continuity: how fraud patterns evolve across slices.
    """
    feat = data.x.numpy()
    groups = _timestep_groups(data)
    sorted_times = sorted(groups.keys())

    all_src, all_dst = [], []
    for i, t in enumerate(tqdm(sorted_times, desc="temporal-graph", leave=False)):
        idx_t = groups[t]
        for delta in range(1, window + 1):
            if i + delta >= len(sorted_times):
                break
            t2 = sorted_times[i + delta]
            idx_t2 = groups[t2]
            if len(idx_t) == 0 or len(idx_t2) == 0:
                continue
            kk = min(k, len(idx_t2))
            nbrs = NearestNeighbors(n_neighbors=kk, algorithm="auto").fit(feat[idx_t2])
            _, knn_idx = nbrs.kneighbors(feat[idx_t])   # each node in t → k nodes in t2
            for local_i, neighbours in enumerate(knn_idx):
                for local_j in neighbours:
                    all_src.append(idx_t[local_i])
                    all_dst.append(idx_t2[local_j])

    if all_src:
        src = np.array(all_src, dtype=np.int64)
        dst = np.array(all_dst, dtype=np.int64)
        new_ei = _edges_from_pairs(src, dst)
    else:
        new_ei = torch.zeros((2, 0), dtype=torch.long)

    out = deepcopy(data)
    out.edge_index = new_ei
    out.edge_attr  = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4 – augmented (original + similarity)
# ─────────────────────────────────────────────────────────────────────────────

def build_augmented_graph(data: Data, threshold: float = 0.92) -> Data:
    """
    Combines original transaction edges with within-timestep similarity edges.
    """
    sim_data = build_similarity_graph(data, threshold=threshold)
    combined = torch.cat([data.edge_index, sim_data.edge_index], dim=1)
    combined = coalesce(combined, num_nodes=data.num_nodes)
    combined = to_undirected(combined)

    out = deepcopy(data)
    out.edge_index = combined
    out.edge_attr  = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(data: Data, graph_type: str, **kwargs) -> Data:
    """
    Build a graph variant.

    Parameters
    ----------
    data        : original PyG Data object (Elliptic)
    graph_type  : one of original | similarity | feature_knn | temporal | augmented
    **kwargs    : forwarded to the specific builder (threshold, k, window, …)
    """
    if graph_type == "original":
        return deepcopy(data)
    elif graph_type == "similarity":
        return build_similarity_graph(data, threshold=kwargs.get("threshold", 0.92))
    elif graph_type == "feature_knn":
        return build_feature_knn_graph(data, k=kwargs.get("k", 5))
    elif graph_type == "temporal":
        return build_temporal_graph(
            data, k=kwargs.get("k", 3), window=kwargs.get("window", 1)
        )
    elif graph_type == "augmented":
        return build_augmented_graph(data, threshold=kwargs.get("threshold", 0.92))
    elif graph_type == "no_edges":
        out = deepcopy(data)
        out.edge_index = torch.zeros((2, 0), dtype=torch.long)
        out.edge_attr  = None
        return out
    else:
        raise ValueError(f"Unknown graph_type: '{graph_type}'. "
                         f"Choose from: original, similarity, feature_knn, temporal, "
                         f"augmented, no_edges")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def graph_summary(data: Data, name: str = "") -> dict:
    """Return basic topology statistics for a Data object."""
    N = data.num_nodes
    E = data.edge_index.shape[1]
    # For undirected graph, density = E / (N*(N-1))
    density = E / max(N * (N - 1), 1)
    deg = torch.zeros(N, dtype=torch.long)
    deg.scatter_add_(0, data.edge_index[0], torch.ones(E, dtype=torch.long))
    avg_deg    = deg.float().mean().item()
    max_deg    = deg.max().item()
    isolated   = (deg == 0).sum().item()
    return {
        "graph_type":   name,
        "num_nodes":    N,
        "num_edges":    E,
        "density":      round(density, 8),
        "avg_degree":   round(avg_deg, 4),
        "max_degree":   max_deg,
        "isolated_nodes": isolated,
    }
