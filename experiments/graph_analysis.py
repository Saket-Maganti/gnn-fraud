"""
experiments/graph_analysis.py
Computes topology statistics for every graph construction variant.

Metrics per graph
-----------------
  num_nodes, num_edges, density, avg_degree, max_degree, isolated_nodes,
  avg_clustering (sampled), connected_components

Output: results/graph_stats.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import load_elliptic_raw, preprocess
from data.graph_builder import build_graph, graph_summary

VARIANTS = ["original", "similarity", "feature_knn", "temporal", "augmented"]

# Subset of nodes to use for expensive metrics (clustering, components)
SAMPLE_SIZE = 5_000


# ─────────────────────────────────────────────────────────────────────────────

def _clustering_sample(edge_index: torch.Tensor, num_nodes: int,
                       sample_size: int = SAMPLE_SIZE) -> float:
    """
    Compute average clustering coefficient on a random sample of nodes.
    Uses NetworkX on the subgraph induced by sampled nodes + their neighbours.
    """
    rng   = np.random.default_rng(42)
    n_sample = min(sample_size, num_nodes)
    sampled  = rng.choice(num_nodes, size=n_sample, replace=False)
    sampled_set = set(sampled.tolist())

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    # Include edges where both endpoints are in sample
    mask = np.isin(src, sampled) & np.isin(dst, sampled)
    G = nx.Graph()
    G.add_nodes_from(sampled)
    G.add_edges_from(zip(src[mask].tolist(), dst[mask].tolist()))

    clust = nx.average_clustering(G)
    return round(float(clust), 6)


def _connected_components(edge_index: torch.Tensor, num_nodes: int) -> int:
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    return nx.number_connected_components(G)


# ─────────────────────────────────────────────────────────────────────────────

def run_graph_analysis(results_dir: str = "results") -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("GRAPH CONSTRUCTION ANALYSIS")
    print("=" * 60)

    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)

    rows = []
    for gtype in VARIANTS:
        print(f"\n  [{gtype}] building …", end=" ", flush=True)
        gdata = build_graph(data, gtype)
        stats = graph_summary(gdata, name=gtype)
        print(f"nodes={stats['num_nodes']:,}  edges={stats['num_edges']:,}  "
              f"avg_deg={stats['avg_degree']:.2f}", end=" ", flush=True)

        # Clustering (expensive — sampled)
        print("clustering …", end=" ", flush=True)
        clust = _clustering_sample(gdata.edge_index, gdata.num_nodes)
        stats["avg_clustering_sampled"] = clust
        print(f"clust={clust:.4f}", end=" ", flush=True)

        # Connected components (can be slow for large graphs — skip if >500K edges)
        if gdata.edge_index.shape[1] <= 500_000:
            print("components …", end=" ", flush=True)
            cc = _connected_components(gdata.edge_index, gdata.num_nodes)
            stats["connected_components"] = cc
            print(f"cc={cc}", end="")
        else:
            stats["connected_components"] = -1  # too large to compute
            print("components=SKIP (>500K edges)", end="")

        rows.append(stats)
        print()

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "graph_stats.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Graph stats saved → {out_path}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_graph_analysis()
