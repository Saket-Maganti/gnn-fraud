"""
scripts/addition9_subgraph_viz.py
Fraud community subgraph visualisation.

Visualises 2-3 fraud ego-graphs vs legitimate ego-graphs side by side.
Shows visually why graph structure should theoretically help.
Uses NetworkX + matplotlib. No additional dependencies needed.

Key visual findings to highlight:
- Fraud nodes tend to cluster densely with other fraud nodes
- Licit nodes have sparser, more diverse neighbourhoods
- BUT: this pattern only holds in early time steps (temporal drift)

Codepath : scripts/addition9_subgraph_viz.py
Runtime  : ~2 min
Output   : results/additions/fraud_subgraph_viz.png
           results/additions/community_stats.csv
"""

import sys, os, torch
import numpy as np
import pandas as pd
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/additions", exist_ok=True)


def get_ego_graph(data, seed_node, n_hops=2):
    """Extract n-hop ego graph around a seed node."""
    sub_nodes, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=seed_node, num_hops=n_hops,
        edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub_data = Data(
        x          = data.x[sub_nodes],
        y          = data.y[sub_nodes],
        edge_index = sub_ei,
    )
    return sub_data, sub_nodes


def compute_community_stats(data, node_type, n_samples=50, n_hops=1):
    """
    Compute structural statistics for ego-graphs of fraud vs licit nodes.
    Returns mean density, mean clustering coefficient, mean fraud neighbour %.
    """
    if node_type == "fraud":
        candidates = torch.where(
            data.train_mask & (data.y == 1))[0]
    else:
        candidates = torch.where(
            data.train_mask & (data.y == 2))[0]

    if len(candidates) == 0:
        return {}

    n_samples = min(n_samples, len(candidates))
    indices   = candidates[torch.randperm(len(candidates))[:n_samples]]

    stats = []
    for idx in indices:
        sub_nodes, sub_ei, _, _ = k_hop_subgraph(
            node_idx=idx.item(), num_hops=n_hops,
            edge_index=data.edge_index,
            relabel_nodes=True, num_nodes=data.num_nodes,
        )
        if sub_nodes.size(0) < 3:
            continue

        sub_data = Data(x=data.x[sub_nodes], y=data.y[sub_nodes],
                        edge_index=sub_ei)
        G = to_networkx(sub_data, to_undirected=True)

        # Fraction of neighbours that are also fraud
        orig_labels = data.y[sub_nodes].numpy()
        fraud_frac  = (orig_labels == 1).mean()

        # Graph structural properties
        density = nx.density(G)
        try:
            clustering = nx.average_clustering(G)
        except Exception:
            clustering = 0.0

        degree = np.mean([d for _, d in G.degree()])

        stats.append({
            "n_nodes":     sub_nodes.size(0),
            "density":     density,
            "clustering":  clustering,
            "mean_degree": degree,
            "fraud_frac":  fraud_frac,
        })

    if not stats:
        return {}

    return {k: np.mean([s[k] for s in stats]) for k in stats[0]}


def draw_ego_graph(ax, data, seed_node, n_hops, title, seed_color):
    """Draw a single ego-graph with fraud/licit node coloring."""
    sub_nodes, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=seed_node, num_hops=n_hops,
        edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )

    if sub_nodes.size(0) > 80:
        ax.text(0.5, 0.5, f"Subgraph too large\n({sub_nodes.size(0)} nodes)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return

    sub_data = Data(
        x=data.x[sub_nodes], y=data.y[sub_nodes], edge_index=sub_ei
    )
    G = to_networkx(sub_data, to_undirected=True)

    labels    = data.y[sub_nodes].numpy()
    # seed node is index 0 after relabeling (mapping gives original pos)
    seed_idx  = mapping.item() if mapping.numel() == 1 else 0

    node_colors = []
    node_sizes  = []
    for i in range(len(sub_nodes)):
        if i == seed_idx:
            node_colors.append(seed_color)
            node_sizes.append(300)
        elif labels[i] == 1:
            node_colors.append("#EF4444")
            node_sizes.append(120)
        elif labels[i] == 2:
            node_colors.append("#3B82F6")
            node_sizes.append(80)
        else:
            node_colors.append("#9CA3AF")
            node_sizes.append(60)

    try:
        pos = nx.spring_layout(G, seed=42, k=1.2)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                           edge_color="#6B7280", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.9)

    n_fraud = (labels == 1).sum()
    n_licit = (labels == 2).sum()
    n_unk   = (labels == 0).sum()
    density = nx.density(G)

    ax.set_title(
        f"{title}\n"
        f"N={len(sub_nodes)} | "
        f"Fraud={n_fraud} | Licit={n_licit} | Unk={n_unk}\n"
        f"Density={density:.3f}",
        fontsize=8, fontweight="bold"
    )
    ax.axis("off")


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data    = preprocess(f, c, e)

    print("Computing community statistics...")
    fraud_stats = compute_community_stats(data, "fraud",  n_samples=100)
    licit_stats = compute_community_stats(data, "licit",  n_samples=100)

    print("\nFraud node ego-graph statistics (1-hop, 100 samples):")
    for k, v in fraud_stats.items():
        print(f"  {k:<18}: {v:.4f}")
    print("\nLicit node ego-graph statistics (1-hop, 100 samples):")
    for k, v in licit_stats.items():
        print(f"  {k:<18}: {v:.4f}")

    # Save stats
    stats_df = pd.DataFrame([
        {"type": "Fraud (illicit)", **fraud_stats},
        {"type": "Licit",           **licit_stats},
    ])
    stats_df.to_csv("results/additions/community_stats.csv", index=False)

    # Select representative seed nodes
    fraud_seeds = torch.where(
        data.train_mask & (data.y == 1) &
        (data.time_step <= 20)
    )[0]
    licit_seeds = torch.where(
        data.train_mask & (data.y == 2) &
        (data.time_step <= 20)
    )[0]

    # Pick seeds with moderate-sized neighbourhoods (10-50 nodes)
    def pick_seed(candidates, target_size=20, n_hops=2):
        torch.manual_seed(42)
        perm = torch.randperm(min(len(candidates), 200))
        for i in perm:
            idx = candidates[i].item()
            sub_nodes, _, _, _ = k_hop_subgraph(
                idx, n_hops, data.edge_index,
                relabel_nodes=True, num_nodes=data.num_nodes)
            if 8 <= sub_nodes.size(0) <= 60:
                return idx
        return candidates[0].item()

    fraud_seed1 = pick_seed(fraud_seeds, target_size=20, n_hops=2)
    fraud_seed2 = pick_seed(
        fraud_seeds[fraud_seeds != fraud_seed1], target_size=15, n_hops=2)
    licit_seed1 = pick_seed(licit_seeds, target_size=20, n_hops=2)
    licit_seed2 = pick_seed(
        licit_seeds[licit_seeds != licit_seed1], target_size=15, n_hops=2)

    # Draw
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Fraud vs Licit Node Ego-Graphs (2-hop, training time steps 1-20)\n"
        "Red=Fraud  Blue=Licit  Gray=Unknown  Large=Seed node",
        fontsize=12, fontweight="bold"
    )

    draw_ego_graph(axes[0,0], data, fraud_seed1, 2,
                   "Fraud ego-graph #1", "#DC2626")
    draw_ego_graph(axes[0,1], data, fraud_seed2, 2,
                   "Fraud ego-graph #2", "#DC2626")
    draw_ego_graph(axes[1,0], data, licit_seed1, 2,
                   "Licit ego-graph #1", "#1D4ED8")
    draw_ego_graph(axes[1,1], data, licit_seed2, 2,
                   "Licit ego-graph #2", "#1D4ED8")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#DC2626", label="Fraud seed node"),
        mpatches.Patch(color="#1D4ED8", label="Licit seed node"),
        mpatches.Patch(color="#EF4444", label="Fraud neighbour"),
        mpatches.Patch(color="#3B82F6", label="Licit neighbour"),
        mpatches.Patch(color="#9CA3AF", label="Unknown"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("results/additions/fraud_subgraph_viz.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Stats comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Ego-Graph Structural Properties: Fraud vs Licit Nodes",
                 fontsize=12, fontweight="bold")

    metrics = [
        ("density",     "Graph Density"),
        ("clustering",  "Clustering Coeff."),
        ("mean_degree", "Mean Degree"),
        ("fraud_frac",  "Fraud Neighbour %"),
    ]
    colors = ["#EF4444", "#3B82F6"]
    for ax, (key, title) in zip(axes, metrics):
        vals = [fraud_stats.get(key, 0), licit_stats.get(key, 0)]
        bars = ax.bar(["Fraud", "Licit"], vals, color=colors, alpha=0.85,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/community_stats_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: results/additions/fraud_subgraph_viz.png")
    print("Saved: results/additions/community_stats_comparison.png")
    print("Saved: results/additions/community_stats.csv")


if __name__ == "__main__":
    main()