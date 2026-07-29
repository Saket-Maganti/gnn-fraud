"""
models/gnn.py
All GNN architectures for Levels 1-4.

Level 1 : GCN           — basic 2-layer GCN (baseline)
Level 2 : GraphSAGE     — mean aggregation, inductive
Level 2 : GAT           — multi-head attention
Level 3 : EdgeGAT       — GAT + edge features
Level 4 : EvolveGCN_O   — GRU-evolved weights for temporal graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, BatchNorm, NNConv
)
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Shared MLP block
# ─────────────────────────────────────────────────────────────────────────────

def mlp(dims: List[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — GCN baseline
# ─────────────────────────────────────────────────────────────────────────────

class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network.
    Kipf & Welling (2017). Pure baseline — no BatchNorm, no input proj.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 3, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — GraphSAGE
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGE(nn.Module):
    """
    GraphSAGE with mean aggregation.
    Hamilton et al. (2017).
    Architecture: Linear(in→h) → [SAGEConv+BN+ReLU+Drop] × L → Linear(h→out)
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 3, num_layers: int = 3,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for c in self.convs: c.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — GAT
# ─────────────────────────────────────────────────────────────────────────────

class GAT(nn.Module):
    """
    Graph Attention Network, multi-head.
    Veličković et al. (2018).
    Architecture: Linear(in→h) → [GATConv+BN+ReLU+Drop] × L → Linear(h→out)
    heads=4, hidden must be divisible by heads.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 3, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.5,
                 edge_dropout: float = 0.2, **kwargs):
        super().__init__()
        self.dropout      = dropout
        assert hidden_channels % heads == 0, "hidden must be divisible by heads"
        head_dim          = hidden_channels // heads
        self.input_proj   = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, head_dim, heads=heads,
                    dropout=edge_dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for c in self.convs: c.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — EdgeGAT (GAT + edge features via NNConv)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeGAT(nn.Module):
    """
    GAT with edge feature integration.
    Uses NNConv to incorporate edge attributes (6-dim) into message passing.
    First layer: NNConv(edge features → message weights)
    Remaining layers: GATConv (standard)
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 3, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.5,
                 edge_dim: int = 6, **kwargs):
        super().__init__()
        self.dropout    = dropout
        head_dim        = hidden_channels // heads
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # First layer uses edge features
        edge_nn         = mlp([edge_dim, 64, hidden_channels * hidden_channels])
        self.edge_conv  = NNConv(hidden_channels, hidden_channels, edge_nn, aggr="mean")
        self.bn0        = BatchNorm(hidden_channels)

        # Remaining layers: standard GAT
        self.convs = nn.ModuleList([
            GATConv(hidden_channels, head_dim, heads=heads,
                    dropout=0.2, concat=True)
            for _ in range(num_layers - 1)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers - 1)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Edge-feature-aware first layer
        if edge_attr is not None:
            x = self.edge_conv(x, edge_index, edge_attr)
        else:
            x = self.edge_conv(x, edge_index,
                               torch.zeros(edge_index.size(1), 6, device=x.device))
        x = self.bn0(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.edge_conv.reset_parameters()
        for c in self.convs: c.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Level 4 — EvolveGCN-O (temporal GNN)
# ─────────────────────────────────────────────────────────────────────────────

class EvolveGCN(nn.Module):
    """
    Lightweight EvolveGCN-style temporal GNN.

    The original implementation attempted to evolve a full hidden x hidden
    weight matrix with a GRU cell, which scales quadratically in the hidden
    size and becomes impractical for hidden=128. This version uses graph
    convolutions for per-snapshot node updates and a compact GRU over a
    global graph state across snapshots.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 3, num_layers: int = 2,
                 dropout: float = 0.4, **kwargs):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, snapshots: List[tuple]) -> torch.Tensor:
        """
        Args:
            snapshots: list of (x, edge_index) tuples, sorted ascending by time.
                       x shape: [N, in_channels]
        Returns:
            logits: [N, out_channels] for the LAST snapshot
        """
        h_gru = None
        x_out = None
        for x, edge_index in snapshots:
            h = F.relu(self.input_proj(x))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for conv, bn in zip(self.convs, self.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            h_mean = h.mean(dim=0, keepdim=True)
            if h_gru is None:
                h_gru = h_mean
            h_gru = self.gru(h_mean, h_gru)

            h = h + h_gru.expand_as(h)
            x_out = h

        return self.classifier(x_out)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.gru.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Stronger SAGE variants (for ablation / deeper architectures)
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGE_Deep(nn.Module):
    """
    Deeper GraphSAGE (5 layers, mean aggregation).
    Tests whether additional depth helps on the Elliptic dataset.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 3, dropout: float = 0.5, **kwargs):
        super().__init__()
        self.dropout    = dropout
        num_layers      = 5
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x + residual)   # residual connection to mitigate over-smoothing
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for c in self.convs: c.reset_parameters()
        self.classifier.reset_parameters()


class GraphSAGE_MaxPool(nn.Module):
    """
    GraphSAGE with max-pool aggregation (vs. mean in the base model).
    Tests whether max-pool captures more discriminative neighbourhood signals.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 out_channels: int = 3, num_layers: int = 3,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([
            SAGEConv(hidden_channels, hidden_channels, aggr="max")
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            BatchNorm(hidden_channels) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for c in self.convs: c.reset_parameters()
        self.classifier.reset_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gcn":            GCN,
    "sage":           GraphSAGE,
    "graphsage":      GraphSAGE,
    "sage_deep":      GraphSAGE_Deep,
    "sage_maxpool":   GraphSAGE_MaxPool,
    "gat":            GAT,
    "edgegat":        EdgeGAT,
    "evolvegcn":      EvolveGCN,
}

def build_model(name: str, in_channels: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](in_channels=in_channels, **kwargs)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
