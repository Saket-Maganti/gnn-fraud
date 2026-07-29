"""
models/mlp_baseline.py
MLP baseline — uses only node features, no graph structure.
Purpose: prove that GNN's graph structure contributes meaningfully.
Expected F1: 0.55-0.62 (vs GNN 0.72-0.74, ~10-15pp gap proves graph helps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    3-layer MLP. Identical interface to GNN models:
        forward(x, edge_index, edge_attr) → logits [N, 3]
    edge_index and edge_attr are accepted but ignored —
    this is intentional to show the graph structure baseline.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 3, num_layers: int = 3,
                 dropout: float = 0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index=None, edge_attr=None):
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
