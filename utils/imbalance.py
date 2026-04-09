import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from typing import Tuple


class WeightedCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, warmup_epochs: int = 20):
        super().__init__()
        self.register_buffer("weight", class_weights)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def step_epoch(self):
        self.current_epoch += 1

    def forward(self, logits, targets, mask):
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            w_ones = torch.ones_like(self.weight)
            w_ones[0] = 0.0
            w = w_ones + alpha * (self.weight - w_ones)
        else:
            w = self.weight
        return F.cross_entropy(logits[mask], targets[mask], weight=w)


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", class_weights)
        self.gamma = gamma

    def forward(self, logits, targets, mask):
        ce   = F.cross_entropy(logits[mask], targets[mask],
                               weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class MaskedCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask):
        return F.cross_entropy(logits[mask], targets[mask])


def oversample_minority(data, factor=3.0, noise_std=0.01):
    device = data.x.device
    idx   = torch.where(data.train_mask & (data.y == 1))[0]
    n_add = int(len(idx) * (factor - 1))
    if n_add == 0:
        return data
    samp    = idx[torch.randint(len(idx), (n_add,))]
    extra_x = data.x[samp] + torch.randn_like(data.x[samp]) * noise_std
    from torch_geometric.data import Data as D
    aug = D(
        x          = torch.cat([data.x, extra_x]),
        y          = torch.cat([data.y, data.y[samp]]),
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
    )
    aug.train_mask = torch.cat([
        data.train_mask,
        torch.ones(n_add, dtype=torch.bool, device=device),
    ])
    aug.test_mask  = torch.cat([
        data.test_mask,
        torch.zeros(n_add, dtype=torch.bool, device=device),
    ])
    aug.time_step  = torch.cat([data.time_step,  data.time_step[samp]])
    return aug


def graph_augmentation(data, n_subgraphs=30, n_hops=1, noise_std=0.02):
    device = data.x.device
    seeds = torch.where(data.train_mask & (data.y == 1))[0]
    if len(seeds) == 0:
        return data
    all_x, all_y, all_edges = [], [], []
    offset = data.num_nodes
    chosen = seeds[torch.randint(len(seeds), (n_subgraphs,))]
    for seed in chosen:
        sub_nodes, sub_ei, _, _ = k_hop_subgraph(
            seed.item(), n_hops, data.edge_index,
            relabel_nodes=True, num_nodes=data.num_nodes)
        n_sub = sub_nodes.size(0)
        noise = torch.randn(n_sub, data.x.size(1), device=device) * noise_std
        all_x.append(data.x[sub_nodes] + noise)
        all_y.append(torch.ones(n_sub, dtype=torch.long, device=device))
        all_edges.append(sub_ei + offset)
        offset += n_sub
    nx = torch.cat(all_x); ny = torch.cat(all_y); ne = torch.cat(all_edges, dim=1)
    n_new = nx.size(0)
    from torch_geometric.data import Data as D
    aug = D(
        x          = torch.cat([data.x, nx]),
        y          = torch.cat([data.y, ny]),
        edge_index = torch.cat([data.edge_index, ne], dim=1),
        edge_attr  = data.edge_attr,
    )
    aug.train_mask = torch.cat([
        data.train_mask,
        torch.ones(n_new, dtype=torch.bool, device=device),
    ])
    aug.test_mask  = torch.cat([
        data.test_mask,
        torch.zeros(n_new, dtype=torch.bool, device=device),
    ])
    aug.time_step  = torch.cat([
        data.time_step,
        torch.zeros(n_new, dtype=torch.long, device=device),
    ])
    return aug


class DriftDetector:
    def __init__(self, window=5, threshold=0.04):
        self.window = window; self.threshold = threshold
        self.history = []; self.drift_at = []

    def update(self, f1):
        self.history.append(f1)
        if len(self.history) < self.window:
            return False
        drift = np.mean(self.history[-self.window:-1]) - f1
        if drift > self.threshold:
            self.drift_at.append(len(self.history)); return True
        return False


def apply_strategy(data, strategy, class_weights):
    s = strategy.lower()
    if s == "baseline":
        return data, MaskedCELoss()
    elif s == "weighted":
        return data, WeightedCELoss(class_weights)
    elif s == "focal":
        return data, FocalLoss(class_weights)
    elif s == "oversample":
        return oversample_minority(data), MaskedCELoss()
    elif s in ("graph_aug", "adaptive"):
        return graph_augmentation(data), WeightedCELoss(class_weights)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
