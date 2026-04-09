"""
utils/temporal.py
Temporal analysis utilities for Levels 3 and 4.

Level 3: per-timestep F1 evaluation (the key publishable finding)
Level 4: snapshot builder for EvolveGCN
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from typing import Dict, List, Tuple, Optional
from utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder (L4 — EvolveGCN)
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshots(
    data:       Data,
    time_range: range,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a list of (x, edge_index) snapshots for each time step.

    For EvolveGCN: the model receives snapshots in temporal order and
    evolves its GCN weights across them via GRU.

    Each snapshot contains ALL nodes but only edges where BOTH endpoints
    exist in that time step's window (approximation — Elliptic edges don't
    have explicit timestamps, so we use the source node's time step).

    Args:
        data: full PyG Data object
        time_range: e.g. range(1, 35) for training, range(35, 50) for test

    Returns:
        List of (node_features, edge_index) per time step
    """
    snapshots = []
    for t in time_range:
        node_mask  = (data.time_step == t)
        node_idx   = torch.where(node_mask)[0]

        if node_idx.numel() == 0:
            continue

        # Subgraph induced by nodes in this timestep
        sub_ei, _ = subgraph(
            node_idx, data.edge_index,
            relabel_nodes=False,
            num_nodes=data.num_nodes,
        )

        snapshots.append((data.x, sub_ei))

    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Per-timestep evaluation (L3 — temporal drift analysis)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_per_timestep(
    model:       torch.nn.Module,
    data:        Data,
    device:      torch.device,
    model_type:  str = "static",
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model on each test time step separately (steps 35-49).

    This is the KEY analysis for the paper:
    - Shows whether model degrades on later time steps (temporal drift)
    - Graph augmentation typically degrades after step ~42 because
      synthetic subgraphs don't reflect evolving fraud patterns

    Args:
        model_type: 'static' for GCN/SAGE/GAT, 'evolve' for EvolveGCN

    Returns:
        { timestep: { f1, precision, recall, accuracy, auc } }
    """
    model.eval()
    data = data.to(device)
    results = {}

    if model_type == "evolve":
        # Feed all training snapshots first, then evaluate per test step
        train_snaps = build_snapshots(data, time_range=range(1, 35))
        snaps_dev   = [(x.to(device), ei.to(device)) for x, ei in train_snaps]
        logits      = model(snaps_dev)
    else:
        logits = model(data.x, data.edge_index,
                       getattr(data, "edge_attr", None))

    for t in range(35, 50):
        mask = data.test_mask & (data.time_step == t)
        mask = mask.to(device)
        if mask.sum().item() < 5:
            continue
        metrics = compute_metrics(logits, data.y, mask)
        results[t] = metrics

    return results


def summarise_temporal(results: Dict[int, Dict]) -> str:
    """Print a temporal F1 table."""
    lines = [f"{'Step':>5} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}"]
    lines.append("-" * 42)
    for t in sorted(results):
        m = results[t]
        lines.append(
            f"{t:>5} {m['f1']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['auc']:>8.4f}"
        )
    f1s = [results[t]["f1"] for t in sorted(results)]
    lines.append("-" * 42)
    lines.append(f"{'Mean':>5} {np.mean(f1s):>8.4f}")
    lines.append(f"{'Trend':>5} {'↓ drift' if f1s[-1] < f1s[0] - 0.05 else '→ stable':>8}")
    return "\n".join(lines)
