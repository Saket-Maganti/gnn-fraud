"""
models/registry.py

Unified model factory covering the legacy ``models.gnn`` registry *and* the
new modern / temporal / solution modules. Every experiment script should
call ``build_model`` from this module so that adding an architecture only
requires updating this one place.

Entries grouped by category:

    Legacy GNN (models/gnn.py)
        gcn, sage/graphsage, sage_deep, sage_maxpool, gat, edgegat, evolvegcn

    Modern GNN (models/modern_gnn.py)
        graph_transformer, gps, pcgnn, bwgnn

    Temporal GNN (models/temporal_gnn.py)
        snapshot_tgn

The category tag is returned alongside the instance so the training harness
knows whether it needs to call ``set_time_step`` (temporal models), feed in
edge attributes, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

from models.gnn import (
    GCN, GraphSAGE, GraphSAGE_Deep, GraphSAGE_MaxPool,
    GAT, EdgeGAT, EvolveGCN,
    count_parameters as _count_parameters,
)
from models.modern_gnn import GraphTransformer, GPS, PCGNN, BWGNN, GIN
from models.temporal_gnn import SnapshotTGN


LEGACY_MODELS = {
    "gcn":          GCN,
    "sage":         GraphSAGE,
    "graphsage":    GraphSAGE,
    "sage_deep":    GraphSAGE_Deep,
    "sage_maxpool": GraphSAGE_MaxPool,
    "gat":          GAT,
    "edgegat":      EdgeGAT,
    "evolvegcn":    EvolveGCN,
}

# ─────────────────────────────────────────────────────────────────────────────
# Optional, memory-safe LIGHT variants of the OOM-prone models.
#
# These are factory callables (not bare classes) so they FORCE small, T4-safe
# configs regardless of the harness-supplied hidden/layers/dropout. They are
# deliberately NOT drop-in equivalents of the full models and are never part of
# the default stable run — see scripts/runs_harness_common.OPTIONAL_LIGHT_VARIANTS
# and scripts/run_optional_memory_risky_models.py.
#
#   gps_local          : GPS with global attention DISABLED (local MPNN + FFN
#                        only). The full O(N^2) attention is ~664 GB on
#                        Elliptic, so the only T4-feasible variant drops it. NOT
#                        equivalent to full GPS.
#   gps_light          : same (global attention disabled) with a slightly larger
#                        local backbone.
#   snapshot_tgn_light : full SnapshotTGN architecture at a much smaller config
#                        (hidden 32 / 1 layer / 1 head) to cut the across-snapshot
#                        autograd memory that OOMs the default (hidden 128 / 2 / 4).
# ─────────────────────────────────────────────────────────────────────────────

def _gps_local(in_channels: int, out_channels: int = 3, **_ignored) -> nn.Module:
    """Local-only GPS (no global attention). T4-feasible; NOT full GPS."""
    return GPS(in_channels=in_channels, hidden_channels=32, out_channels=out_channels,
               num_layers=1, heads=1, dropout=0.3, use_global_attention=False)


def _gps_light(in_channels: int, out_channels: int = 3, **_ignored) -> nn.Module:
    """Slightly larger local-only GPS (no global attention). NOT full GPS."""
    return GPS(in_channels=in_channels, hidden_channels=64, out_channels=out_channels,
               num_layers=2, heads=2, dropout=0.3, use_global_attention=False)


def _snapshot_tgn_light(in_channels: int, out_channels: int = 3, **_ignored) -> nn.Module:
    """Memory-light SnapshotTGN: hidden 32 / 1 layer / 1 head (vs 128 / 2 / 4)."""
    return SnapshotTGN(in_channels=in_channels, hidden_channels=32, out_channels=out_channels,
                       num_layers=1, heads=1, dropout=0.3, time_dim=16)


MODERN_MODELS = {
    "graph_transformer": GraphTransformer,
    "transformer":       GraphTransformer,  # alias
    "gps":               GPS,
    "gin":               GIN,
    "gps_local":         _gps_local,         # optional light (global attn OFF)
    "gps_light":         _gps_light,         # optional light (global attn OFF)
    "pcgnn":             PCGNN,
    "bwgnn":             BWGNN,
    "beta_wavelet":      BWGNN,              # alias
}

TEMPORAL_MODELS = {
    "snapshot_tgn":       SnapshotTGN,
    "tgn":                SnapshotTGN,        # alias
    "snapshot_tgn_light": _snapshot_tgn_light,  # optional light config
}


def _category(name: str) -> str:
    if name in TEMPORAL_MODELS: return "temporal"
    if name in MODERN_MODELS:   return "modern"
    if name in LEGACY_MODELS:   return "legacy"
    raise ValueError(f"Unknown model '{name}'")


def all_model_names() -> list:
    return sorted(set(LEGACY_MODELS) | set(MODERN_MODELS) | set(TEMPORAL_MODELS))


@dataclass
class BuiltModel:
    model:    nn.Module
    category: str      # "legacy" | "modern" | "temporal"
    name:     str


def build_model(name: str, in_channels: int, **kwargs) -> BuiltModel:
    """Instantiate a model by registry key and return it with metadata.

    The training harness branches on ``category`` — temporal models get a
    ``set_time_step`` call before each forward pass; edge-feature models
    receive ``edge_attr``; everything else uses the plain
    ``(x, edge_index)`` signature.
    """
    name = name.lower()
    for table in (LEGACY_MODELS, MODERN_MODELS, TEMPORAL_MODELS):
        if name in table:
            model = table[name](in_channels=in_channels, **kwargs)
            return BuiltModel(model=model, category=_category(name), name=name)
    raise ValueError(
        f"Unknown model '{name}'. Available: {all_model_names()}"
    )


def count_parameters(model: nn.Module) -> int:
    return _count_parameters(model)
