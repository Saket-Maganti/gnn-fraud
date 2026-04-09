"""
config.py
Central experiment configuration.  Pass a Config instance to any runner.

Graph types
-----------
  original     : Raw Elliptic transaction edges (existing graph)
  similarity   : Within-timestep cosine-similarity edges (threshold-gated)
  feature_knn  : Within-timestep k-NN in feature space
  temporal     : Cross-timestep k-NN edges linking adjacent time slices
  augmented    : Original edges PLUS similarity edges (additive)
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Graph construction ─────────────────────────────────────────────────
    graph_type: str    = "original"   # original | similarity | feature_knn | temporal | augmented
    sim_threshold: float = 0.92       # cosine similarity cutoff
    knn_k: int         = 5            # neighbours in knn / temporal graphs
    temporal_window: int = 1          # how many adjacent timesteps to bridge

    # ── Model ──────────────────────────────────────────────────────────────
    model: str         = "sage"       # gcn | sage | gat
    strategy: str      = "weighted"   # baseline | weighted | oversample | graph_aug
    hidden_channels: int = 256
    num_layers: int    = 3
    dropout: float     = 0.5

    # ── Training ───────────────────────────────────────────────────────────
    epochs: int        = 300
    lr: float          = 1e-3
    weight_decay: float= 5e-4
    patience: int      = 30
    seed: int          = 42

    # ── Hybrid ─────────────────────────────────────────────────────────────
    hybrid_clf: str    = "xgboost"    # xgboost | rf
    embed_concat_raw: bool = False    # whether to concat raw features with embeddings

    # ── Paths ──────────────────────────────────────────────────────────────
    results_dir: str   = "results"
    checkpoint_dir: str= "results/checkpoints"

    # ── Device ─────────────────────────────────────────────────────────────
    device: str        = "cpu"        # cpu | cuda | mps


# Convenience presets
GRAPH_TYPES = ["original", "similarity", "feature_knn", "temporal", "augmented"]
BASELINE_MODELS = ["lr", "rf", "xgboost"]
GNN_MODELS = ["gcn", "sage", "gat"]
