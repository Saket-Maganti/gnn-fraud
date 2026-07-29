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
    # ── Dataset ────────────────────────────────────────────────────────────
    dataset: str       = "elliptic"   # elliptic | dgraphfin | tfinance
    scaler_mode: str   = "train_only" # train_only | full_population | none

    # ── Graph construction ─────────────────────────────────────────────────
    graph_type: str    = "original"   # original | similarity | feature_knn | temporal | augmented
    sim_threshold: float = 0.92       # cosine similarity cutoff
    knn_k: int         = 5            # neighbours in knn / temporal graphs
    temporal_window: int = 1          # how many adjacent timesteps to bridge

    # ── Model ──────────────────────────────────────────────────────────────
    # legacy:        gcn | sage | gat | edgegat | evolvegcn | sage_deep | sage_maxpool
    # modern:        graph_transformer | gps | pcgnn
    # temporal:      snapshot_tgn
    model: str         = "sage"
    strategy: str      = "weighted"   # baseline | weighted | oversample | graph_aug
    hidden_channels: int = 256
    num_layers: int    = 3
    dropout: float     = 0.5

    # ── TPC + TTA (solution method) ────────────────────────────────────────
    tpc_enabled: bool  = False        # wrap inference with TPC+TTA
    tpc_window:  int   = 3            # rolling prior window (time buckets)

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
DATASETS        = ["elliptic", "dgraphfin", "tfinance"]
SCALER_MODES    = ["train_only", "full_population", "none"]
GRAPH_TYPES     = ["original", "similarity", "feature_knn", "temporal", "augmented"]
BASELINE_MODELS = ["lr", "rf", "xgboost"]
GNN_MODELS      = ["gcn", "sage", "gat"]
MODERN_GNN_MODELS   = ["graph_transformer", "gps", "pcgnn"]
TEMPORAL_GNN_MODELS = ["snapshot_tgn", "evolvegcn"]
