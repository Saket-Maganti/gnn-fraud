"""
data/datasets/ — multi-dataset abstraction.

Each dataset loader returns a unified ``FraudDataset`` (see ``base.py``) that
exposes the same attributes regardless of source:

    x           [N, F]   node features (float32)
    y           [N]      class labels (long; 0 = unknown / unlabeled)
    edge_index  [2, E]   undirected edge index
    time_step   [N]      integer time bucket per node (1..T)
    train_mask  [N]      bool; labeled nodes in the train window
    val_mask    [N]      bool; labeled nodes in the validation window
    test_mask   [N]      bool; labeled nodes in the test window
    pos_label   int      which ``y`` value counts as fraud (always 1)
    name        str      dataset identifier

The temporal split is a per-dataset choice made by the loader. The standard
contract is: train on earlier time buckets, hold out a small val slice
adjacent to the test window, evaluate on the latest time buckets.

Registry entries:
    elliptic       — Elliptic Bitcoin (existing; 49 timesteps, 9.8% illicit)
    ellipticpp     — Elliptic++ actors/wallets graph (arXiv:2306.06108; 49
                     timesteps). Bitcoin-derived like Elliptic → used as
                     TEMPORAL-ROBUSTNESS corroboration (correlated domain), not
                     domain generalization. dgraphfin stays the independent one.
    dgraphfin      — DGraphFin (ICLR'22; 3M nodes, financial fraud)
    tfinance       — T-Finance (Tsinghua; smaller, labelled temporal fraud)
"""

from .base import FraudDataset
from .elliptic import load_elliptic_dataset
from .ellipticpp import load_ellipticpp_dataset
from .dgraphfin import load_dgraphfin_dataset
from .tfinance import load_tfinance_dataset


DATASET_REGISTRY = {
    "elliptic":   load_elliptic_dataset,
    "ellipticpp": load_ellipticpp_dataset,
    "dgraphfin":  load_dgraphfin_dataset,
    "tfinance":   load_tfinance_dataset,
}


def load_dataset(name: str, **kwargs) -> FraudDataset:
    """Entry point used by every experiment script.

    Each loader is responsible for raising a clear error (with download
    instructions) if the raw files are missing; we never silently fall back
    to a synthetic surrogate.
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](**kwargs)


__all__ = [
    "FraudDataset",
    "DATASET_REGISTRY",
    "load_dataset",
]
