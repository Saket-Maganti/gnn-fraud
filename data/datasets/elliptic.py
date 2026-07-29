"""
data/datasets/elliptic.py

Wraps the existing Elliptic loader so it returns a :class:`FraudDataset`
conforming to the unified multi-dataset schema.

The raw loader (``data.dataset.preprocess``) already produces a PyG ``Data``
with the right tensors; we re-emit it as ``FraudDataset`` and carve a small
validation window off the final training timesteps so every downstream
script has a non-empty ``val_mask`` to use for early stopping and
threshold / prior calibration.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import load_elliptic_raw, preprocess  # noqa: E402
from data.datasets.base import FraudDataset, SplitBoundaries  # noqa: E402
from data.feature_scaling import resolve_scaler_mode, scale_features  # noqa: E402


# Published Elliptic split. We shave off the last few training timesteps as a
# validation window so TPC/TTA has a realistic, held-out prior estimate and
# early stopping can't peek at test labels.
_DEFAULT_SPLIT = SplitBoundaries(train_max=30, val_max=34, test_max=49)


def load_elliptic_dataset(
    split: Optional[SplitBoundaries] = None,
    use_edge_features: bool = False,
    scaler_mode: Optional[str] = "train_only",
    normalize: Optional[bool] = None,
) -> FraudDataset:
    split = split or _DEFAULT_SPLIT
    mode = resolve_scaler_mode(
        scaler_mode,
        normalize=normalize,
        default="train_only",
    )
    features, classes, edges = load_elliptic_raw()
    data = preprocess(
        features,
        classes,
        edges,
        scaler_mode="none",
        use_edge_features=use_edge_features,
    )

    labeled    = data.y != 0
    t          = data.time_step
    train_mask = labeled & (t <= split.train_max)
    val_mask   = labeled & (t > split.train_max) & (t <= split.val_max)
    test_mask  = labeled & (t > split.val_max)   & (t <= split.test_max)

    x_np = data.x.numpy()
    x_scaled, mode = scale_features(x_np, train_mask.numpy(), mode)

    meta = {
        "source":       "elliptic_bitcoin",
        "url":          "https://www.kaggle.com/ellipticco/elliptic-data-set",
        "n_timesteps":  int(t.max().item()),
        "split":        split.describe(),
        "scaler_mode":  mode,
    }
    return FraudDataset.build(
        x          = torch.from_numpy(x_scaled),
        y          = data.y,
        edge_index = data.edge_index,
        edge_attr  = getattr(data, "edge_attr", None),
        time_step  = t,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        name       = "elliptic",
        meta       = meta,
    )
