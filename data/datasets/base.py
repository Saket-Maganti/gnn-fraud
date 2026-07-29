"""
data/datasets/base.py

Unified schema for temporal fraud-detection datasets.

``FraudDataset`` subclasses :class:`torch_geometric.data.Data` so every
downstream experiment (training loops, ablations, calibration scripts)
continues to work unchanged on the multi-dataset path — we just guarantee
that a fixed set of attributes is present, and that semantics match across
sources.

Design choices
--------------
* Labels are stored as ``long`` with ``0 = unknown/unlabeled``. The positive
  (fraud) class is always ``1``. Datasets with different raw encodings are
  remapped inside their loader. Keeping ``0 = unknown`` matches Elliptic and
  lets us reuse the existing label-masking logic throughout the repo.
* ``time_step`` is an integer tensor with the per-node time bucket. Datasets
  without node-level timestamps (e.g. purely transactional logs) expose an
  edge-derived bucketisation so the inductive / temporal evaluation harness
  can still apply.
* ``train_mask`` / ``val_mask`` / ``test_mask`` are all bool tensors
  restricted to labeled nodes. They are disjoint by construction.

The ``meta`` dict carries dataset-specific bookkeeping (split boundaries,
class counts, source URL). It's serialised with the results so later
analysis scripts know exactly which split each experiment used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch_geometric.data import Data


@dataclass
class SplitBoundaries:
    """Time-step boundaries for the chronological split.

    ``train_max``       : last time bucket used for training.
    ``val_max``         : last time bucket used for validation (> train_max).
    ``test_max``        : last time bucket used for test evaluation.

    Semantics: a node with ``time_step <= train_max`` goes to train, a node
    with ``train_max < time_step <= val_max`` goes to val, and anything in
    ``(val_max, test_max]`` goes to test. A dataset that does not need a
    validation split sets ``val_max == train_max``.
    """
    train_max: int
    val_max: int
    test_max: int

    def describe(self) -> str:
        return (f"train<=t{self.train_max} | "
                f"val t{self.train_max+1}..t{self.val_max} | "
                f"test t{self.val_max+1}..t{self.test_max}")


class FraudDataset(Data):
    """Thin PyG ``Data`` subclass with a stable schema across datasets."""

    @classmethod
    def build(
        cls,
        x:          torch.Tensor,
        y:          torch.Tensor,
        edge_index: torch.Tensor,
        time_step:  torch.Tensor,
        train_mask: torch.Tensor,
        val_mask:   torch.Tensor,
        test_mask:  torch.Tensor,
        name:       str,
        meta:       Optional[Dict] = None,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> "FraudDataset":
        obj = cls(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        obj.time_step  = time_step.long()
        obj.train_mask = train_mask.bool()
        obj.val_mask   = val_mask.bool()
        obj.test_mask  = test_mask.bool()
        obj.name       = name
        obj.meta       = meta or {}
        obj.pos_label  = 1
        _sanity_check(obj)
        return obj

    # Convenience accessors used by downstream scripts
    def labeled_mask(self) -> torch.Tensor:
        return self.y != 0

    def pos_mask(self) -> torch.Tensor:
        return self.y == self.pos_label


def _sanity_check(obj: FraudDataset) -> None:
    """Cheap invariants that catch loader bugs before training starts."""
    n = obj.num_nodes
    assert obj.time_step.numel() == n, "time_step length mismatch"
    for name in ("train_mask", "val_mask", "test_mask"):
        m = getattr(obj, name)
        assert m.dtype == torch.bool, f"{name} must be bool"
        assert m.numel() == n, f"{name} length mismatch"
    overlap_tv = (obj.train_mask & obj.val_mask).any().item()
    overlap_tt = (obj.train_mask & obj.test_mask).any().item()
    overlap_vt = (obj.val_mask   & obj.test_mask).any().item()
    assert not (overlap_tv or overlap_tt or overlap_vt), (
        "train/val/test masks must be disjoint"
    )
    assert obj.y.min().item() >= 0, "labels must be non-negative"


def describe_dataset(data: FraudDataset) -> str:
    """One-screen summary used by every experiment runner."""
    y = data.y
    pos = (y == data.pos_label).sum().item()
    neg = ((y != data.pos_label) & (y != 0)).sum().item()
    lines = [
        "=" * 58,
        f" DATASET  {data.name}",
        "=" * 58,
        f"  Nodes         : {data.num_nodes:,}",
        f"  Edges         : {data.num_edges:,}",
        f"  Features      : {data.num_node_features}",
        f"  Time buckets  : {data.time_step.min().item()} .. "
            f"{data.time_step.max().item()}",
        f"  Positives (y==1) : {pos:,}  "
            f"({100 * pos / max(pos + neg, 1):.2f}% of labeled)",
        f"  Negatives        : {neg:,}",
        f"  Train / Val / Test nodes : "
            f"{data.train_mask.sum().item():,} / "
            f"{data.val_mask.sum().item():,} / "
            f"{data.test_mask.sum().item():,}",
    ]
    if data.meta:
        split = data.meta.get("split")
        if split is not None:
            lines.append(f"  Split boundaries : {split}")
        scaler_mode = data.meta.get("scaler_mode")
        if scaler_mode is not None:
            lines.append(f"  Scaler mode      : {scaler_mode}")
    lines.append("=" * 58)
    return "\n".join(lines)


def get_class_weights(data: FraudDataset) -> torch.Tensor:
    """Square-root-scaled, capped class weights over all observed classes.

    Mirrors the recipe the existing Elliptic experiments use, extended to an
    arbitrary number of classes. Index 0 is reserved for unknown; its weight
    is 0 so it never contributes to the loss.
    """
    import math
    n_classes = int(data.y.max().item()) + 1
    n_classes = max(n_classes, 3)  # preserve the 3-class Elliptic layout
    w = torch.zeros(n_classes, dtype=torch.float)
    train_y = data.y[data.train_mask]
    total   = int(train_y.numel())
    for c in range(1, n_classes):
        count = int((train_y == c).sum().item())
        if count == 0:
            w[c] = 1.0
            continue
        raw = total / (2 * count)
        w[c] = math.sqrt(raw)
    return w
