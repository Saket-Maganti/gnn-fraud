"""Binary predictive losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float | None = None,
) -> torch.Tensor:
    target = targets.float()
    pos_weight = (
        logits.new_tensor(positive_weight) if positive_weight is not None else None
    )
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    if gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("focal gamma must be non-negative and alpha in [0,1]")
    targets_float = targets.float()
    base = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets_float + (1 - probabilities) * (1 - targets_float)
    alpha_t = alpha * targets_float + (1 - alpha) * (1 - targets_float)
    return (alpha_t * (1 - p_t).pow(gamma) * base).mean()


def class_balanced_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float = 0.999,
) -> torch.Tensor:
    if not 0 <= beta < 1:
        raise ValueError("class-balanced beta must be in [0,1)")
    target = targets.long()
    counts = torch.stack([(target == 0).sum(), (target == 1).sum()]).float()
    effective = 1 - beta ** counts.clamp_min(1)
    weights = (1 - beta) / effective
    weights = weights / weights.sum() * 2
    sample_weight = weights[target]
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return (sample_weight * loss).mean()
