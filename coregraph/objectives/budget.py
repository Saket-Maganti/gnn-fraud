"""Differentiable review-budget surrogates."""

from __future__ import annotations

import torch


def soft_topk_weights(
    scores: torch.Tensor,
    k: int,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return soft review weights summing approximately to K.

    A bisection threshold solves sum(sigmoid((score-tau)/T)) = K. This is a
    surrogate, not exact top-K optimisation.
    """

    if k < 0 or k > scores.numel():
        raise ValueError("k must lie between 0 and number of scores")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if k == 0:
        return torch.zeros_like(scores)
    if k == scores.numel():
        return torch.ones_like(scores)
    low = scores.min().detach() - 20 * temperature
    high = scores.max().detach() + 20 * temperature
    for _ in range(48):
        middle = (low + high) / 2
        mass = torch.sigmoid((scores - middle) / temperature).sum().detach()
        if mass > k:
            low = middle
        else:
            high = middle
    threshold = (low + high) / 2
    return torch.sigmoid((scores - threshold) / temperature)


def soft_recall_at_k_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    weights = soft_topk_weights(scores, k, temperature=temperature)
    positives = targets.float().sum().clamp_min(1)
    soft_recall = (weights * targets.float()).sum() / positives
    return 1 - soft_recall


def soft_precision_at_k_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    weights = soft_topk_weights(scores, k, temperature=temperature)
    soft_precision = (weights * targets.float()).sum() / weights.sum().clamp_min(1e-8)
    return 1 - soft_precision
