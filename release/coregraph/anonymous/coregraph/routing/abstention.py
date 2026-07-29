"""Selective-prediction helpers."""

from __future__ import annotations

import torch


def apply_abstention_capacity(
    abstention_probability: torch.Tensor,
    capacity: float,
) -> torch.Tensor:
    if not 0 <= capacity <= 1:
        raise ValueError("abstention capacity must lie in [0,1]")
    n = abstention_probability.numel()
    k = min(n, max(0, int(round(capacity * n))))
    decision = torch.zeros_like(abstention_probability, dtype=torch.bool)
    if k:
        indices = torch.argsort(abstention_probability, descending=True, stable=True)[:k]
        decision[indices] = True
    return decision


def selective_risk(
    losses: torch.Tensor,
    abstain: torch.Tensor,
) -> torch.Tensor:
    accepted = ~abstain.bool()
    if not accepted.any():
        return losses.new_tensor(float("nan"))
    return losses[accepted].mean()
