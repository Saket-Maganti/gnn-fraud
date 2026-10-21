"""Stable empirical CVaR over contract groups."""

from __future__ import annotations

import torch


def empirical_cvar(losses: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    """Mean of the worst ``1-alpha`` empirical contract losses."""

    if not 0 <= alpha < 1:
        raise ValueError("CVaR alpha must lie in [0,1)")
    flat = losses.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("CVaR losses cannot be empty")
    tail_count = max(1, int(torch.ceil(flat.new_tensor((1 - alpha) * flat.numel())).item()))
    return torch.topk(flat, k=tail_count, largest=True, sorted=False).values.mean()


def variational_cvar(
    losses: torch.Tensor,
    eta: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """Rockafellar–Uryasev differentiable CVaR objective."""

    if not 0 <= alpha < 1:
        raise ValueError("CVaR alpha must lie in [0,1)")
    return eta + torch.relu(losses - eta).mean() / max(1 - alpha, 1e-8)
