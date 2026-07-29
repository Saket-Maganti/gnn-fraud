"""Validated Group DRO, VREx, and IRM objective adapters."""

from __future__ import annotations

import torch


def group_dro_loss(
    group_losses: torch.Tensor,
    group_logits: torch.Tensor,
    *,
    step_size: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponentiated-gradient Group DRO update over source contracts."""

    if group_losses.ndim != 1 or group_logits.shape != group_losses.shape:
        raise ValueError("one Group DRO loss and log-weight per source group required")
    updated = group_logits + step_size * group_losses.detach()
    weights = torch.softmax(updated, dim=0)
    return (weights * group_losses).sum(), updated


def vrex_loss(group_losses: torch.Tensor, *, penalty_weight: float = 1.0) -> torch.Tensor:
    if group_losses.numel() < 2:
        return group_losses.mean()
    return group_losses.mean() + penalty_weight * group_losses.var(unbiased=False)


def irm_penalty(losses: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """IRMv1 gradient penalty with an explicit scalar classifier scale."""

    gradient = torch.autograd.grad(
        losses.mean(),
        [scale],
        create_graph=True,
    )[0]
    return gradient.pow(2).sum()
