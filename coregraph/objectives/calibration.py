"""Differentiable calibration penalty."""

from __future__ import annotations

import torch


def soft_brier_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return ((torch.sigmoid(logits) - targets.float()) ** 2).mean()
