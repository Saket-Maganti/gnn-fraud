"""Differentiable calibration penalties with explicit score semantics."""

from __future__ import annotations

import torch

from coregraph.objectives.scores import ScoreType, probabilities


def soft_brier_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
) -> torch.Tensor:
    return ((probabilities(scores, score_type) - targets.float()) ** 2).mean()
