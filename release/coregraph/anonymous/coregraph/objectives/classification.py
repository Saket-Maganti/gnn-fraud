"""Binary predictive losses with explicit score semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from coregraph.objectives.scores import ScoreType, probabilities, validate_scores


def binary_cross_entropy_values(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
    positive_weight: float | None = None,
) -> torch.Tensor:
    validate_scores(scores, score_type)
    if score_type is ScoreType.RANK_SCORE:
        raise ValueError("rank scores are incompatible with binary cross entropy")
    target = targets.float()
    if target.shape != scores.shape:
        raise ValueError("binary targets and scores must align")
    if score_type is ScoreType.LOGIT:
        pos_weight = (
            scores.new_tensor(positive_weight)
            if positive_weight is not None
            else None
        )
        return F.binary_cross_entropy_with_logits(
            scores,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
    weights = None
    if positive_weight is not None:
        weights = torch.where(
            target.bool(),
            scores.new_tensor(positive_weight),
            scores.new_tensor(1.0),
        )
    return F.binary_cross_entropy(
        scores,
        target,
        weight=weights,
        reduction="none",
    )


def binary_cross_entropy(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
    positive_weight: float | None = None,
) -> torch.Tensor:
    return binary_cross_entropy_values(
        scores,
        targets,
        score_type=score_type,
        positive_weight=positive_weight,
    ).mean()


def focal_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    if gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("focal gamma must be non-negative and alpha in [0,1]")
    targets_float = targets.float()
    base = binary_cross_entropy_values(
        scores,
        targets_float,
        score_type=score_type,
    )
    probability = probabilities(scores, score_type)
    p_t = (
        probability * targets_float
        + (1 - probability) * (1 - targets_float)
    )
    alpha_t = (
        alpha * targets_float
        + (1 - alpha) * (1 - targets_float)
    )
    return (alpha_t * (1 - p_t).pow(gamma) * base).mean()


def class_balanced_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
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
    loss = binary_cross_entropy_values(
        scores,
        target.float(),
        score_type=score_type,
    )
    return (sample_weight * loss).mean()
