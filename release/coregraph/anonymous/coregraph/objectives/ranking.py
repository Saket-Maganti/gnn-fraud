"""Mini-batch pairwise ranking surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from coregraph.objectives.scores import ScoreType, validate_scores


def pairwise_logistic_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_type: ScoreType,
    max_pairs: int = 8192,
) -> torch.Tensor:
    """Penalise positive scores below negative scores.

    This is a smooth ordering surrogate related to AUROC and useful for rare
    positives. It does not exactly optimise AUPRC or a discontinuous top-K
    metric. Pair subsampling is deterministic in tensor order.
    """

    validate_scores(scores, score_type)
    positives = scores[targets.bool()]
    negatives = scores[~targets.bool()]
    if positives.numel() == 0 or negatives.numel() == 0:
        return scores.sum() * 0
    pairs = positives[:, None] - negatives[None, :]
    flat = pairs.reshape(-1)
    if flat.numel() > max_pairs:
        step = max(1, flat.numel() // max_pairs)
        flat = flat[::step][:max_pairs]
    return F.softplus(-flat).mean()
