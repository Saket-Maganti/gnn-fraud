"""Bounded interactions for factorised contract representations."""

from __future__ import annotations

import torch


def bounded_pairwise_interactions(
    axis_embeddings: torch.Tensor,
    *,
    bound: float = 1.0,
) -> torch.Tensor:
    """Return bounded pairwise dot products for ``[batch, axes, dim]``."""

    if axis_embeddings.ndim != 3:
        raise ValueError("axis embeddings must have shape [batch,axes,dimension]")
    if bound <= 0:
        raise ValueError("interaction bound must be positive")
    interactions = []
    scale = axis_embeddings.shape[-1] ** 0.5
    for left in range(axis_embeddings.shape[1]):
        for right in range(left + 1, axis_embeddings.shape[1]):
            raw = (axis_embeddings[:, left] * axis_embeddings[:, right]).sum(-1) / scale
            interactions.append(torch.tanh(raw / bound) * bound)
    if not interactions:
        return axis_embeddings.new_zeros((axis_embeddings.shape[0], 0))
    return torch.stack(interactions, dim=-1)
