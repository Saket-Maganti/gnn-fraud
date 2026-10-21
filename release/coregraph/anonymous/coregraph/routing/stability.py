"""Router consistency penalties and metrics."""

from __future__ import annotations

import torch


def consistency_penalty(
    weights: torch.Tensor,
    perturbed_weights: torch.Tensor,
) -> torch.Tensor:
    if weights.shape != perturbed_weights.shape:
        raise ValueError("routing weights must align")
    return ((weights - perturbed_weights) ** 2).sum(dim=-1).mean()


def routing_flip_rate(
    weights: torch.Tensor,
    perturbed_weights: torch.Tensor,
) -> float:
    if weights.shape != perturbed_weights.shape:
        raise ValueError("routing weights must align")
    flips = weights.argmax(dim=-1) != perturbed_weights.argmax(dim=-1)
    return float(flips.float().mean().item())


def empirical_lipschitz_penalty(
    weights: torch.Tensor,
    perturbed_weights: torch.Tensor,
    perturbation_norm: torch.Tensor,
    *,
    limit: float = 1.0,
) -> torch.Tensor:
    response = torch.linalg.vector_norm(weights - perturbed_weights, dim=-1)
    ratio = response / perturbation_norm.clamp_min(1e-8)
    return torch.relu(ratio - limit).mean()


def availability_mask_consistency(
    original_weights: torch.Tensor,
    masked_weights: torch.Tensor,
    availability_mask: torch.Tensor,
) -> torch.Tensor:
    retained = original_weights * availability_mask.float()
    retained = retained / retained.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return ((retained - masked_weights) ** 2).sum(dim=-1).mean()
