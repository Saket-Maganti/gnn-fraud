"""Selective-prediction helpers and source-validation thresholding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AbstentionThreshold:
    threshold: float
    coverage: float
    selective_risk: float
    abstention_fraction: float
    capacity: float
    fitted_on: str = "source_validation"


def apply_abstention_capacity(
    abstention_probability: torch.Tensor,
    capacity: float,
    *,
    forced_abstention: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0 <= capacity <= 1:
        raise ValueError("abstention capacity must lie in [0,1]")
    flat = abstention_probability.reshape(-1)
    forced = (
        torch.zeros_like(flat, dtype=torch.bool)
        if forced_abstention is None
        else forced_abstention.reshape(-1).bool()
    )
    if forced.shape != flat.shape:
        raise ValueError("forced abstention must align with probabilities")
    n = flat.numel()
    capacity_k = min(n, max(0, int(round(capacity * n))))
    decision = forced.clone()
    remaining = max(0, capacity_k - int(forced.sum().item()))
    if remaining:
        candidates = torch.where(~forced)[0]
        ordered = candidates[
            torch.argsort(flat[candidates], descending=True, stable=True)
        ]
        decision[ordered[:remaining]] = True
    return decision.reshape(abstention_probability.shape)


def coverage(abstain: torch.Tensor) -> torch.Tensor:
    if abstain.numel() == 0:
        raise ValueError("coverage requires at least one decision")
    return (~abstain.bool()).float().mean()


def selective_risk(
    losses: torch.Tensor,
    abstain: torch.Tensor,
) -> torch.Tensor:
    if losses.shape != abstain.shape:
        raise ValueError("losses and abstention decisions must align")
    accepted = ~abstain.bool()
    if not accepted.any():
        return losses.new_tensor(float("nan"))
    return losses[accepted].mean()


def area_under_risk_coverage_curve(
    losses: torch.Tensor,
    abstention_probability: torch.Tensor,
) -> torch.Tensor:
    """Mean prefix risk when accepting least-abstention examples first."""

    if losses.shape != abstention_probability.shape or losses.numel() == 0:
        raise ValueError("AURC inputs must be non-empty aligned tensors")
    order = torch.argsort(
        abstention_probability.reshape(-1),
        descending=False,
        stable=True,
    )
    ordered_losses = losses.reshape(-1)[order]
    prefix = ordered_losses.cumsum(dim=0) / torch.arange(
        1,
        ordered_losses.numel() + 1,
        device=losses.device,
        dtype=losses.dtype,
    )
    return prefix.mean()


def abstention_cost(abstain: torch.Tensor, *, cost: float) -> torch.Tensor:
    if cost < 0:
        raise ValueError("abstention cost cannot be negative")
    return abstain.float().mean() * cost


def abstention_capacity_penalty(
    abstention_probability: torch.Tensor,
    *,
    capacity: float,
) -> torch.Tensor:
    if not 0 <= capacity <= 1:
        raise ValueError("abstention capacity must lie in [0,1]")
    return torch.relu(abstention_probability.mean() - capacity)


def select_abstention_threshold(
    validation_losses: torch.Tensor,
    abstention_probability: torch.Tensor,
    *,
    capacity: float,
    abstention_cost_value: float = 0.0,
) -> AbstentionThreshold:
    """Choose a source-validation threshold under a declared capacity."""

    if validation_losses.shape != abstention_probability.shape:
        raise ValueError("validation losses and abstention scores must align")
    if validation_losses.numel() == 0:
        raise ValueError("abstention threshold needs source-validation examples")
    if not 0 <= capacity <= 1 or abstention_cost_value < 0:
        raise ValueError("invalid abstention threshold configuration")
    values = torch.unique(abstention_probability.detach()).sort().values
    candidates = torch.cat(
        [
            values.new_tensor([torch.inf]),
            values,
            values.new_tensor([-torch.inf]),
        ]
    )
    best: tuple[float, float, float, float] | None = None
    for threshold_tensor in candidates:
        decision = abstention_probability >= threshold_tensor
        fraction = float(decision.float().mean().item())
        if fraction > capacity + 1e-12:
            continue
        risk = selective_risk(validation_losses, decision)
        if not torch.isfinite(risk):
            continue
        objective = float(
            risk.item() + abstention_cost_value * fraction
        )
        record = (
            objective,
            -float(coverage(decision).item()),
            float(threshold_tensor.item()),
            fraction,
        )
        if best is None or record < best:
            best = record
    if best is None:
        raise ValueError("no finite abstention threshold satisfies capacity")
    _, negative_coverage, threshold, fraction = best
    decision = abstention_probability >= threshold
    return AbstentionThreshold(
        threshold=threshold,
        coverage=-negative_coverage,
        selective_risk=float(
            selective_risk(validation_losses, decision).item()
        ),
        abstention_fraction=fraction,
        capacity=capacity,
    )
