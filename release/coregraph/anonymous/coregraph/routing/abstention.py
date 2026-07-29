"""Selective-prediction helpers and source-validation thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class AbstentionThreshold:
    threshold: float
    coverage: float
    selective_risk: float
    abstention_fraction: float
    capacity: float
    fitted_on: str = "source_validation"


@dataclass(frozen=True)
class GroupedAbstentionThreshold:
    threshold: float
    balanced_selective_objective: float
    group_coverages: Mapping[int, float]
    group_abstention_fractions: Mapping[int, float]
    group_capacities: Mapping[int, float | None]
    fitted_on: str = "source_validation_balanced_contracts"


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


def apply_frozen_abstention_decision(
    abstention_probability: torch.Tensor,
    *,
    threshold: float,
    capacity: float | None,
    forced_abstention: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a frozen threshold, then a label-free target capacity.

    Capacity can remove threshold-selected abstentions but never invents
    additional abstentions merely to fill the operational allowance. Forced
    abstentions remain selected even when they alone exceed the allowance.
    """

    flat = abstention_probability.reshape(-1)
    forced = (
        torch.zeros_like(flat, dtype=torch.bool)
        if forced_abstention is None
        else forced_abstention.reshape(-1).bool()
    )
    if forced.shape != flat.shape:
        raise ValueError("forced abstention must align with probabilities")
    learned = flat >= threshold
    if capacity is None:
        return (forced | learned).reshape(abstention_probability.shape)
    if not 0 <= capacity <= 1:
        raise ValueError("abstention capacity must lie in [0,1]")
    allowed = max(0, int(round(capacity * flat.numel())))
    remaining = max(0, allowed - int(forced.sum().item()))
    decision = forced.clone()
    if remaining:
        candidates = torch.where(learned & ~forced)[0]
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
    upper = torch.nextafter(
        values[-1],
        values.new_tensor(torch.inf),
    ).reshape(1)
    lower = torch.nextafter(
        values[0],
        values.new_tensor(-torch.inf),
    ).reshape(1)
    candidates = torch.cat(
        [
            upper,
            values,
            lower,
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


def select_grouped_abstention_threshold(
    validation_losses: torch.Tensor,
    abstention_probability: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    capacities: Mapping[int, float | None],
    abstention_cost_value: float = 0.0,
    forced_abstention: torch.Tensor | None = None,
) -> GroupedAbstentionThreshold:
    """Select one threshold while enforcing every source contract separately."""

    if (
        validation_losses.shape != abstention_probability.shape
        or group_indices.shape != validation_losses.shape
    ):
        raise ValueError("grouped threshold inputs must be aligned vectors")
    if validation_losses.numel() == 0:
        raise ValueError("grouped threshold needs source-validation examples")
    if abstention_cost_value < 0:
        raise ValueError("abstention cost cannot be negative")
    groups = tuple(
        int(value)
        for value in torch.unique(group_indices, sorted=True).tolist()
    )
    if set(capacities) != set(groups):
        raise ValueError("source capacity map must cover every contract group exactly")
    for capacity in capacities.values():
        if capacity is not None and not 0 <= capacity <= 1:
            raise ValueError("abstention capacity must lie in [0,1]")
    forced = (
        torch.zeros_like(validation_losses, dtype=torch.bool)
        if forced_abstention is None
        else forced_abstention.bool()
    )
    if forced.shape != validation_losses.shape:
        raise ValueError("forced abstention must align with grouped threshold inputs")
    values = torch.unique(abstention_probability.detach()).sort().values
    upper = torch.nextafter(
        values[-1],
        values.new_tensor(torch.inf),
    ).reshape(1)
    lower = torch.nextafter(
        values[0],
        values.new_tensor(-torch.inf),
    ).reshape(1)
    candidates = torch.cat(
        (
            upper,
            values,
            lower,
        )
    )
    best: tuple[
        float,
        float,
        dict[int, float],
        dict[int, float],
    ] | None = None
    for threshold_tensor in candidates:
        decision = forced | (abstention_probability >= threshold_tensor)
        group_objectives: list[float] = []
        coverages: dict[int, float] = {}
        fractions: dict[int, float] = {}
        valid = True
        for group in groups:
            keep = group_indices == group
            group_decision = decision[keep]
            fraction = float(group_decision.float().mean().item())
            capacity = capacities[group]
            if capacity is not None and fraction > capacity + 1e-12:
                valid = False
                break
            group_risk = selective_risk(
                validation_losses[keep],
                group_decision,
            )
            if not torch.isfinite(group_risk):
                valid = False
                break
            group_objectives.append(
                float(group_risk.item()) + abstention_cost_value * fraction
            )
            coverages[group] = float(coverage(group_decision).item())
            fractions[group] = fraction
        if not valid:
            continue
        objective = float(sum(group_objectives) / len(group_objectives))
        record = (
            objective,
            float(threshold_tensor.item()),
            coverages,
            fractions,
        )
        if best is None or record[:2] < best[:2]:
            best = record
    if best is None:
        raise ValueError(
            "no finite grouped abstention threshold satisfies every source contract"
        )
    objective, threshold, coverages, fractions = best
    return GroupedAbstentionThreshold(
        threshold=threshold,
        balanced_selective_objective=objective,
        group_coverages=coverages,
        group_abstention_fractions=fractions,
        group_capacities=dict(capacities),
    )
