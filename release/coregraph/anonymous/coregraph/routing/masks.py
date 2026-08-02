"""Exact resource-feasible masking before expert selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskedRouting:
    probabilities: torch.Tensor
    feasible_mask: torch.Tensor
    all_unavailable: torch.Tensor
    selected_expert: torch.Tensor


def compose_feasible_mask(
    availability: torch.Tensor,
    *,
    memory_gb: torch.Tensor | None = None,
    memory_cap_gb: torch.Tensor | float | None = None,
    latency_ms: torch.Tensor | None = None,
    latency_cap_ms: torch.Tensor | float | None = None,
    invocation_cost: torch.Tensor | None = None,
    invocation_cap: torch.Tensor | float | None = None,
) -> torch.Tensor:
    mask = availability.bool().clone()
    if mask.ndim != 2:
        raise ValueError("availability must have shape [batch,experts]")

    def apply(values: torch.Tensor | None, cap: torch.Tensor | float | None, name: str) -> None:
        nonlocal mask
        if (values is None) != (cap is None):
            raise ValueError(f"{name} values and cap must be supplied together")
        if values is None:
            return
        values = values.to(device=mask.device, dtype=torch.float32)
        if values.shape == (mask.shape[1],):
            values = values[None, :].expand(mask.shape[0], -1)
        if values.shape != mask.shape or not torch.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError(f"{name} values must be finite non-negative [batch,experts]")
        cap_tensor = torch.as_tensor(cap, device=mask.device, dtype=torch.float32)
        if cap_tensor.ndim == 0:
            cap_tensor = cap_tensor.expand(mask.shape[0])
        if cap_tensor.shape != (mask.shape[0],) or bool((cap_tensor < 0).any()):
            raise ValueError(f"{name} cap must be non-negative scalar or [batch]")
        mask &= values <= cap_tensor[:, None]

    apply(memory_gb, memory_cap_gb, "memory")
    apply(latency_ms, latency_cap_ms, "latency")
    apply(invocation_cost, invocation_cap, "invocation")
    return mask


def apply_feasible_mask(logits: torch.Tensor, feasible_mask: torch.Tensor) -> MaskedRouting:
    if logits.shape != feasible_mask.shape or logits.ndim != 2:
        raise ValueError("logits and feasible mask must align as [batch,experts]")
    mask = feasible_mask.bool()
    all_unavailable = ~mask.any(dim=-1)
    masked_logits = logits.masked_fill(~mask & ~all_unavailable[:, None], -torch.inf)
    safe_logits = torch.where(all_unavailable[:, None], torch.zeros_like(masked_logits), masked_logits)
    probabilities = torch.softmax(safe_logits, dim=-1) * mask.float()
    # Re-normalize to make the invariant explicit even on unusual dtypes.
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        all_unavailable[:, None],
        torch.zeros_like(probabilities),
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
    )
    selected = probabilities.argmax(dim=-1)
    selected = torch.where(all_unavailable, torch.full_like(selected, -1), selected)
    return MaskedRouting(probabilities, mask, all_unavailable, selected)
