"""Estimated or measured expert activation cost."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CostProvenance(str, Enum):
    DRY_RUN_ESTIMATE = "DRY_RUN_ESTIMATE"
    PROFILED = "PROFILED"
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"  # Legacy import only.


@dataclass(frozen=True)
class ExpertCost:
    expert_id: str
    latency_ms: float
    memory_gb: float
    flops: float | None
    provenance: CostProvenance

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.memory_gb < 0:
            raise ValueError("expert latency and memory costs cannot be negative")
        if self.flops is not None and self.flops < 0:
            raise ValueError("expert FLOP cost cannot be negative")


def expected_compute_cost(
    expert_weights: torch.Tensor,
    costs: torch.Tensor,
) -> torch.Tensor:
    if costs.shape == (expert_weights.shape[-1],):
        pass
    elif costs.shape != expert_weights.shape:
        raise ValueError(
            "compute costs must be per expert or per example-expert"
        )
    return (expert_weights * costs).sum(dim=-1).mean()
