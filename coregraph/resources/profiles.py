"""Typed resource interventions used by counterfactual and future runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceProfile:
    profile_id: str
    unavailable_experts: tuple[str, ...] = ()
    memory_cap_gb: float | None = None
    latency_cap_ms: float | None = None
    review_budget_fraction: float | None = None
    dynamic: bool = False

    def __post_init__(self) -> None:
        if self.memory_cap_gb is not None and self.memory_cap_gb <= 0:
            raise ValueError("memory cap must be positive")
        if self.latency_cap_ms is not None and self.latency_cap_ms <= 0:
            raise ValueError("latency cap must be positive")
        if self.review_budget_fraction is not None and not 0 <= self.review_budget_fraction <= 1:
            raise ValueError("review budget fraction must lie in [0,1]")


def standard_profiles() -> tuple[ResourceProfile, ...]:
    return (
        ResourceProfile("all_experts_available"),
        ResourceProfile("one_graph_expert_unavailable", ("gcn",)),
        ResourceProfile("all_graph_experts_unavailable", ("gcn", "graphsage")),
        ResourceProfile("tight_memory", memory_cap_gb=4.0),
        ResourceProfile("tight_latency", latency_cap_ms=10.0),
        ResourceProfile("tight_review_budget", review_budget_fraction=0.005),
        ResourceProfile("combined_graph_resource_shift", ("graphsage",), memory_cap_gb=4.0),
        ResourceProfile("dynamic_availability_change", dynamic=True),
    )
