"""Legacy graph-model wrappers with sampling and status guards."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from coregraph.experts.base import Expert, OfficialStatus, ResourceRequirements
from coregraph.tasks.base import TaskBatch, TaskType

DIAGNOSTIC_LEGACY_MODELS = {
    "graph_transformer",
    "transformer",
    "gps",
    "gps_local",
    "gps_light",
    "pcgnn",
    "snapshot_tgn",
    "snapshot_tgn_light",
    "tgn",
}


@dataclass
class LegacyGraphExpert(Expert):
    """Metadata-safe wrapper.

    Full fitting is delegated to the later sampled expert runner. This wrapper
    exists now to enforce feasibility and prevent diagnostic promotion.
    """

    model_name: str
    expert_id: str
    official_status: OfficialStatus = OfficialStatus.DIAGNOSTIC_APPROXIMATION
    supported_tasks: tuple[TaskType, ...] = (TaskType.NODE_CLASSIFICATION,)
    neighbor_sampling: bool = True
    max_full_graph_nodes: int = 100_000
    max_full_graph_edges: int = 2_000_000

    def __post_init__(self) -> None:
        if self.model_name in DIAGNOSTIC_LEGACY_MODELS:
            self.official_status = OfficialStatus.DIAGNOSTIC_APPROXIMATION
            warnings.warn(
                f"{self.model_name} is a diagnostic approximation and cannot be "
                "reported as an official headline baseline",
                RuntimeWarning,
                stacklevel=2,
            )

    def fit(self, batch: TaskBatch) -> "LegacyGraphExpert":
        available = self.availability(batch, batch.graph_view.contract) if batch.graph_view else None
        if available is not None and not available.available:
            raise RuntimeError(available.reason)
        raise NotImplementedError(
            "Use coregraph.experiments.runner sampled_graph_fit for later real-data "
            "execution; the pre-run adapter never performs full-graph dense training"
        )

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        raise RuntimeError("legacy graph expert has no fitted sampled checkpoint")

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            min_memory_gb=4.0,
            expected_latency_ms=50.0,
            device_classes=("cpu", "single_t4", "dual_t4"),
            requires_graph=True,
            max_nodes_full_graph=None if self.neighbor_sampling else self.max_full_graph_nodes,
            max_edges_full_graph=None if self.neighbor_sampling else self.max_full_graph_edges,
            cost_provenance="DRY_RUN_ESTIMATE",
        )


def estimate_full_graph_memory_gb(
    *,
    nodes: int,
    edges: int,
    hidden_channels: int,
    attention: bool,
    bytes_per_value: int = 4,
) -> float:
    activations = nodes * hidden_channels * bytes_per_value * 8
    adjacency = edges * 2 * 8
    attention_cost = nodes * nodes * bytes_per_value if attention else 0
    return float((activations + adjacency + attention_cost) / 1024**3)


def guard_full_graph_execution(
    *,
    nodes: int,
    edges: int,
    memory_cap_gb: float,
    hidden_channels: int,
    attention: bool,
) -> float:
    estimate = estimate_full_graph_memory_gb(
        nodes=nodes,
        edges=edges,
        hidden_channels=hidden_channels,
        attention=attention,
    )
    if estimate > memory_cap_gb:
        raise MemoryError(
            f"estimated full-graph allocation {estimate:.2f} GiB exceeds "
            f"{memory_cap_gb:.2f} GiB; enable neighbour/event sampling"
        )
    return estimate
