"""Faithful temporal-model integration boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coregraph.experts.base import Expert, OfficialStatus, ResourceRequirements
from coregraph.tasks.base import TaskBatch, TaskType


@dataclass
class ExternalTemporalExpert(Expert):
    expert_id: str
    method_name: str
    official_status: OfficialStatus = OfficialStatus.PENDING_INTEGRATION
    supported_tasks: tuple[TaskType, ...] = (
        TaskType.EDGE_CLASSIFICATION,
        TaskType.TRANSACTION_CLASSIFICATION,
    )

    def fit(self, batch: TaskBatch) -> "ExternalTemporalExpert":
        raise RuntimeError(
            f"{self.method_name} official code is not installed. Run the pinned "
            "acquisition command from external_baselines/BASELINE_REGISTRY.yaml."
        )

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        raise RuntimeError(f"{self.method_name} official checkpoint is unavailable")

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            min_memory_gb=8.0,
            device_classes=("single_t4", "dual_t4"),
            requires_graph=True,
            cost_provenance="UNKNOWN",
        )
