"""Task- and contract-aware expert API."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.tasks.base import TaskBatch, TaskType, split_name


class OfficialStatus(str, Enum):
    OFFICIAL_CODE = "OFFICIAL_CODE"
    VALIDATED_REIMPLEMENTATION = "VALIDATED_REIMPLEMENTATION"
    DIAGNOSTIC_APPROXIMATION = "DIAGNOSTIC_APPROXIMATION"
    PENDING_INTEGRATION = "PENDING_INTEGRATION"
    UNAVAILABLE_LICENSE = "UNAVAILABLE_LICENSE"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


@dataclass(frozen=True)
class ResourceRequirements:
    min_memory_gb: float = 0.0
    expected_latency_ms: float = 0.0
    device_classes: tuple[str, ...] = ("cpu",)
    requires_graph: bool = False
    requires_edge_features: bool = False
    max_nodes_full_graph: Optional[int] = None
    max_edges_full_graph: Optional[int] = None
    cost_provenance: str = "DRY_RUN_ESTIMATE"


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str
    status: OfficialStatus


class Expert(ABC):
    expert_id: str
    official_status: OfficialStatus
    supported_tasks: tuple[TaskType, ...]

    @abstractmethod
    def fit(self, batch: TaskBatch) -> "Expert":
        raise NotImplementedError

    @abstractmethod
    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        raise NotImplementedError

    def predict_embeddings(self, batch: TaskBatch) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def resource_requirements(self) -> ResourceRequirements:
        raise NotImplementedError

    def supports_task(self, task: TaskType) -> bool:
        return task in self.supported_tasks

    def supports_contract(self, contract: DeploymentContract) -> bool:
        requirements = self.resource_requirements()
        if requirements.requires_graph and contract.visibility.value == "missing_graph":
            return False
        return True

    def availability(self, batch: TaskBatch, contract: DeploymentContract) -> Availability:
        if not self.supports_contract(contract):
            return Availability(False, "contract_incompatible", self.official_status)
        requirements = self.resource_requirements()
        if requirements.requires_graph and batch.graph_view is None:
            return Availability(False, "graph_missing", self.official_status)
        if requirements.requires_edge_features and batch.edge_attributes is None:
            return Availability(False, "edge_features_missing", self.official_status)
        n_nodes = (
            len(batch.graph_view.visible_node_ids) if batch.graph_view is not None else 0
        )
        n_edges = batch.graph_view.edge_count if batch.graph_view is not None else 0
        if requirements.max_nodes_full_graph is not None and n_nodes > requirements.max_nodes_full_graph:
            return Availability(False, "node_guard_requires_sampling", OfficialStatus.RESOURCE_BLOCKED)
        if requirements.max_edges_full_graph is not None and n_edges > requirements.max_edges_full_graph:
            return Availability(False, "edge_guard_requires_sampling", OfficialStatus.RESOURCE_BLOCKED)
        if self.official_status in {
            OfficialStatus.PENDING_INTEGRATION,
            OfficialStatus.UNAVAILABLE_LICENSE,
            OfficialStatus.RESOURCE_BLOCKED,
        }:
            return Availability(False, self.official_status.value.lower(), self.official_status)
        return Availability(True, "available", self.official_status)

    def export_predictions(
        self,
        batch: TaskBatch,
        scores: np.ndarray,
        *,
        config_hash: str,
    ) -> list[dict[str, Any]]:
        if len(scores) != len(batch.identifiers):
            raise ValueError("prediction scores do not align to task examples")
        rows = []
        id_column = f"{batch.prediction_unit.value}_id"
        for index, identifier in enumerate(batch.identifiers):
            rows.append(
                {
                    id_column: str(identifier),
                    "contract_id": batch.contract_id,
                    "split": split_name(batch, index),
                    "y_true": int(batch.labels[index]),
                    "label_known": bool(batch.label_mask[index]),
                    "score": float(scores[index]),
                    "expert_id": self.expert_id,
                    "config_hash": config_hash,
                    "official_status": self.official_status.value,
                }
            )
        return rows

    def config_hash(self) -> str:
        payload = {
            "class": type(self).__name__,
            "expert_id": self.expert_id,
            "official_status": self.official_status.value,
            "state": {
                key: value
                for key, value in self.__dict__.items()
                if isinstance(value, (str, int, float, bool, type(None), tuple))
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
