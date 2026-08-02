"""Task- and contract-aware expert API."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from coregraph.contracts.axes import (
    DeviceClass,
    EdgeFeaturePolicy,
    EdgeVisibility,
    TopologyTransform,
)
from coregraph.contracts.contract import DeploymentContract
from coregraph.tasks.base import PredictionUnit, TaskBatch, TaskType, split_name


class OfficialStatus(str, Enum):
    OFFICIAL_CODE = "OFFICIAL_CODE"
    VALIDATED_REIMPLEMENTATION = "VALIDATED_REIMPLEMENTATION"
    DIAGNOSTIC_APPROXIMATION = "DIAGNOSTIC_APPROXIMATION"
    PENDING_INTEGRATION = "PENDING_INTEGRATION"
    UNAVAILABLE_LICENSE = "UNAVAILABLE_LICENSE"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


class AvailabilityReason(str, Enum):
    AVAILABLE = "available"
    EXPLICITLY_UNAVAILABLE = "explicitly_unavailable"
    DEVICE_UNDECLARED = "device_undeclared"
    DEVICE_INCOMPATIBLE = "device_incompatible"
    MEMORY_CAP_EXCEEDED = "memory_cap_exceeded"
    LATENCY_CAP_EXCEEDED = "latency_cap_exceeded"
    TASK_UNSUPPORTED = "task_unsupported"
    GRAPH_CONTRACT_UNAVAILABLE = "graph_contract_unavailable"
    GRAPH_DATA_MISSING = "graph_data_missing"
    EDGE_FEATURES_CONTRACT_UNAVAILABLE = "edge_features_contract_unavailable"
    EDGE_FEATURES_DATA_MISSING = "edge_features_data_missing"
    FULL_GRAPH_NODE_GUARD = "full_graph_node_guard"
    FULL_GRAPH_EDGE_GUARD = "full_graph_edge_guard"
    LICENSE_UNAVAILABLE = "license_unavailable"
    INTEGRATION_PENDING = "integration_pending"
    INTEGRATION_RESOURCE_BLOCKED = "integration_resource_blocked"


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

    def __post_init__(self) -> None:
        if self.min_memory_gb < 0 or self.expected_latency_ms < 0:
            raise ValueError("resource requirements cannot be negative")
        if not self.device_classes:
            raise ValueError("at least one compatible device class is required")


@dataclass(frozen=True)
class Availability:
    available: bool
    reason_codes: tuple[AvailabilityReason, ...]
    status: OfficialStatus

    @property
    def reason(self) -> str:
        return ",".join(reason.value for reason in self.reason_codes)


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
        if (
            requirements.requires_graph
            and (
                contract.visibility.edge_visibility is EdgeVisibility.NONE
                or contract.construction.topology_transform
                in {TopologyTransform.NO_GRAPH, TopologyTransform.DEGREE_ONLY}
            )
        ):
            return False
        return True

    def availability(self, batch: TaskBatch, contract: DeploymentContract) -> Availability:
        requirements = self.resource_requirements()
        reasons: list[AvailabilityReason] = []
        if self.expert_id in contract.resource.unavailable_experts:
            reasons.append(AvailabilityReason.EXPLICITLY_UNAVAILABLE)
        device = contract.resource.device_class
        if device is DeviceClass.UNKNOWN:
            reasons.append(AvailabilityReason.DEVICE_UNDECLARED)
        elif device.value not in requirements.device_classes:
            reasons.append(AvailabilityReason.DEVICE_INCOMPATIBLE)
        if (
            contract.resource.memory_cap_gb is not None
            and requirements.min_memory_gb > contract.resource.memory_cap_gb
        ):
            reasons.append(AvailabilityReason.MEMORY_CAP_EXCEEDED)
        if (
            contract.resource.latency_cap_ms is not None
            and requirements.expected_latency_ms > contract.resource.latency_cap_ms
        ):
            reasons.append(AvailabilityReason.LATENCY_CAP_EXCEEDED)
        task = {
            PredictionUnit.NODE: TaskType.NODE_CLASSIFICATION,
            PredictionUnit.EDGE: TaskType.EDGE_CLASSIFICATION,
            PredictionUnit.TRANSACTION: TaskType.TRANSACTION_CLASSIFICATION,
        }[batch.prediction_unit]
        if not self.supports_task(task):
            reasons.append(AvailabilityReason.TASK_UNSUPPORTED)
        if requirements.requires_graph and not self.supports_contract(contract):
            reasons.append(AvailabilityReason.GRAPH_CONTRACT_UNAVAILABLE)
        if requirements.requires_graph and batch.graph_view is None:
            reasons.append(AvailabilityReason.GRAPH_DATA_MISSING)
        if requirements.requires_edge_features:
            if (
                contract.construction.edge_feature_policy
                is EdgeFeaturePolicy.DROP
            ):
                reasons.append(
                    AvailabilityReason.EDGE_FEATURES_CONTRACT_UNAVAILABLE
                )
            graph_edge_features = (
                batch.graph_view.edge_attributes
                if batch.graph_view is not None
                else None
            )
            if batch.edge_attributes is None and graph_edge_features is None:
                reasons.append(AvailabilityReason.EDGE_FEATURES_DATA_MISSING)
        n_nodes = (
            len(batch.graph_view.visible_node_ids) if batch.graph_view is not None else 0
        )
        n_edges = batch.graph_view.edge_count if batch.graph_view is not None else 0
        if requirements.max_nodes_full_graph is not None and n_nodes > requirements.max_nodes_full_graph:
            reasons.append(AvailabilityReason.FULL_GRAPH_NODE_GUARD)
        if requirements.max_edges_full_graph is not None and n_edges > requirements.max_edges_full_graph:
            reasons.append(AvailabilityReason.FULL_GRAPH_EDGE_GUARD)
        status_reason = {
            OfficialStatus.PENDING_INTEGRATION: AvailabilityReason.INTEGRATION_PENDING,
            OfficialStatus.UNAVAILABLE_LICENSE: AvailabilityReason.LICENSE_UNAVAILABLE,
            OfficialStatus.RESOURCE_BLOCKED: (
                AvailabilityReason.INTEGRATION_RESOURCE_BLOCKED
            ),
        }.get(self.official_status)
        if status_reason is not None:
            reasons.append(status_reason)
        if reasons:
            resource_codes = {
                AvailabilityReason.EXPLICITLY_UNAVAILABLE,
                AvailabilityReason.DEVICE_UNDECLARED,
                AvailabilityReason.DEVICE_INCOMPATIBLE,
                AvailabilityReason.MEMORY_CAP_EXCEEDED,
                AvailabilityReason.LATENCY_CAP_EXCEEDED,
                AvailabilityReason.FULL_GRAPH_NODE_GUARD,
                AvailabilityReason.FULL_GRAPH_EDGE_GUARD,
                AvailabilityReason.INTEGRATION_RESOURCE_BLOCKED,
            }
            status = (
                OfficialStatus.RESOURCE_BLOCKED
                if resource_codes.intersection(reasons)
                else self.official_status
            )
            return Availability(False, tuple(dict.fromkeys(reasons)), status)
        return Availability(
            True,
            (AvailabilityReason.AVAILABLE,),
            self.official_status,
        )

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
                    "score_type": "PROBABILITY",
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
