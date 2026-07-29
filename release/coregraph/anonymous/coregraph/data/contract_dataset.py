"""Dataset container that keeps task units and graph semantics explicit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

import numpy as np

from coregraph.data.graph_views import GraphViewBundle, ViewRole
from coregraph.tasks.base import TaskAdapter, TaskBatch


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    variant: str
    source: str
    licence_status: str
    raw_checksum: str
    timestamp_quality: str
    correlated_domain: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ContractDataset:
    manifest: DatasetManifest
    task_adapter: TaskAdapter
    batch: TaskBatch
    node_ids: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_timestamps: Optional[np.ndarray]
    edge_types: Optional[np.ndarray]
    edge_attributes: Optional[np.ndarray]
    graph_views: Optional[GraphViewBundle] = None
    provenance: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if np.asarray(self.edge_index).ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        edge_count = self.edge_index.shape[1]
        for name in ("edge_timestamps", "edge_types", "edge_attributes"):
            value = getattr(self, name)
            if value is not None and len(value) != edge_count:
                raise ValueError(f"{name} must align to edge count")

    def summary(self) -> Mapping[str, object]:
        return {
            "dataset_id": self.manifest.dataset_id,
            "variant": self.manifest.variant,
            "prediction_unit": self.batch.prediction_unit.value,
            "examples": len(self.batch.identifiers),
            "nodes": len(self.node_ids),
            "edges": int(self.edge_index.shape[1]),
            "timestamp_quality": self.manifest.timestamp_quality,
            "correlated_domain": self.manifest.correlated_domain,
        }

    def batch_for_role(self, role: ViewRole) -> TaskBatch:
        if self.graph_views is None:
            if self.batch.graph_view is not None:
                raise ValueError("dataset has a graph view but no fold-specific graph bundle")
            return self.batch
        view = {
            ViewRole.TRAIN: self.graph_views.train,
            ViewRole.VALIDATION: self.graph_views.validation,
            ViewRole.TARGET: self.graph_views.target,
        }[role]
        return replace(self.batch, graph_view=view)
