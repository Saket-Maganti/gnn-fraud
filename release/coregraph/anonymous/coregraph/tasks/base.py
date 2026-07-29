"""Common task records and invariants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from coregraph.data.graph_views import GraphView


class _Value(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskType(_Value):
    NODE_CLASSIFICATION = "node_classification"
    EDGE_CLASSIFICATION = "edge_classification"
    TRANSACTION_CLASSIFICATION = "transaction_classification"


class PredictionUnit(_Value):
    NODE = "node"
    EDGE = "edge"
    TRANSACTION = "transaction"


@dataclass(frozen=True)
class TaskBatch:
    identifiers: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    timestamps: np.ndarray
    graph_view: Optional[GraphView]
    edge_attributes: Optional[np.ndarray]
    prediction_unit: PredictionUnit
    contract_id: str
    metadata: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.identifiers)
        arrays = {
            "features": self.features,
            "labels": self.labels,
            "label_mask": self.label_mask,
            "train_mask": self.train_mask,
            "validation_mask": self.validation_mask,
            "test_mask": self.test_mask,
            "timestamps": self.timestamps,
        }
        for name, value in arrays.items():
            if len(value) != n:
                raise ValueError(f"{name} length {len(value)} does not match identifiers {n}")
        if len(np.unique(self.identifiers.astype(str))) != n:
            raise ValueError("prediction identifiers must be unique within a task batch")
        for name in ("label_mask", "train_mask", "validation_mask", "test_mask"):
            if np.asarray(getattr(self, name)).dtype != np.bool_:
                raise ValueError(f"{name} must be boolean")
        if np.any(self.train_mask & self.validation_mask):
            raise ValueError("train and validation masks overlap")
        if np.any(self.train_mask & self.test_mask):
            raise ValueError("train and test masks overlap")
        if np.any(self.validation_mask & self.test_mask):
            raise ValueError("validation and test masks overlap")
        supervised = self.train_mask | self.validation_mask | self.test_mask
        if np.any(supervised & ~self.label_mask):
            raise ValueError("unknown labels cannot enter supervised masks")

    def subset(self, mask: np.ndarray) -> "TaskBatch":
        keep = np.asarray(mask, dtype=bool)
        if len(keep) != len(self.identifiers):
            raise ValueError("subset mask length mismatch")
        return TaskBatch(
            identifiers=self.identifiers[keep],
            features=self.features[keep],
            labels=self.labels[keep],
            label_mask=self.label_mask[keep],
            train_mask=self.train_mask[keep],
            validation_mask=self.validation_mask[keep],
            test_mask=self.test_mask[keep],
            timestamps=self.timestamps[keep],
            graph_view=self.graph_view,
            edge_attributes=(
                self.edge_attributes[keep] if self.edge_attributes is not None else None
            ),
            prediction_unit=self.prediction_unit,
            contract_id=self.contract_id,
            metadata=self.metadata,
        )


class TaskAdapter(ABC):
    """Dataset/task bridge used by experts, metrics, and prediction exports."""

    task_type: TaskType
    prediction_unit: PredictionUnit
    positive_label: int = 1
    unknown_label: int = 0

    def train_subset(self, batch: TaskBatch) -> TaskBatch:
        return batch.subset(batch.train_mask)

    def validation_subset(self, batch: TaskBatch) -> TaskBatch:
        return batch.subset(batch.validation_mask)

    def test_subset(self, batch: TaskBatch) -> TaskBatch:
        return batch.subset(batch.test_mask)

    def budget_count(self, batch: TaskBatch, budget: int | float) -> int:
        n = len(batch.identifiers)
        if isinstance(budget, float) and 0 < budget <= 1:
            return min(n, max(1, int(np.ceil(n * budget)))) if n else 0
        return min(n, max(0, int(budget)))

    def prediction_export_schema(self) -> tuple[str, ...]:
        return (
            f"{self.prediction_unit.value}_id",
            "contract_id",
            "split",
            "y_true",
            "label_known",
            "score",
            "expert_id",
            "config_hash",
        )

    def metric_compatible(self, metric: str) -> bool:
        return metric in {
            "auroc",
            "auprc",
            "f1",
            "precision",
            "recall",
            "precision_at_k",
            "recall_at_k",
            "brier",
            "ece",
        }

    def class_mapping(self) -> Mapping[int, str]:
        return {0: "unknown", 1: "fraud", 2: "normal"}

    @abstractmethod
    def build_batch(self, **kwargs: Any) -> TaskBatch:
        raise NotImplementedError

    @abstractmethod
    def construct_graph_view(self, **kwargs: Any) -> Optional[GraphView]:
        raise NotImplementedError


def split_name(batch: TaskBatch, index: int) -> str:
    if batch.train_mask[index]:
        return "train"
    if batch.validation_mask[index]:
        return "validation"
    if batch.test_mask[index]:
        return "test"
    return "unscored"


def align_prediction_rows(
    rows_by_expert: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    id_column: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Align expert scores by a typed identifier and verify labels agree."""

    if not rows_by_expert:
        raise ValueError("no expert predictions supplied")
    ids_by_expert: dict[str, set[str]] = {}
    row_maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for expert, rows in rows_by_expert.items():
        mapping: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            key = str(row[id_column])
            if key in mapping:
                raise ValueError(f"duplicate {id_column} {key!r} for expert {expert}")
            mapping[key] = row
        row_maps[expert] = mapping
        ids_by_expert[expert] = set(mapping)
    common = set.intersection(*ids_by_expert.values())
    if not common:
        raise ValueError("experts have no aligned prediction identifiers")
    ids = np.asarray(sorted(common))
    scores: dict[str, np.ndarray] = {}
    labels: Optional[np.ndarray] = None
    for expert, mapping in sorted(row_maps.items()):
        scores[expert] = np.asarray([float(mapping[key]["score"]) for key in ids])
        current = np.asarray([int(mapping[key]["y_true"]) for key in ids])
        if labels is None:
            labels = current
        elif not np.array_equal(labels, current):
            raise ValueError(f"label mismatch after alignment for expert {expert}")
    assert labels is not None
    return ids, scores, labels
