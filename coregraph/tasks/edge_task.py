"""Edge-classification adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from coregraph.data.graph_views import GraphView
from coregraph.tasks.base import PredictionUnit, TaskAdapter, TaskBatch, TaskType


class EdgeTaskAdapter(TaskAdapter):
    task_type = TaskType.EDGE_CLASSIFICATION
    prediction_unit = PredictionUnit.EDGE

    def build_batch(
        self,
        *,
        edge_ids: np.ndarray,
        edge_features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
        test_mask: np.ndarray,
        timestamps: np.ndarray,
        graph_view: Optional[GraphView],
        contract_id: str,
        edge_attributes: Optional[np.ndarray] = None,
        **_: Any,
    ) -> TaskBatch:
        labels = np.asarray(labels, dtype=int)
        return TaskBatch(
            identifiers=np.asarray([f"edge:{value}" for value in edge_ids]),
            features=np.asarray(edge_features),
            labels=labels,
            label_mask=labels != self.unknown_label,
            train_mask=np.asarray(train_mask, dtype=bool),
            validation_mask=np.asarray(validation_mask, dtype=bool),
            test_mask=np.asarray(test_mask, dtype=bool),
            timestamps=np.asarray(timestamps),
            graph_view=graph_view,
            edge_attributes=(
                np.asarray(edge_attributes) if edge_attributes is not None else None
            ),
            prediction_unit=self.prediction_unit,
            contract_id=contract_id,
        )

    def construct_graph_view(self, **kwargs: Any) -> GraphView:
        return GraphView(**kwargs)
