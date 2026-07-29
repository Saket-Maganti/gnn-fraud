"""Node-classification adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from coregraph.data.graph_views import GraphView
from coregraph.tasks.base import PredictionUnit, TaskAdapter, TaskBatch, TaskType


class NodeTaskAdapter(TaskAdapter):
    task_type = TaskType.NODE_CLASSIFICATION
    prediction_unit = PredictionUnit.NODE

    def build_batch(
        self,
        *,
        node_ids: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
        test_mask: np.ndarray,
        timestamps: np.ndarray,
        graph_view: Optional[GraphView],
        contract_id: str,
        **_: Any,
    ) -> TaskBatch:
        labels = np.asarray(labels, dtype=int)
        return TaskBatch(
            identifiers=np.asarray([f"node:{value}" for value in node_ids]),
            features=np.asarray(features),
            labels=labels,
            label_mask=labels != self.unknown_label,
            train_mask=np.asarray(train_mask, dtype=bool),
            validation_mask=np.asarray(validation_mask, dtype=bool),
            test_mask=np.asarray(test_mask, dtype=bool),
            timestamps=np.asarray(timestamps),
            graph_view=graph_view,
            edge_attributes=None,
            prediction_unit=self.prediction_unit,
            contract_id=contract_id,
        )

    def construct_graph_view(self, **kwargs: Any) -> GraphView:
        return GraphView(**kwargs)
