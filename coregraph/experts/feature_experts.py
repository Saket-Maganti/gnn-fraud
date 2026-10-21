"""Feature-only experts with deterministic CPU fits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from coregraph.experts.base import Expert, OfficialStatus, ResourceRequirements
from coregraph.tasks.base import TaskBatch, TaskType


@dataclass
class LogisticRegressionExpert(Expert):
    expert_id: str = "logistic_regression"
    seed: int = 0
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = tuple(TaskType)
    _model: LogisticRegression | None = field(default=None, init=False, repr=False)

    def fit(self, batch: TaskBatch) -> "LogisticRegressionExpert":
        if not batch.train_mask.any():
            raise ValueError("expert fit requires labelled training examples")
        self._model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.seed,
        )
        self._model.fit(
            batch.features[batch.train_mask],
            (batch.labels[batch.train_mask] == 1).astype(int),
        )
        return self

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit expert before prediction")
        return self._model.predict_proba(batch.features)[:, 1]

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(expected_latency_ms=2.0, cost_provenance="DRY_RUN_ESTIMATE")


@dataclass
class FeatureMLPExpert(Expert):
    expert_id: str = "feature_mlp"
    seed: int = 0
    hidden_layers: tuple[int, ...] = (64, 32)
    max_iter: int = 100
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = tuple(TaskType)
    _model: MLPClassifier | None = field(default=None, init=False, repr=False)

    def fit(self, batch: TaskBatch) -> "FeatureMLPExpert":
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
        )
        self._model.fit(
            batch.features[batch.train_mask],
            (batch.labels[batch.train_mask] == 1).astype(int),
        )
        return self

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit expert before prediction")
        return self._model.predict_proba(batch.features)[:, 1]

    def predict_embeddings(self, batch: TaskBatch) -> np.ndarray | None:
        if self._model is None:
            return None
        hidden = batch.features
        for weight, bias in zip(self._model.coefs_[:-1], self._model.intercepts_[:-1]):
            hidden = np.maximum(0, hidden @ weight + bias)
        return hidden

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(expected_latency_ms=5.0, cost_provenance="DRY_RUN_ESTIMATE")
