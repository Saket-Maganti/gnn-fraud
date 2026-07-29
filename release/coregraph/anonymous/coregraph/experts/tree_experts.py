"""CPU tree experts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from coregraph.experts.base import Expert, OfficialStatus, ResourceRequirements
from coregraph.tasks.base import TaskBatch, TaskType


@dataclass
class HistGBExpert(Expert):
    expert_id: str = "histgb"
    seed: int = 0
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = tuple(TaskType)
    _model: HistGradientBoostingClassifier | None = field(default=None, init=False, repr=False)

    def fit(self, batch: TaskBatch) -> "HistGBExpert":
        self._model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            random_state=self.seed,
            class_weight="balanced",
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
        return ResourceRequirements(expected_latency_ms=8.0, cost_provenance="DRY_RUN_ESTIMATE")


@dataclass
class RandomForestExpert(Expert):
    expert_id: str = "random_forest"
    seed: int = 0
    n_estimators: int = 200
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = tuple(TaskType)
    _model: RandomForestClassifier | None = field(default=None, init=False, repr=False)

    def fit(self, batch: TaskBatch) -> "RandomForestExpert":
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced_subsample",
            n_jobs=1,
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
        return ResourceRequirements(min_memory_gb=1.0, expected_latency_ms=20.0)
