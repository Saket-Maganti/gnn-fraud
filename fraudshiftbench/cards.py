"""Dataset and metric cards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetCard:
    dataset_name: str
    variant: str
    node_or_edge_task: str
    temporal_unit: str
    raw_source_status: str
    label_semantics: str
    graph_construction: str
    scale: str
    known_limitations: list[str] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetCard":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MetricCard:
    metric_name: str
    decision_role: str
    higher_is_better: bool
    thresholded_or_ranked: str
    review_budget_relevance: str
    failure_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricCard":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_metric_cards() -> list[MetricCard]:
    return [
        MetricCard("AUPRC", "ranking quality under class imbalance", True, "ranked", "high", ["base-rate sensitive"]),
        MetricCard("AUROC", "global rank separation", True, "ranked", "medium", ["can overstate utility under rare positives"]),
        MetricCard("Precision@K", "review queue purity at fixed budget", True, "ranked", "high", ["depends on K and deployment capacity"]),
        MetricCard("Recall@K", "fraud capture at fixed budget", True, "ranked", "high", ["depends on positive base rate"]),
        MetricCard("F1", "thresholded precision/recall balance", True, "thresholded", "medium", ["threshold transfer may fail under shift"]),
        MetricCard("ECE", "calibration error", False, "thresholded", "medium", ["binning choice matters"]),
    ]
