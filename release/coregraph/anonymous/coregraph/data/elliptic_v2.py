"""Corrected Elliptic adapter with explicit temporal graph views."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.data.contract_dataset import ContractDataset, DatasetManifest
from coregraph.data.graph_views import build_temporal_graph_views
from coregraph.tasks.node_task import NodeTaskAdapter


def _scale_train_only(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[str, str], ...]]:
    x = np.asarray(features, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if not mask.any():
        raise ValueError("train-only scaling requires at least one training example")
    mean = x[mask].mean(axis=0)
    std = x[mask].std(axis=0)
    std[std < 1e-12] = 1.0
    return ((x - mean) / std).astype(np.float32), (
        ("scaler_fit", "train_only"),
        ("scaler_rows", str(int(mask.sum()))),
    )


class EllipticV2Adapter:
    dataset_id = "elliptic"
    official_steps = 49

    @classmethod
    def from_arrays(
        cls,
        *,
        features: np.ndarray,
        labels: np.ndarray,
        node_timestamps: np.ndarray,
        edge_index: np.ndarray,
        contract: DeploymentContract,
        edge_timestamps: Optional[np.ndarray] = None,
        edge_attributes: Optional[np.ndarray] = None,
        train_cutoff: int = 30,
        validation_cutoff: int = 34,
        target_cutoff: int = 49,
        raw_checksum: str = "fixture",
    ) -> ContractDataset:
        labels = np.asarray(labels, dtype=int)
        if not set(np.unique(labels)).issubset({0, 1, 2}):
            raise ValueError("Elliptic V2 labels must use {0 unknown,1 illicit,2 licit}")
        times = np.asarray(node_timestamps)
        if times.min(initial=1) < 1 or times.max(initial=49) > cls.official_steps:
            raise ValueError("Elliptic time steps must lie in the official range 1..49")
        node_ids = np.arange(len(features), dtype=int)
        labeled = labels != 0
        train_mask = labeled & (times <= train_cutoff)
        validation_mask = labeled & (times > train_cutoff) & (
            times <= validation_cutoff
        )
        test_mask = labeled & (times > validation_cutoff) & (times <= target_cutoff)
        scaled, scaler_provenance = _scale_train_only(features, train_mask)
        if edge_timestamps is None:
            edge_timestamps = np.maximum(
                times[np.asarray(edge_index)[0]],
                times[np.asarray(edge_index)[1]],
            )
        views = build_temporal_graph_views(
            node_ids=node_ids,
            node_timestamps=times,
            edge_index=np.asarray(edge_index),
            edge_timestamps=np.asarray(edge_timestamps),
            edge_attributes=edge_attributes,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
            target_cutoff=target_cutoff,
            contract=contract,
        )
        task = NodeTaskAdapter()
        batch = task.build_batch(
            node_ids=node_ids,
            features=scaled,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            timestamps=times,
            graph_view=views.target,
            contract_id=contract.contract_id,
        )
        return ContractDataset(
            manifest=DatasetManifest(
                dataset_id=cls.dataset_id,
                variant="official_temporal_v2",
                source="Elliptic Data Set",
                licence_status="dataset_terms_review_required",
                raw_checksum=raw_checksum,
                timestamp_quality="official_49_steps",
            ),
            task_adapter=task,
            batch=batch,
            node_ids=node_ids,
            node_features=scaled,
            edge_index=np.asarray(edge_index, dtype=int),
            edge_timestamps=np.asarray(edge_timestamps),
            edge_types=None,
            edge_attributes=(
                np.asarray(edge_attributes) if edge_attributes is not None else None
            ),
            graph_views=views,
            provenance=scaler_provenance
            + (
                ("graph_views", "train_validation_target_separate"),
                ("legacy_results_interchangeable", "prediction_schema_only"),
            ),
        )

    @classmethod
    def default_raw_root(cls) -> Path:
        root = os.environ.get("COREGRAPH_DATA_ROOT")
        if not root:
            raise FileNotFoundError(
                "Set COREGRAPH_DATA_ROOT to an external dataset directory. "
                "CoReGraph does not copy or download Elliptic data."
            )
        return Path(root) / "elliptic"
