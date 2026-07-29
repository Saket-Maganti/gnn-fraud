"""DGraphFin V2 adapter preserving event semantics and causal timing."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from coregraph.contracts.contract import DeploymentContract
from coregraph.data.contract_dataset import ContractDataset, DatasetManifest
from coregraph.data.elliptic_v2 import _scale_train_only
from coregraph.data.graph_views import build_temporal_graph_views
from coregraph.data.temporal_index import first_incident_timestamp, quantile_buckets
from coregraph.tasks.edge_task import EdgeTaskAdapter
from coregraph.tasks.node_task import NodeTaskAdapter


class DGraphTiming(str, Enum):
    FIRST_INCIDENT_EVENT = "first_incident_event"
    FIRST_LABELLED_OBSERVATION = "first_labelled_observation"
    EVENT_LEVEL_EDGE_TASK = "event_level_edge_task"
    HISTORICAL_ROLLING_AGGREGATION = "historical_rolling_aggregation"


def remap_dgraphfin_labels(raw_labels: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_labels, dtype=int)
    labels = np.zeros_like(raw)
    labels[raw == 1] = 1
    labels[raw == 0] = 2
    return labels


def causal_node_time(
    *,
    num_nodes: int,
    edge_index: np.ndarray,
    edge_timestamps: np.ndarray,
    timing: DGraphTiming,
    labelled_observation_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    if timing in {
        DGraphTiming.FIRST_INCIDENT_EVENT,
        DGraphTiming.HISTORICAL_ROLLING_AGGREGATION,
    }:
        return first_incident_timestamp(num_nodes, edge_index, edge_timestamps)
    if timing is DGraphTiming.FIRST_LABELLED_OBSERVATION:
        if labelled_observation_time is None:
            raise ValueError(
                "first-labelled-observation timing requires provider-semantic "
                "labelled_observation_time"
            )
        observed = np.asarray(labelled_observation_time)
        if len(observed) != num_nodes:
            raise ValueError("labelled observation times must align to nodes")
        return observed
    raise ValueError("event-level edge task does not define a node deployment time")


class DGraphFinV2Adapter:
    dataset_id = "dgraphfin"

    @classmethod
    def node_from_arrays(
        cls,
        *,
        features: np.ndarray,
        raw_labels: np.ndarray,
        edge_index: np.ndarray,
        edge_timestamps: np.ndarray,
        edge_types: np.ndarray,
        contract: DeploymentContract,
        timing: DGraphTiming = DGraphTiming.FIRST_INCIDENT_EVENT,
        labelled_observation_time: Optional[np.ndarray] = None,
        n_buckets: int = 20,
        train_bucket: int = 14,
        validation_bucket: int = 16,
        raw_checksum: str = "fixture",
    ) -> ContractDataset:
        features = np.asarray(features)
        edges = np.asarray(edge_index, dtype=int)
        event_time = np.asarray(edge_timestamps)
        event_type = np.asarray(edge_types)
        if edges.shape != (2, len(event_time)) or len(event_type) != len(event_time):
            raise ValueError("DGraphFin edges types and timestamps must align")
        labels = remap_dgraphfin_labels(raw_labels)
        node_time = causal_node_time(
            num_nodes=len(features),
            edge_index=edges,
            edge_timestamps=event_time,
            timing=timing,
            labelled_observation_time=labelled_observation_time,
        )
        buckets = quantile_buckets(node_time, n_buckets)
        observed = np.isfinite(node_time)
        labeled = (labels != 0) & observed
        train_mask = labeled & (buckets <= train_bucket)
        validation_mask = labeled & (buckets > train_bucket) & (
            buckets <= validation_bucket
        )
        test_mask = labeled & (buckets > validation_bucket)
        scaled, scaler_provenance = _scale_train_only(features, train_mask)
        bucket_edges = quantile_buckets(event_time, n_buckets)
        views = build_temporal_graph_views(
            node_ids=np.arange(len(features)),
            node_timestamps=buckets,
            edge_index=edges,
            edge_timestamps=bucket_edges,
            edge_attributes=event_type[:, None],
            train_cutoff=train_bucket,
            validation_cutoff=validation_bucket,
            target_cutoff=n_buckets,
            contract=contract,
        )
        task = NodeTaskAdapter()
        batch = task.build_batch(
            node_ids=np.arange(len(features)),
            features=scaled,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            timestamps=buckets,
            graph_view=views.target,
            contract_id=contract.contract_id,
        )
        return ContractDataset(
            manifest=DatasetManifest(
                dataset_id=cls.dataset_id,
                variant=f"node_{timing.value}_v2",
                source="DGraphFin provider archive",
                licence_status="provider_terms_review_required",
                raw_checksum=raw_checksum,
                timestamp_quality="provider_edge_timestamps_causal_first_observation",
                notes="Not directly interchangeable with lifecycle-median historical results.",
            ),
            task_adapter=task,
            batch=batch,
            node_ids=np.arange(len(features)),
            node_features=scaled,
            edge_index=edges,
            edge_timestamps=event_time,
            edge_types=event_type,
            edge_attributes=event_type[:, None],
            graph_views=views,
            provenance=scaler_provenance
            + (
                ("direction", "preserved"),
                ("edge_type", "preserved"),
                ("edge_timestamp", "preserved"),
                ("unobserved_nodes_excluded", str(int((~observed).sum()))),
                ("legacy_results_interchangeable", "false"),
            ),
        )

    @classmethod
    def edge_from_arrays(
        cls,
        *,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_timestamps: np.ndarray,
        edge_types: np.ndarray,
        edge_labels: np.ndarray,
        contract: DeploymentContract,
        train_cutoff: float,
        validation_cutoff: float,
        labels_are_canonical: bool = False,
        raw_checksum: str = "fixture",
    ) -> ContractDataset:
        edges = np.asarray(edge_index, dtype=int)
        timestamps = np.asarray(edge_timestamps)
        raw_labels = np.asarray(edge_labels, dtype=int)
        if labels_are_canonical:
            if not set(np.unique(raw_labels)).issubset({0, 1, 2}):
                raise ValueError("canonical edge labels must use {0,1,2}")
            labels = raw_labels
        else:
            if not set(np.unique(raw_labels)).issubset({0, 1}):
                raise ValueError("raw edge labels must be binary")
            labels = np.where(raw_labels == 1, 1, 2)
        edge_types = np.asarray(edge_types)
        if edges.shape != (2, len(timestamps)) or len(labels) != len(timestamps):
            raise ValueError("edge task arrays must align")
        edge_features = np.concatenate(
            [
                np.asarray(node_features)[edges[0]],
                np.asarray(node_features)[edges[1]],
                edge_types.reshape(-1, 1),
            ],
            axis=1,
        )
        known = labels != 0
        train = known & (timestamps <= train_cutoff)
        validation = known & (timestamps > train_cutoff) & (
            timestamps <= validation_cutoff
        )
        test = known & (timestamps > validation_cutoff)
        scaled, scaler_provenance = _scale_train_only(edge_features, train)
        task = EdgeTaskAdapter()
        batch = task.build_batch(
            edge_ids=np.arange(len(labels)),
            edge_features=scaled,
            labels=labels,
            train_mask=train,
            validation_mask=validation,
            test_mask=test,
            timestamps=timestamps,
            graph_view=None,
            contract_id=contract.contract_id,
            edge_attributes=edge_types[:, None],
        )
        return ContractDataset(
            manifest=DatasetManifest(
                dataset_id=cls.dataset_id,
                variant="event_edge_task_v2",
                source="DGraphFin provider archive",
                licence_status="provider_terms_review_required",
                raw_checksum=raw_checksum,
                timestamp_quality="provider_event_timestamp",
            ),
            task_adapter=task,
            batch=batch,
            node_ids=np.arange(len(node_features)),
            node_features=np.asarray(node_features),
            edge_index=edges,
            edge_timestamps=timestamps,
            edge_types=edge_types,
            edge_attributes=edge_types[:, None],
            provenance=scaler_provenance
            + (("prediction_unit", "edge"), ("direction", "preserved")),
        )

    @staticmethod
    def default_raw_path() -> Path:
        root = os.environ.get("COREGRAPH_DATA_ROOT")
        if not root:
            raise FileNotFoundError(
                "Set COREGRAPH_DATA_ROOT; DGraphFin is a manual provider download "
                "and has no synthetic fallback."
            )
        return Path(root) / "dgraphfin" / "dgraphfin.npz"
