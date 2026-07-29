from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coregraph.contracts.axes import AccessRegime
from coregraph.data.dgraphfin_v2 import DGraphFinV2Adapter, DGraphTiming, causal_node_time
from coregraph.data.ellipticpp_v2 import validate_ellipticpp_frames
from coregraph.data.good_adapter import (
    SELECTED_GOOD_SETTINGS,
    tiny_good_fixture,
    validate_good_record,
)
from coregraph.data.graph_views import audit_view_bundle, build_temporal_graph_views
from coregraph.data.ibm_aml_adapter import IBMAMLSize, build_ibm_aml_transaction_dataset
from coregraph.data.leakage import (
    FitAccessRecord,
    audit_component_access,
    audit_identifier_features,
    audit_split_masks,
)
from coregraph.data.tfinance_v2 import TimestampQuality, require_temporal_timestamps
from coregraph.tasks.base import PredictionUnit, TaskBatch, align_prediction_rows
from coregraph.tasks.edge_task import EdgeTaskAdapter
from coregraph.tasks.node_task import NodeTaskAdapter
from coregraph.tasks.transaction_task import TransactionTaskAdapter


def test_temporal_graph_views_have_no_future_edges(contract_factory) -> None:
    nodes = np.arange(6)
    timestamps = np.arange(6, dtype=float)
    edges = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_times = np.asarray([1, 2, 3, 4, 5], dtype=float)
    bundle = build_temporal_graph_views(
        node_ids=nodes,
        node_timestamps=timestamps,
        edge_index=edges,
        edge_timestamps=edge_times,
        edge_attributes=None,
        train_cutoff=1,
        validation_cutoff=3,
        target_cutoff=5,
        contract=contract_factory(),
    )
    assert bundle.train.edge_timestamps.max(initial=-1) <= 1
    assert bundle.validation.edge_timestamps.max(initial=-1) <= 3
    assert audit_view_bundle(
        bundle,
        test_node_ids=np.asarray([4, 5]),
        train_cutoff=1,
        validation_cutoff=3,
    ) == []


def test_access_audit_blocks_target_fit() -> None:
    records = [
        FitAccessRecord("scaler", "target", ("features",), False, False),
        FitAccessRecord("router", "target", ("labels",), True, False),
    ]
    violations = audit_component_access(records, regime=AccessRegime.DG_NO_TARGET)
    assert "scaler:fit_on_target" in violations
    assert "router:target_label_access" in violations


def test_dgraph_first_incident_time_is_immune_to_future_lifecycle_events() -> None:
    edges = np.asarray([[0, 1, 0], [1, 2, 2]])
    early = causal_node_time(
        num_nodes=3,
        edge_index=edges[:, :2],
        edge_timestamps=np.asarray([1.0, 2.0]),
        timing=DGraphTiming.FIRST_INCIDENT_EVENT,
    )
    with_future = causal_node_time(
        num_nodes=3,
        edge_index=edges,
        edge_timestamps=np.asarray([1.0, 2.0, 999.0]),
        timing=DGraphTiming.FIRST_INCIDENT_EVENT,
    )
    assert np.array_equal(early, with_future)


def test_dgraph_v2_preserves_directed_typed_timed_edges(contract_factory) -> None:
    edges = np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]])
    timestamps = np.asarray([1.0, 2.0, 3.0, 4.0])
    edge_types = np.asarray([9, 8, 7, 6])
    dataset = DGraphFinV2Adapter.node_from_arrays(
        features=np.arange(12, dtype=float).reshape(4, 3),
        raw_labels=np.asarray([0, 1, 0, 1]),
        edge_index=edges,
        edge_timestamps=timestamps,
        edge_types=edge_types,
        contract=contract_factory(),
        n_buckets=4,
        train_bucket=2,
        validation_bucket=3,
    )
    assert np.array_equal(dataset.edge_index, edges)
    assert np.array_equal(dataset.edge_timestamps, timestamps)
    assert np.array_equal(dataset.edge_types, edge_types)


def test_tfinance_temporal_claim_rejects_edge_order_proxy() -> None:
    with pytest.raises(ValueError, match="inadmissible"):
        require_temporal_timestamps(
            np.arange(3),
            quality=TimestampQuality.EDGE_ORDER_PROXY,
            temporal_claim=True,
            edge_count=3,
        )


def test_good_fixture_and_selected_settings_are_explicit() -> None:
    assert len(SELECTED_GOOD_SETTINGS) == 3
    validate_good_record(tiny_good_fixture(), SELECTED_GOOD_SETTINGS[0])


def test_ellipticpp_rejects_duplicate_occurrence() -> None:
    features = pd.DataFrame({"address": ["a", "a"], "time": [1, 1]})
    classes = pd.DataFrame({"address": ["a"]})
    edges = pd.DataFrame({"source": ["a"], "target": ["a"]})
    with pytest.raises(ValueError, match="occurrences must be unique"):
        validate_ellipticpp_frames(
            features,
            classes,
            edges,
            address_column="address",
            time_column="time",
            class_address_column="address",
            edge_source_column="source",
            edge_target_column="target",
        )


def test_ibm_large_is_explicitly_resource_blocked(contract_factory) -> None:
    with pytest.raises(RuntimeError, match="RESOURCE_BLOCKED"):
        build_ibm_aml_transaction_dataset(
            transaction_ids=np.arange(2),
            account_ids=np.asarray([[0, 1], [1, 2]]),
            transaction_features=np.zeros((2, 1)),
            labels=np.asarray([0, 1]),
            timestamps=np.arange(2),
            contract=contract_factory(),
            size=IBMAMLSize.LARGE,
            train_cutoff=0,
            validation_cutoff=1,
        )


def test_masks_and_identifier_mutations_are_detected() -> None:
    assert audit_split_masks(
        train=np.asarray([True, False]),
        validation=np.asarray([True, False]),
        target=np.asarray([False, True]),
        label_known=np.asarray([True, True]),
    ) == ("train_validation_overlap",)
    assert audit_identifier_features(("amount", "transaction_id")) == (
        "identifier_feature:transaction_id",
    )


def test_task_batch_rejects_unknown_label_and_duplicates() -> None:
    args = dict(
        identifiers=np.asarray(["node:1", "node:1"]),
        features=np.zeros((2, 2)),
        labels=np.asarray([1, 0]),
        label_mask=np.asarray([True, False]),
        train_mask=np.asarray([True, False]),
        validation_mask=np.asarray([False, False]),
        test_mask=np.asarray([False, True]),
        timestamps=np.asarray([0, 1]),
        graph_view=None,
        edge_attributes=None,
        prediction_unit=PredictionUnit.NODE,
        contract_id="fixture",
    )
    with pytest.raises(ValueError, match="unique"):
        TaskBatch(**args)
    args["identifiers"] = np.asarray(["node:1", "node:2"])
    with pytest.raises(ValueError, match="unknown labels"):
        TaskBatch(**args)


def test_prediction_alignment_is_by_id_not_row_order() -> None:
    ids, scores, labels = align_prediction_rows(
        {
            "a": [
                {"node_id": "2", "score": 0.8, "y_true": 1},
                {"node_id": "1", "score": 0.2, "y_true": 0},
            ],
            "b": [
                {"node_id": "1", "score": 0.3, "y_true": 0},
                {"node_id": "2", "score": 0.7, "y_true": 1},
            ],
        },
        id_column="node_id",
    )
    assert ids.tolist() == ["1", "2"]
    assert scores["a"].tolist() == [0.2, 0.8]
    assert labels.tolist() == [0, 1]


def test_ibm_raw_normal_labels_are_not_treated_as_unknown(contract_factory) -> None:
    dataset = build_ibm_aml_transaction_dataset(
        transaction_ids=np.arange(6),
        account_ids=np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]),
        transaction_features=np.arange(12, dtype=float).reshape(6, 2),
        labels=np.asarray([0, 1, 0, 1, 0, 1]),
        timestamps=np.arange(6),
        contract=contract_factory(),
        size=IBMAMLSize.SMALL,
        train_cutoff=2,
        validation_cutoff=4,
    )
    assert set(dataset.batch.labels.tolist()) == {1, 2}
    assert dataset.batch.label_mask.all()


@pytest.mark.parametrize(
    ("adapter", "id_key", "feature_key", "prefix"),
    [
        (NodeTaskAdapter(), "node_ids", "features", "node:"),
        (EdgeTaskAdapter(), "edge_ids", "edge_features", "edge:"),
        (
            TransactionTaskAdapter(),
            "transaction_ids",
            "transaction_features",
            "transaction:",
        ),
    ],
)
def test_task_adapters_emit_typed_ids_and_budget_units(
    adapter,
    id_key: str,
    feature_key: str,
    prefix: str,
) -> None:
    kwargs = {
        id_key: np.asarray([10, 11, 12]),
        feature_key: np.zeros((3, 2)),
        "labels": np.asarray([1, 2, 1]),
        "train_mask": np.asarray([True, False, False]),
        "validation_mask": np.asarray([False, True, False]),
        "test_mask": np.asarray([False, False, True]),
        "timestamps": np.arange(3),
        "graph_view": None,
        "contract_id": "fixture",
    }
    batch = adapter.build_batch(**kwargs)
    assert all(str(identifier).startswith(prefix) for identifier in batch.identifiers)
    assert adapter.budget_count(batch, 0.5) == 2
