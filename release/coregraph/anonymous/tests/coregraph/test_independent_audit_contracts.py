from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coregraph.contracts.axes import (
    AccessRegime,
    BudgetSpec,
    ConstructionSpec,
    ContractRole,
    DeviceClass,
    EdgeFeaturePolicy,
    HistoryPolicy,
    MeasurementStatus,
    Orientation,
    ResourceSpec,
    ReviewMode,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    TopologyTransform,
    VisibilitySpec,
)
from coregraph.contracts.contract import DeploymentContract
from coregraph.data.dgraphfin_v2 import DGraphFinV2Adapter, DGraphTiming, causal_node_time
from coregraph.data.graph_views import build_temporal_graph_views
from coregraph.experiments.contract_splits import (
    ContractSplit,
    SplitFamily,
    leave_one_contract_out,
)


def _contract(
    environment_id: str = "source",
    *,
    role: ContractRole = ContractRole.SOURCE,
    visibility: VisibilitySpec | None = None,
    construction: ConstructionSpec | None = None,
) -> DeploymentContract:
    return DeploymentContract(
        environment_id=environment_id,
        role=role,
        time=TimeSpec(TimeAxis.CHRONOLOGICAL_HOLDOUT),
        visibility=visibility or VisibilitySpec.strict_inductive(),
        construction=construction or ConstructionSpec(),
        selection=SelectionAxis.NO_TARGET_ACCESS,
        budget=BudgetSpec(
            review_mode=ReviewMode.FRACTION,
            review_fraction=0.01,
            cost_matrix=((0.0, 1.0), (5.0, 0.0)),
            abstention_capacity=0.1,
            latency_allowance_ms=40.0,
        ),
        resource=ResourceSpec(
            device_class=DeviceClass.DUAL_T4,
            memory_cap_gb=16.0,
            latency_cap_ms=45.0,
            unavailable_experts=("blocked_expert",),
            measurement_status=MeasurementStatus.ESTIMATED,
        ),
        access_regime=AccessRegime.DG_NO_TARGET,
        dataset_id="fixture",
        task_id="node_fraud",
    )


def test_compositional_contract_fields_and_hash_boundaries() -> None:
    construction = ConstructionSpec(
        history_policy=HistoryPolicy.RECENT_WINDOW,
        recent_window=3,
        degree_cap=5,
        orientation=Orientation.DIRECTED,
        edge_feature_policy=EdgeFeaturePolicy.DROP,
        topology_transform=TopologyTransform.DEGREE_CAPPED,
    )
    left = _contract("source_a", construction=construction)
    right = replace(
        left,
        environment_id="target_b",
        role=ContractRole.TARGET,
        dataset_id="other_dataset",
        task_id="edge_fraud",
    )
    assert left.schema_version == 3
    assert left.construction.orientation is Orientation.DIRECTED
    assert left.construction.recent_window == 3
    assert left.construction.edge_feature_policy is EdgeFeaturePolicy.DROP
    assert left.budget.cost_matrix is not None
    assert left.budget.abstention_capacity == 0.1
    assert left.resource.unavailable_experts == ("blocked_expert",)
    assert left.coordinate_hash == right.coordinate_hash
    assert left.artifact_environment_hash != right.artifact_environment_hash
    assert left.stable_hash == left.artifact_environment_hash


def test_v2_contract_migrates_one_way_to_v3() -> None:
    payload = {
        "schema_version": 2,
        "environment_id": "legacy",
        "role": "source",
        "time": {"mode": "chronological_holdout"},
        "visibility": "historical_only",
        "construction": {
            "mode": "recent_window",
            "recent_window": 4,
            "degree_cap": None,
            "custom_transform_id": None,
        },
        "selection": "no_target_access",
        "budget": {"mode": "fractional_review_capacity", "value": 0.02},
        "resource": {
            "mode": "memory_cap",
            "memory_gb": 8.0,
            "latency_ms": 20.0,
            "unavailable_experts": ["gcn"],
        },
        "access_regime": "DG_NO_TARGET",
        "dataset_id": "fixture",
        "task_id": "node_fraud",
    }
    migrated = DeploymentContract.from_dict(payload)
    assert migrated.schema_version == 3
    assert migrated.visibility.historical_only
    assert migrated.construction.history_policy is HistoryPolicy.RECENT_WINDOW
    assert migrated.construction.orientation is Orientation.PRESERVE_PROVIDER
    assert migrated.budget.review_fraction == 0.02
    assert migrated.resource.memory_cap_gb == 8.0
    assert migrated.resource.unavailable_experts == ("gcn",)
    assert DeploymentContract.from_dict(migrated.to_dict()) == migrated


@pytest.mark.parametrize(
    ("access", "selection"),
    [
        (AccessRegime.DG_NO_TARGET, SelectionAxis.NO_TARGET_ACCESS),
        (
            AccessRegime.TTA_UNLABELLED_TARGET,
            SelectionAxis.UNLABELLED_TARGET_ADAPTATION,
        ),
        (AccessRegime.FEW_LABEL_TARGET, SelectionAxis.FEW_LABEL_ADAPTATION),
    ],
)
def test_split_access_regime_is_propagated_and_validated(
    access: AccessRegime,
    selection: SelectionAxis,
) -> None:
    contracts = (_contract("source_a"), _contract("source_b"))
    split = leave_one_contract_out(contracts, 1, access_regime=access)
    assert split.target[0].access_regime is access
    assert split.target[0].selection is selection
    manifest = split.manifest()
    assert manifest["access_regime"] == access.value
    assert manifest["target_access_regimes"] == [access.value]
    assert manifest["target_selection_policies"] == [selection.value]
    assert split.target[0].contract_id not in {
        contract.contract_id for contract in split.source
    }

    wrong_access = (
        AccessRegime.TTA_UNLABELLED_TARGET
        if access is AccessRegime.DG_NO_TARGET
        else AccessRegime.DG_NO_TARGET
    )
    wrong_target = split.target[0].with_access_regime(wrong_access)
    with pytest.raises(ValueError, match="split access"):
        ContractSplit(
            split_id="mismatch",
            family=SplitFamily.LEAVE_ONE_CONTRACT_OUT,
            source=split.source,
            target=(wrong_target,),
            access_regime=access,
        )


def test_unobserved_dgraphfin_nodes_are_explicit_and_excluded() -> None:
    edges = np.asarray([[0, 2], [1, 3]])
    node_time = causal_node_time(
        num_nodes=5,
        edge_index=edges,
        edge_timestamps=np.asarray([1.0, 2.0]),
        timing=DGraphTiming.FIRST_INCIDENT_EVENT,
    )
    assert np.isnan(node_time[4])

    dataset = DGraphFinV2Adapter.node_from_arrays(
        features=np.arange(15, dtype=float).reshape(5, 3),
        raw_labels=np.asarray([0, 1, 0, 1, 1]),
        edge_index=edges,
        edge_timestamps=np.asarray([1.0, 2.0]),
        edge_types=np.asarray([9, 8]),
        contract=_contract(),
        n_buckets=4,
        train_bucket=2,
        validation_bucket=3,
    )
    assert dataset.batch.timestamps[4] == 0
    assert not (
        dataset.batch.train_mask[4]
        or dataset.batch.validation_mask[4]
        or dataset.batch.test_mask[4]
    )
    assert ("unobserved_nodes_excluded", "1") in dataset.provenance


@pytest.mark.parametrize(
    "construction",
    [
        ConstructionSpec(
            history_policy=HistoryPolicy.RECENT_WINDOW,
            recent_window=2,
            orientation=Orientation.DIRECTED,
        ),
        ConstructionSpec(
            degree_cap=1,
            orientation=Orientation.DIRECTED,
            topology_transform=TopologyTransform.DEGREE_CAPPED,
        ),
    ],
)
def test_directed_history_and_topology_policies_compose(
    construction: ConstructionSpec,
) -> None:
    bundle = build_temporal_graph_views(
        node_ids=np.arange(5),
        node_timestamps=np.arange(5, dtype=float),
        edge_index=np.asarray([[0, 0, 1, 2], [1, 2, 3, 4]]),
        edge_timestamps=np.asarray([1.0, 2.0, 3.0, 4.0]),
        edge_attributes=np.arange(4)[:, None],
        train_cutoff=1,
        validation_cutoff=3,
        target_cutoff=5,
        contract=_contract(construction=construction),
    )
    assert bundle.target.directed
    if construction.history_policy is HistoryPolicy.RECENT_WINDOW:
        assert np.all(bundle.validation.edge_timestamps > 1)
    if construction.degree_cap is not None:
        degrees = np.bincount(bundle.target.edge_index.reshape(-1), minlength=5)
        assert int(degrees.max(initial=0)) <= construction.degree_cap
