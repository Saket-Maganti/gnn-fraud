from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from coregraph.contracts.axes import (
    ConstructionAxis,
    ConstructionSpec,
    DeviceClass,
    EdgeFeaturePolicy,
    ResourceSpec,
    VisibilityAxis,
    VisibilitySpec,
)
from coregraph.data.graph_views import GraphView, ViewRole
from coregraph.experts.base import (
    AvailabilityReason,
    Expert,
    OfficialStatus,
    ResourceRequirements,
)
from coregraph.tasks.base import PredictionUnit, TaskBatch, TaskType


@dataclass
class FixtureExpert(Expert):
    expert_id: str = "fixture_expert"
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = (TaskType.NODE_CLASSIFICATION,)
    requirements: ResourceRequirements = ResourceRequirements(
        min_memory_gb=4.0,
        expected_latency_ms=20.0,
        device_classes=("cpu",),
        requires_graph=True,
        requires_edge_features=True,
        max_nodes_full_graph=3,
        max_edges_full_graph=1,
    )

    def fit(self, batch: TaskBatch) -> "FixtureExpert":
        return self

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        return np.zeros(len(batch.identifiers))

    def resource_requirements(self) -> ResourceRequirements:
        return self.requirements


def _batch(contract, *, graph: bool = True, edge_features: bool = True) -> TaskBatch:
    view = None
    if graph:
        view = GraphView(
            visible_node_ids=np.arange(4),
            edge_index=np.asarray([[0, 1], [1, 2]]),
            directed=True,
            edge_attributes=np.ones((2, 1)) if edge_features else None,
            edge_timestamps=np.asarray([0.0, 1.0]),
            time_cutoff=1.0,
            source_mask=np.asarray([True, True, False, False]),
            target_mask=np.asarray([False, False, True, True]),
            construction=contract.construction,
            contract=contract,
            provenance=(),
            role=ViewRole.TARGET,
        )
    return TaskBatch(
        identifiers=np.asarray(["node:0", "node:1", "node:2", "node:3"]),
        features=np.zeros((4, 2)),
        labels=np.asarray([1, 2, 1, 2]),
        label_mask=np.ones(4, dtype=bool),
        train_mask=np.asarray([True, True, False, False]),
        validation_mask=np.asarray([False, False, True, False]),
        test_mask=np.asarray([False, False, False, True]),
        timestamps=np.arange(4),
        graph_view=view,
        edge_attributes=None,
        prediction_unit=PredictionUnit.NODE,
        contract_id=contract.contract_id,
    )


@pytest.mark.parametrize(
    ("mutate_contract", "mutate_expert", "batch_options", "expected"),
    [
        (
            lambda contract: replace(
                contract,
                resource=replace(
                    contract.resource,
                    device_class=DeviceClass.UNKNOWN,
                ),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.DEVICE_UNDECLARED,
        ),
        (
            lambda contract: replace(
                contract,
                resource=replace(
                    contract.resource,
                    unavailable_experts=("fixture_expert",),
                ),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.EXPLICITLY_UNAVAILABLE,
        ),
        (
            lambda contract: replace(
                contract,
                resource=replace(
                    contract.resource,
                    device_class=DeviceClass.SINGLE_T4,
                ),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.DEVICE_INCOMPATIBLE,
        ),
        (
            lambda contract: replace(
                contract,
                resource=replace(contract.resource, memory_cap_gb=2.0),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.MEMORY_CAP_EXCEEDED,
        ),
        (
            lambda contract: replace(
                contract,
                resource=replace(contract.resource, latency_cap_ms=10.0),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.LATENCY_CAP_EXCEEDED,
        ),
        (
            lambda contract: replace(
                contract,
                visibility=VisibilitySpec.from_v2(
                    VisibilityAxis.ISOLATED_INDUCTIVE
                ),
                construction=ConstructionSpec(ConstructionAxis.NO_GRAPH),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.GRAPH_CONTRACT_UNAVAILABLE,
        ),
        (
            lambda contract: contract,
            lambda expert: replace(
                expert,
                supported_tasks=(TaskType.EDGE_CLASSIFICATION,),
            ),
            {},
            AvailabilityReason.TASK_UNSUPPORTED,
        ),
        (
            lambda contract: contract,
            lambda expert: expert,
            {"graph": False},
            AvailabilityReason.GRAPH_DATA_MISSING,
        ),
        (
            lambda contract: replace(
                contract,
                construction=replace(
                    contract.construction,
                    edge_feature_policy=EdgeFeaturePolicy.DROP,
                ),
            ),
            lambda expert: expert,
            {},
            AvailabilityReason.EDGE_FEATURES_CONTRACT_UNAVAILABLE,
        ),
        (
            lambda contract: contract,
            lambda expert: expert,
            {"edge_features": False},
            AvailabilityReason.EDGE_FEATURES_DATA_MISSING,
        ),
        (
            lambda contract: contract,
            lambda expert: expert,
            {},
            AvailabilityReason.FULL_GRAPH_NODE_GUARD,
        ),
        (
            lambda contract: contract,
            lambda expert: replace(
                expert,
                requirements=replace(
                    expert.requirements,
                    max_nodes_full_graph=10,
                ),
            ),
            {},
            AvailabilityReason.FULL_GRAPH_EDGE_GUARD,
        ),
        (
            lambda contract: contract,
            lambda expert: replace(
                expert,
                official_status=OfficialStatus.UNAVAILABLE_LICENSE,
            ),
            {},
            AvailabilityReason.LICENSE_UNAVAILABLE,
        ),
        (
            lambda contract: contract,
            lambda expert: replace(
                expert,
                official_status=OfficialStatus.PENDING_INTEGRATION,
            ),
            {},
            AvailabilityReason.INTEGRATION_PENDING,
        ),
        (
            lambda contract: contract,
            lambda expert: replace(
                expert,
                official_status=OfficialStatus.RESOURCE_BLOCKED,
            ),
            {},
            AvailabilityReason.INTEGRATION_RESOURCE_BLOCKED,
        ),
    ],
)
def test_every_availability_state_is_enforced(
    contract_factory,
    mutate_contract,
    mutate_expert,
    batch_options,
    expected,
) -> None:
    contract = mutate_contract(
        contract_factory(
            resource=ResourceSpec(
                device_class=DeviceClass.CPU,
                memory_cap_gb=8.0,
                latency_cap_ms=100.0,
            )
        )
    )
    expert = mutate_expert(FixtureExpert())
    state = expert.availability(_batch(contract, **batch_options), contract)
    assert not state.available
    assert expected in state.reason_codes


def test_availability_combines_reasons_and_allows_a_feasible_expert(
    contract_factory,
) -> None:
    contract = contract_factory(
        resource=ResourceSpec(
            device_class=DeviceClass.SINGLE_T4,
            memory_cap_gb=1.0,
            latency_cap_ms=1.0,
            unavailable_experts=("fixture_expert",),
        )
    )
    blocked = FixtureExpert().availability(
        _batch(contract, graph=False),
        contract,
    )
    assert {
        AvailabilityReason.EXPLICITLY_UNAVAILABLE,
        AvailabilityReason.DEVICE_INCOMPATIBLE,
        AvailabilityReason.MEMORY_CAP_EXCEEDED,
        AvailabilityReason.LATENCY_CAP_EXCEEDED,
        AvailabilityReason.GRAPH_DATA_MISSING,
    } <= set(blocked.reason_codes)

    feasible_contract = contract_factory(
        resource=ResourceSpec(
            device_class=DeviceClass.CPU,
            memory_cap_gb=8.0,
            latency_cap_ms=100.0,
        )
    )
    expert = replace(
        FixtureExpert(),
        requirements=replace(
            FixtureExpert().requirements,
            max_nodes_full_graph=10,
            max_edges_full_graph=10,
        ),
    )
    feasible = expert.availability(_batch(feasible_contract), feasible_contract)
    assert feasible.available
    assert feasible.reason_codes == (AvailabilityReason.AVAILABLE,)
