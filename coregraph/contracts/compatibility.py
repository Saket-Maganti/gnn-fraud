"""One-way adapters from frozen FraudShiftBench protocol metadata."""

from __future__ import annotations

from coregraph.contracts.axes import (
    BudgetAxis,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    ResourceAxis,
    ResourceSpec,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    VisibilityAxis,
)
from coregraph.contracts.contract import DeploymentContract
from fraudshiftbench.protocols import ProtocolContract

_MAP = {
    "transductive_static": (
        TimeAxis.STATIC_SNAPSHOT,
        VisibilityAxis.TRANSDUCTIVE_STRUCTURE,
        ConstructionAxis.FULL_GRAPH,
        SelectionAxis.VALIDATION_ONLY,
    ),
    "strict_inductive_temporal": (
        TimeAxis.CHRONOLOGICAL_HOLDOUT,
        VisibilityAxis.STRICT_INDUCTIVE,
        ConstructionAxis.FULL_GRAPH,
        SelectionAxis.VALIDATION_ONLY,
    ),
    "rolling_deployment": (
        TimeAxis.ROLLING,
        VisibilityAxis.HISTORICAL_ONLY,
        ConstructionAxis.RECENT_WINDOW,
        SelectionAxis.ROLLING_VALIDATION,
    ),
    "isolated_feature_control": (
        TimeAxis.CHRONOLOGICAL_HOLDOUT,
        VisibilityAxis.ISOLATED_INDUCTIVE,
        ConstructionAxis.NO_GRAPH,
        SelectionAxis.VALIDATION_ONLY,
    ),
}


def from_protocol_contract(
    protocol: ProtocolContract,
    *,
    environment_id: str,
    role: ContractRole,
    dataset_id: str = "unknown",
    task_id: str = "node_fraud",
) -> DeploymentContract:
    """Map a known legacy protocol without changing its historical record."""

    if protocol.name not in _MAP:
        raise ValueError(
            f"legacy protocol {protocol.name!r} has no curated V2 mapping; "
            "manual construct review is required"
        )
    time, visibility, construction, selection = _MAP[protocol.name]
    return DeploymentContract(
        environment_id=environment_id,
        role=role,
        time=TimeSpec(mode=time, window=1 if time is TimeAxis.ROLLING else None),
        visibility=visibility,
        construction=ConstructionSpec(
            mode=construction,
            recent_window=1 if construction is ConstructionAxis.RECENT_WINDOW else None,
        ),
        selection=selection,
        budget=BudgetSpec(BudgetAxis.UNCONSTRAINED_RANKING),
        resource=ResourceSpec(ResourceAxis.UNKNOWN),
        dataset_id=dataset_id,
        task_id=task_id,
    )
