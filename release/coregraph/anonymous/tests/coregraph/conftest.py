from __future__ import annotations

import pytest

from coregraph.contracts.axes import (
    AccessRegime,
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


@pytest.fixture
def contract_factory():
    def build(
        environment_id: str = "source_a",
        *,
        role: ContractRole = ContractRole.SOURCE,
        visibility: VisibilityAxis = VisibilityAxis.STRICT_INDUCTIVE,
        construction: ConstructionSpec | None = None,
        resource: ResourceSpec | None = None,
    ) -> DeploymentContract:
        return DeploymentContract(
            environment_id=environment_id,
            role=role,
            time=TimeSpec(TimeAxis.CHRONOLOGICAL_HOLDOUT),
            visibility=visibility,
            construction=construction or ConstructionSpec(ConstructionAxis.FULL_GRAPH),
            selection=SelectionAxis.NO_TARGET_ACCESS,
            budget=BudgetSpec(BudgetAxis.FRACTIONAL_REVIEW_CAPACITY, value=0.01),
            resource=resource or ResourceSpec(ResourceAxis.CPU),
            access_regime=AccessRegime.DG_NO_TARGET,
            dataset_id="fixture",
            task_id="node_fraud",
        )

    return build
