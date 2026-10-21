from __future__ import annotations

import json

import pytest

from coregraph.contracts.axes import (
    BudgetAxis,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    EdgeVisibility,
    ResourceAxis,
    ResourceSpec,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    VisibilityAxis,
)
from coregraph.contracts.contract import DeploymentContract
from coregraph.contracts.schemas import deployment_contract_json_schema
from coregraph.experiments.contract_splits import (
    compose_contracts,
    observed_axes_unseen_combination_split,
)


def test_contract_roundtrips_and_hash_is_stable(contract_factory) -> None:
    contract = contract_factory()
    from_json = DeploymentContract.from_json(contract.to_json())
    from_yaml = DeploymentContract.from_yaml(contract.to_yaml())
    assert from_json == contract == from_yaml
    assert from_json.stable_hash == contract.stable_hash
    assert json.loads(contract.to_json())["schema_version"] == 3
    assert contract.contract_id.endswith(contract.stable_hash[:16])


def test_axis_difference_and_projection(contract_factory) -> None:
    left = contract_factory("left")
    right = contract_factory(
        "right",
        visibility=VisibilityAxis.HISTORICAL_ONLY,
    )
    assert set(left.axis_difference(right)) == {"visibility"}
    assert left.claim_projection(("visibility",))["visibility"][
        "edge_visibility"
    ] == "historical_by_cutoff"
    with pytest.raises(ValueError, match="unknown"):
        left.claim_projection(("identity",))


def test_missing_graph_is_jointly_validated(contract_factory) -> None:
    with pytest.raises(ValueError, match="edge-free"):
        contract_factory(visibility=VisibilityAxis.MISSING_GRAPH)
    valid = contract_factory(
        visibility=VisibilityAxis.MISSING_GRAPH,
        construction=ConstructionSpec(ConstructionAxis.NO_GRAPH),
    )
    assert valid.visibility.edge_visibility is EdgeVisibility.NONE


def test_isolated_view_rejects_edge_construction(contract_factory) -> None:
    with pytest.raises(ValueError, match="edge-free"):
        contract_factory(visibility=VisibilityAxis.ISOLATED_INDUCTIVE)


def test_schema_exposes_six_axes() -> None:
    schema = deployment_contract_json_schema()
    required = set(schema["required"])
    assert {"time", "visibility", "construction", "selection", "budget", "resource"} <= required


def test_observed_axis_unseen_combination_split() -> None:
    contracts = compose_contracts(
        dataset_id="fixture",
        task_id="node",
        time_values=(
            TimeSpec(TimeAxis.STATIC_SNAPSHOT),
            TimeSpec(TimeAxis.CHRONOLOGICAL_HOLDOUT),
        ),
        visibility_values=(
            VisibilityAxis.STRICT_INDUCTIVE,
            VisibilityAxis.HISTORICAL_ONLY,
        ),
        construction_values=(ConstructionSpec(ConstructionAxis.FULL_GRAPH),),
        selection_values=(SelectionAxis.NO_TARGET_ACCESS,),
        budget_values=(BudgetSpec(BudgetAxis.UNCONSTRAINED_RANKING),),
        resource_values=(ResourceSpec(ResourceAxis.CPU),),
    )
    split = observed_axes_unseen_combination_split(contracts, 0)
    assert set(contract.contract_id for contract in split.source).isdisjoint(
        contract.contract_id for contract in split.target
    )
    assert not split.atomic_target_id_seen
