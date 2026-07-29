"""Machine-readable schema fragments for deployment contracts."""

from __future__ import annotations

from coregraph.contracts.axes import (
    BudgetAxis,
    ConstructionAxis,
    ResourceAxis,
    SelectionAxis,
    TimeAxis,
    VisibilityAxis,
)


def deployment_contract_json_schema() -> dict[str, object]:
    """Return a compact JSON Schema used by config/notebook validators."""

    enum = lambda cls: [member.value for member in cls]  # noqa: E731
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DeploymentContractV2",
        "type": "object",
        "required": [
            "environment_id",
            "role",
            "time",
            "visibility",
            "construction",
            "selection",
            "budget",
            "resource",
        ],
        "properties": {
            "schema_version": {"const": 2},
            "environment_id": {"type": "string", "pattern": "^[a-z][a-z0-9_.-]{0,63}$"},
            "time": {
                "type": "object",
                "properties": {"mode": {"enum": enum(TimeAxis)}},
                "required": ["mode"],
            },
            "visibility": {"enum": enum(VisibilityAxis)},
            "construction": {
                "type": "object",
                "properties": {"mode": {"enum": enum(ConstructionAxis)}},
                "required": ["mode"],
            },
            "selection": {"enum": enum(SelectionAxis)},
            "budget": {
                "type": "object",
                "properties": {"mode": {"enum": enum(BudgetAxis)}},
                "required": ["mode"],
            },
            "resource": {
                "type": "object",
                "properties": {"mode": {"enum": enum(ResourceAxis)}},
                "required": ["mode"],
            },
        },
        "additionalProperties": False,
    }
