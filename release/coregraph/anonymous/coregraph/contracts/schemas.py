"""Machine-readable schema fragments for deployment contracts."""

from __future__ import annotations

from coregraph.contracts.axes import (
    DeviceClass,
    EdgeFeaturePolicy,
    EdgeVisibility,
    HistoryPolicy,
    MeasurementStatus,
    NodeVisibility,
    Orientation,
    ReviewMode,
    SelectionAxis,
    TimeAxis,
    TopologyTransform,
)


def deployment_contract_json_schema() -> dict[str, object]:
    """Return the compositional DeploymentContract V3 JSON Schema."""

    enum = lambda cls: [member.value for member in cls]  # noqa: E731
    identifier = {"type": "string", "pattern": "^[a-z][a-z0-9_.-]{0,63}$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DeploymentContractV3",
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
            "schema_version": {"const": 3},
            "environment_id": identifier,
            "dataset_id": identifier,
            "task_id": identifier,
            "role": {"enum": ["source", "target", "diagnostic"]},
            "access_regime": {
                "enum": [
                    "DG_NO_TARGET",
                    "TTA_UNLABELLED_TARGET",
                    "FEW_LABEL_TARGET",
                ]
            },
            "time": {
                "type": "object",
                "properties": {
                    "mode": {"enum": enum(TimeAxis)},
                    "start": {"type": ["number", "null"]},
                    "end": {"type": ["number", "null"]},
                    "window": {"type": ["integer", "null"], "minimum": 1},
                },
                "required": ["mode", "start", "end", "window"],
                "additionalProperties": False,
            },
            "visibility": {
                "type": "object",
                "properties": {
                    "node_visibility": {"enum": enum(NodeVisibility)},
                    "edge_visibility": {"enum": enum(EdgeVisibility)},
                    "target_node_availability": {"type": "boolean"},
                    "target_edge_availability": {"type": "boolean"},
                    "label_free_target_covariates": {"type": "boolean"},
                    "test_time_graph_access": {"type": "boolean"},
                    "historical_only": {"type": "boolean"},
                },
                "required": [
                    "node_visibility",
                    "edge_visibility",
                    "target_node_availability",
                    "target_edge_availability",
                    "label_free_target_covariates",
                    "test_time_graph_access",
                    "historical_only",
                ],
                "additionalProperties": False,
            },
            "construction": {
                "type": "object",
                "properties": {
                    "history_policy": {"enum": enum(HistoryPolicy)},
                    "recent_window": {"type": ["integer", "null"], "minimum": 1},
                    "degree_cap": {"type": ["integer", "null"], "minimum": 1},
                    "orientation": {"enum": enum(Orientation)},
                    "edge_feature_policy": {"enum": enum(EdgeFeaturePolicy)},
                    "topology_transform": {"enum": enum(TopologyTransform)},
                    "custom_transform_identifier": {
                        "type": ["string", "null"],
                    },
                },
                "required": [
                    "history_policy",
                    "recent_window",
                    "degree_cap",
                    "orientation",
                    "edge_feature_policy",
                    "topology_transform",
                    "custom_transform_identifier",
                ],
                "additionalProperties": False,
            },
            "selection": {"enum": enum(SelectionAxis)},
            "budget": {
                "type": "object",
                "properties": {
                    "review_mode": {"enum": enum(ReviewMode)},
                    "review_fraction": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "fixed_k": {"type": ["integer", "null"], "minimum": 0},
                    "cost_matrix": {
                        "type": ["array", "null"],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "abstention_capacity": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "latency_allowance_ms": {
                        "type": ["number", "null"],
                        "minimum": 0,
                    },
                },
                "required": [
                    "review_mode",
                    "review_fraction",
                    "fixed_k",
                    "cost_matrix",
                    "abstention_capacity",
                    "latency_allowance_ms",
                ],
                "additionalProperties": False,
            },
            "resource": {
                "type": "object",
                "properties": {
                    "device_class": {"enum": enum(DeviceClass)},
                    "memory_cap_gb": {
                        "type": ["number", "null"],
                        "exclusiveMinimum": 0,
                    },
                    "latency_cap_ms": {
                        "type": ["number", "null"],
                        "minimum": 0,
                    },
                    "unavailable_experts": {
                        "type": "array",
                        "items": identifier,
                        "uniqueItems": True,
                    },
                    "custom_envelope_id": {"type": ["string", "null"]},
                    "measurement_status": {"enum": enum(MeasurementStatus)},
                },
                "required": [
                    "device_class",
                    "memory_cap_gb",
                    "latency_cap_ms",
                    "unavailable_experts",
                    "custom_envelope_id",
                    "measurement_status",
                ],
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
