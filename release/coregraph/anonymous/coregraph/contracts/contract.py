"""Immutable six-coordinate deployment contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import yaml

from coregraph.contracts.axes import (
    AccessRegime,
    BudgetAxis,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    DeviceClass,
    EdgeFeaturePolicy,
    EdgeVisibility,
    HistoryPolicy,
    MeasurementStatus,
    NodeVisibility,
    Orientation,
    ResourceAxis,
    ResourceSpec,
    ReviewMode,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    TopologyTransform,
    VisibilityAxis,
    VisibilitySpec,
    validate_identifier,
)
from coregraph.contracts.serialization import canonical_json, stable_sha256, to_primitive


def _cost_matrix(
    value: object,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("cost matrix must be a nested sequence")
    nested = cast(Sequence[Sequence[float]], value)
    rows = tuple(tuple(float(item) for item in row) for row in nested)
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("cost matrix must be 2x2")
    return rows  # type: ignore[return-value]


def migrate_v2_contract_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One-way migration from the overloaded V2 representation to V3."""

    version = int(payload.get("schema_version", 2))
    if version != 2:
        raise ValueError("one-way migration accepts DeploymentContract V2 only")
    construction_payload = payload["construction"]
    budget_payload = payload["budget"]
    resource_payload = payload["resource"]
    if not isinstance(construction_payload, Mapping):
        raise ValueError("V2 construction must be a mapping")
    if not isinstance(budget_payload, Mapping):
        raise ValueError("V2 budget must be a mapping")
    if not isinstance(resource_payload, Mapping):
        raise ValueError("V2 resource must be a mapping")
    construction = ConstructionSpec(
        ConstructionAxis(construction_payload["mode"]),
        recent_window=construction_payload.get("recent_window"),
        degree_cap=construction_payload.get("degree_cap"),
        custom_transform_id=construction_payload.get("custom_transform_id"),
    )
    budget = BudgetSpec(
        BudgetAxis(budget_payload["mode"]),
        value=budget_payload.get("value"),
        cost_matrix=_cost_matrix(budget_payload.get("cost_matrix")),
    )
    resource = ResourceSpec(
        ResourceAxis(resource_payload["mode"]),
        memory_gb=resource_payload.get("memory_gb"),
        latency_ms=resource_payload.get("latency_ms"),
        unavailable_experts=tuple(resource_payload.get("unavailable_experts", ())),
        custom_envelope_id=resource_payload.get("custom_envelope_id"),
        measurement_status=MeasurementStatus.ESTIMATED,
    )
    migrated = dict(payload)
    migrated.update(
        {
            "schema_version": 3,
            "visibility": to_primitive(
                VisibilitySpec.from_v2(VisibilityAxis(payload["visibility"]))
            ),
            "construction": to_primitive(construction),
            "budget": to_primitive(budget),
            "resource": to_primitive(resource),
        }
    )
    return migrated


@dataclass(frozen=True)
class DeploymentContract:
    """A validated, hash-stable deployment environment.

    The six top-level scientific coordinates remain time, visibility,
    construction, selection, budget, and resource. V3 makes the four
    compositional coordinates structured rather than collapsing simultaneous
    properties into one overloaded mode.
    """

    environment_id: str
    role: ContractRole
    time: TimeSpec
    visibility: VisibilitySpec
    construction: ConstructionSpec
    selection: SelectionAxis
    budget: BudgetSpec
    resource: ResourceSpec
    access_regime: AccessRegime = AccessRegime.DG_NO_TARGET
    dataset_id: str = "unknown"
    task_id: str = "unknown"
    schema_version: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.visibility, VisibilityAxis):
            object.__setattr__(self, "visibility", VisibilitySpec.from_v2(self.visibility))
        validate_identifier(self.environment_id, "environment_id")
        validate_identifier(self.dataset_id, "dataset_id")
        validate_identifier(self.task_id, "task_id")
        if self.schema_version != 3:
            raise ValueError("DeploymentContract schema_version must be 3")
        self._validate_combinations()

    def _validate_combinations(self) -> None:
        no_visible_edges = self.visibility.edge_visibility is EdgeVisibility.NONE
        no_graph = self.construction.topology_transform is TopologyTransform.NO_GRAPH
        if no_graph and not no_visible_edges:
            raise ValueError("no-graph construction cannot expose graph edges")
        if no_visible_edges and self.construction.topology_transform not in {
            TopologyTransform.NO_GRAPH,
            TopologyTransform.DEGREE_ONLY,
        }:
            raise ValueError(
                "edge-free visibility requires no-graph or degree-only construction"
            )
        expected_selection = {
            AccessRegime.TTA_UNLABELLED_TARGET: (
                SelectionAxis.UNLABELLED_TARGET_ADAPTATION
            ),
            AccessRegime.FEW_LABEL_TARGET: SelectionAxis.FEW_LABEL_ADAPTATION,
        }.get(self.access_regime)
        if expected_selection is not None and self.selection is not expected_selection:
            raise ValueError(
                f"{self.access_regime.value} access requires "
                f"{expected_selection.value} selection"
            )
        if self.access_regime is AccessRegime.DG_NO_TARGET and self.selection in {
            SelectionAxis.UNLABELLED_TARGET_ADAPTATION,
            SelectionAxis.FEW_LABEL_ADAPTATION,
        }:
            raise ValueError("DG_NO_TARGET access cannot use a target-adaptation selection")
        if (
            self.resource.latency_cap_ms is not None
            and self.budget.latency_allowance_ms is not None
            and self.budget.latency_allowance_ms > self.resource.latency_cap_ms
        ):
            raise ValueError("budget latency cannot exceed the resource latency cap")

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    def to_json(self, *, pretty: bool = False) -> str:
        if pretty:
            return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        return canonical_json(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True)

    @property
    def coordinate_hash(self) -> str:
        """Hash only scientific coordinates and access semantics."""

        return stable_sha256(
            {
                "schema_version": self.schema_version,
                "time": self.time,
                "visibility": self.visibility,
                "construction": self.construction,
                "selection": self.selection,
                "budget": self.budget,
                "resource": self.resource,
                "access_regime": self.access_regime,
            }
        )

    @property
    def artifact_environment_hash(self) -> str:
        """Hash the complete artifact/environment identity."""

        return stable_sha256(self)

    @property
    def stable_hash(self) -> str:
        """Backward-compatible name for the complete artifact hash."""

        return self.artifact_environment_hash

    @property
    def contract_id(self) -> str:
        return f"{self.environment_id}:{self.artifact_environment_hash[:16]}"

    def axis_difference(self, other: "DeploymentContract") -> dict[str, tuple[Any, Any]]:
        fields = ("time", "visibility", "construction", "selection", "budget", "resource")
        return {
            name: (getattr(self, name), getattr(other, name))
            for name in fields
            if getattr(self, name) != getattr(other, name)
        }

    def claim_projection(self, axes: Iterable[str]) -> dict[str, Any]:
        allowed = {"time", "visibility", "construction", "selection", "budget", "resource"}
        requested = tuple(axes)
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"unknown claim projection axes: {unknown}")
        return {axis: to_primitive(getattr(self, axis)) for axis in requested}

    def compatible_with(self, other: "DeploymentContract", *, axes: Iterable[str] = ()) -> bool:
        selected = tuple(axes) or (
            "time",
            "visibility",
            "construction",
            "selection",
            "budget",
            "resource",
        )
        return all(getattr(self, name) == getattr(other, name) for name in selected)

    def as_role(
        self,
        role: ContractRole,
        *,
        environment_id: str | None = None,
    ) -> "DeploymentContract":
        return replace(
            self,
            role=role,
            environment_id=environment_id or self.environment_id,
        )

    def with_access_regime(self, access_regime: AccessRegime) -> "DeploymentContract":
        selection = {
            AccessRegime.DG_NO_TARGET: SelectionAxis.NO_TARGET_ACCESS,
            AccessRegime.TTA_UNLABELLED_TARGET: (
                SelectionAxis.UNLABELLED_TARGET_ADAPTATION
            ),
            AccessRegime.FEW_LABEL_TARGET: SelectionAxis.FEW_LABEL_ADAPTATION,
        }[access_regime]
        return replace(self, access_regime=access_regime, selection=selection)

    def human_card(self) -> str:
        rows = [
            ("Environment", self.environment_id),
            ("Role", self.role.value),
            ("Time", self.time.mode.value),
            ("Visibility", canonical_json(self.visibility)),
            ("Construction", canonical_json(self.construction)),
            ("Selection", self.selection.value),
            ("Budget", canonical_json(self.budget)),
            ("Resource", canonical_json(self.resource)),
            ("Access", self.access_regime.value),
            ("Dataset/task", f"{self.dataset_id}/{self.task_id}"),
            ("Coordinate hash", self.coordinate_hash),
            ("Artifact/environment hash", self.artifact_environment_hash),
        ]
        return "\n".join(f"{name}: {value}" for name, value in rows)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentContract":
        version = int(payload.get("schema_version", 2))
        if version == 2:
            payload = migrate_v2_contract_payload(payload)
        elif version != 3:
            raise ValueError(f"unsupported DeploymentContract schema version {version}")
        time = payload["time"]
        visibility = payload["visibility"]
        construction = payload["construction"]
        budget = payload["budget"]
        resource = payload["resource"]
        if not all(
            isinstance(value, Mapping)
            for value in (time, visibility, construction, budget, resource)
        ):
            raise ValueError("V3 structured coordinates must be mappings")
        return cls(
            environment_id=str(payload["environment_id"]),
            role=ContractRole(payload["role"]),
            time=TimeSpec(
                mode=TimeAxis(time["mode"]),
                start=time.get("start"),
                end=time.get("end"),
                window=time.get("window"),
            ),
            visibility=VisibilitySpec(
                node_visibility=NodeVisibility(visibility["node_visibility"]),
                edge_visibility=EdgeVisibility(visibility["edge_visibility"]),
                target_node_availability=bool(
                    visibility["target_node_availability"]
                ),
                target_edge_availability=bool(
                    visibility["target_edge_availability"]
                ),
                label_free_target_covariates=bool(
                    visibility["label_free_target_covariates"]
                ),
                test_time_graph_access=bool(visibility["test_time_graph_access"]),
                historical_only=bool(visibility["historical_only"]),
            ),
            construction=ConstructionSpec(
                history_policy=HistoryPolicy(construction["history_policy"]),
                recent_window=construction.get("recent_window"),
                degree_cap=construction.get("degree_cap"),
                orientation=Orientation(construction["orientation"]),
                edge_feature_policy=EdgeFeaturePolicy(
                    construction["edge_feature_policy"]
                ),
                topology_transform=TopologyTransform(
                    construction["topology_transform"]
                ),
                custom_transform_identifier=construction.get(
                    "custom_transform_identifier"
                ),
            ),
            selection=SelectionAxis(payload["selection"]),
            budget=BudgetSpec(
                review_mode=ReviewMode(budget["review_mode"]),
                review_fraction=budget.get("review_fraction"),
                fixed_k=budget.get("fixed_k"),
                cost_matrix=_cost_matrix(budget.get("cost_matrix")),
                abstention_capacity=budget.get("abstention_capacity"),
                latency_allowance_ms=budget.get("latency_allowance_ms"),
            ),
            resource=ResourceSpec(
                device_class=DeviceClass(resource["device_class"]),
                memory_cap_gb=resource.get("memory_cap_gb"),
                latency_cap_ms=resource.get("latency_cap_ms"),
                unavailable_experts=tuple(resource.get("unavailable_experts", ())),
                custom_envelope_id=resource.get("custom_envelope_id"),
                measurement_status=MeasurementStatus(
                    resource["measurement_status"]
                ),
            ),
            access_regime=AccessRegime(
                payload.get("access_regime", AccessRegime.DG_NO_TARGET.value)
            ),
            dataset_id=str(payload.get("dataset_id", "unknown")),
            task_id=str(payload.get("task_id", "unknown")),
            schema_version=3,
        )

    @classmethod
    def from_json(cls, value: str | Path) -> "DeploymentContract":
        path = Path(value) if not isinstance(value, Path) else value
        if isinstance(value, Path) or (
            isinstance(value, str) and value.lstrip()[:1] not in "[{"
        ):
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(str(value))
        return cls.from_dict(payload)

    @classmethod
    def from_yaml(cls, value: str | Path) -> "DeploymentContract":
        path = Path(value) if not isinstance(value, Path) else value
        if isinstance(value, Path) or (
            isinstance(value, str) and "\n" not in value and path.exists()
        ):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(str(value))
        if not isinstance(payload, Mapping):
            raise ValueError("contract YAML must decode to a mapping")
        return cls.from_dict(payload)
