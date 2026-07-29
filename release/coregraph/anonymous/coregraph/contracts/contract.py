"""Immutable six-axis deployment contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

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
    validate_identifier,
)
from coregraph.contracts.serialization import canonical_json, stable_sha256, to_primitive


@dataclass(frozen=True)
class DeploymentContract:
    """A validated, hash-stable deployment environment.

    Descriptive prose is deliberately kept out of the six core coordinates.
    Dataset/task identifiers are validated machine tokens and notes belong in
    evidence metadata rather than the contract hash.
    """

    environment_id: str
    role: ContractRole
    time: TimeSpec
    visibility: VisibilityAxis
    construction: ConstructionSpec
    selection: SelectionAxis
    budget: BudgetSpec
    resource: ResourceSpec
    access_regime: AccessRegime = AccessRegime.DG_NO_TARGET
    dataset_id: str = "unknown"
    task_id: str = "unknown"
    schema_version: int = 2

    def __post_init__(self) -> None:
        validate_identifier(self.environment_id, "environment_id")
        validate_identifier(self.dataset_id, "dataset_id")
        validate_identifier(self.task_id, "task_id")
        if self.schema_version != 2:
            raise ValueError("DeploymentContract schema_version must be 2")
        self._validate_combinations()

    def _validate_combinations(self) -> None:
        missing_graph = self.visibility is VisibilityAxis.MISSING_GRAPH
        no_graph = self.construction.mode is ConstructionAxis.NO_GRAPH
        if missing_graph != no_graph:
            raise ValueError("missing-graph visibility and no-graph construction must occur together")
        if (
            self.visibility is VisibilityAxis.ISOLATED_INDUCTIVE
            and self.construction.mode not in {
                ConstructionAxis.NO_GRAPH,
                ConstructionAxis.DEGREE_ONLY,
                ConstructionAxis.NO_EDGE_FEATURES,
                ConstructionAxis.UNKNOWN,
            }
        ):
            raise ValueError("isolated-inductive visibility cannot expose graph edges")
        if (
            self.selection is SelectionAxis.NO_TARGET_ACCESS
            and self.access_regime is not AccessRegime.DG_NO_TARGET
        ):
            raise ValueError("no-target selection requires DG_NO_TARGET access regime")
        if (
            self.selection is SelectionAxis.UNLABELLED_TARGET_ADAPTATION
            and self.access_regime is not AccessRegime.TTA_UNLABELLED_TARGET
        ):
            raise ValueError("unlabelled adaptation requires TTA_UNLABELLED_TARGET")
        if (
            self.selection is SelectionAxis.FEW_LABEL_ADAPTATION
            and self.access_regime is not AccessRegime.FEW_LABEL_TARGET
        ):
            raise ValueError("few-label selection requires FEW_LABEL_TARGET")
        if (
            self.resource.mode is ResourceAxis.LATENCY_CAP
            and self.budget.mode is BudgetAxis.LATENCY_BUDGET
            and self.resource.latency_ms is not None
            and self.budget.value is not None
            and self.budget.value > self.resource.latency_ms
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
    def stable_hash(self) -> str:
        return stable_sha256(self)

    @property
    def contract_id(self) -> str:
        return f"{self.environment_id}:{self.stable_hash[:16]}"

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

    def as_role(self, role: ContractRole, *, environment_id: str | None = None) -> "DeploymentContract":
        return replace(
            self,
            role=role,
            environment_id=environment_id or self.environment_id,
        )

    def human_card(self) -> str:
        rows = [
            ("Environment", self.environment_id),
            ("Role", self.role.value),
            ("Time", self.time.mode.value),
            ("Visibility", self.visibility.value),
            ("Construction", self.construction.mode.value),
            ("Selection", self.selection.value),
            ("Budget", self.budget.mode.value),
            ("Resource", self.resource.mode.value),
            ("Access", self.access_regime.value),
            ("Dataset/task", f"{self.dataset_id}/{self.task_id}"),
            ("Stable hash", self.stable_hash),
        ]
        return "\n".join(f"{name}: {value}" for name, value in rows)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentContract":
        time = payload["time"]
        construction = payload["construction"]
        budget = payload["budget"]
        resource = payload["resource"]
        cost_matrix = budget.get("cost_matrix")
        return cls(
            environment_id=str(payload["environment_id"]),
            role=ContractRole(payload["role"]),
            time=TimeSpec(
                mode=TimeAxis(time["mode"]),
                start=time.get("start"),
                end=time.get("end"),
                window=time.get("window"),
            ),
            visibility=VisibilityAxis(payload["visibility"]),
            construction=ConstructionSpec(
                mode=ConstructionAxis(construction["mode"]),
                recent_window=construction.get("recent_window"),
                degree_cap=construction.get("degree_cap"),
                custom_transform_id=construction.get("custom_transform_id"),
            ),
            selection=SelectionAxis(payload["selection"]),
            budget=BudgetSpec(
                mode=BudgetAxis(budget["mode"]),
                value=budget.get("value"),
                cost_matrix=(
                    tuple(tuple(float(v) for v in row) for row in cost_matrix)  # type: ignore[arg-type]
                    if cost_matrix is not None
                    else None
                ),
            ),
            resource=ResourceSpec(
                mode=ResourceAxis(resource["mode"]),
                memory_gb=resource.get("memory_gb"),
                latency_ms=resource.get("latency_ms"),
                unavailable_experts=tuple(resource.get("unavailable_experts", ())),
                custom_envelope_id=resource.get("custom_envelope_id"),
            ),
            access_regime=AccessRegime(payload.get("access_regime", "DG_NO_TARGET")),
            dataset_id=str(payload.get("dataset_id", "unknown")),
            task_id=str(payload.get("task_id", "unknown")),
            schema_version=int(payload.get("schema_version", 2)),
        )

    @classmethod
    def from_json(cls, value: str | Path) -> "DeploymentContract":
        path = Path(value) if not isinstance(value, Path) else value
        if isinstance(value, Path) or (isinstance(value, str) and value.lstrip()[:1] not in "[{"):
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(str(value))
        return cls.from_dict(payload)

    @classmethod
    def from_yaml(cls, value: str | Path) -> "DeploymentContract":
        path = Path(value) if not isinstance(value, Path) else value
        if isinstance(value, Path) or (isinstance(value, str) and "\n" not in value and path.exists()):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(str(value))
        if not isinstance(payload, Mapping):
            raise ValueError("contract YAML must decode to a mapping")
        return cls.from_dict(payload)
