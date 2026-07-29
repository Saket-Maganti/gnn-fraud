"""Typed coordinates for the six-axis deployment contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def validate_identifier(value: str, field_name: str) -> str:
    """Validate a stable machine identifier rather than accepting free prose."""
    if not _TOKEN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_TOKEN.pattern!r}; received {value!r}"
        )
    return value


class _Axis(str, Enum):
    def __str__(self) -> str:
        return self.value


class TimeAxis(_Axis):
    UNKNOWN = "unknown"
    STATIC_SNAPSHOT = "static_snapshot"
    CHRONOLOGICAL_HOLDOUT = "chronological_holdout"
    ROLLING = "rolling"
    EXPANDING = "expanding"
    EARLY_TO_LATE = "early_to_late"
    EVENT_STREAM = "event_stream"
    CUSTOM_BOUNDED_INTERVAL = "custom_bounded_interval"


class VisibilityAxis(_Axis):
    UNKNOWN = "unknown"
    TRANSDUCTIVE_STRUCTURE = "transductive_structure"
    STRICT_INDUCTIVE = "strict_inductive"
    ISOLATED_INDUCTIVE = "isolated_inductive"
    HISTORICAL_ONLY = "historical_only"
    TEST_TIME_GRAPH_AVAILABLE = "test_time_graph_available"
    LABEL_FREE_TARGET_COVARIATES = "label_free_target_covariates"
    MISSING_GRAPH = "missing_graph"


class ConstructionAxis(_Axis):
    UNKNOWN = "unknown"
    FULL_GRAPH = "full_graph"
    RECENT_WINDOW = "recent_window"
    DEGREE_CAPPED = "degree_capped"
    DEGREE_ONLY = "degree_only"
    NO_EDGE_FEATURES = "no_edge_features"
    SHUFFLED_EDGE_FEATURES = "shuffled_edge_features"
    ORIGINAL_EDGE_ATTRIBUTES = "original_edge_attributes"
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    TASK_SPECIFIC_CUSTOM_TRANSFORM = "task_specific_custom_transform"
    NO_GRAPH = "no_graph"


class SelectionAxis(_Axis):
    UNKNOWN = "unknown"
    VALIDATION_ONLY = "validation_only"
    ROLLING_VALIDATION = "rolling_validation"
    SOURCE_CONTRACT_CV = "source_contract_cv"
    LEAVE_ONE_CONTRACT_OUT = "leave_one_contract_out"
    NO_TARGET_ACCESS = "no_target_access"
    UNLABELLED_TARGET_ADAPTATION = "unlabelled_target_adaptation"
    FEW_LABEL_ADAPTATION = "few_label_adaptation"


class BudgetAxis(_Axis):
    UNKNOWN = "unknown"
    UNCONSTRAINED_RANKING = "unconstrained_ranking"
    FRACTIONAL_REVIEW_CAPACITY = "fractional_review_capacity"
    FIXED_K = "fixed_k"
    COST_MATRIX = "cost_matrix"
    ABSTENTION_CAPACITY = "abstention_capacity"
    LATENCY_BUDGET = "latency_budget"


class ResourceAxis(_Axis):
    UNKNOWN = "unknown"
    CPU = "cpu"
    SINGLE_T4 = "single_t4"
    DUAL_T4 = "dual_t4"
    MEMORY_CAP = "memory_cap"
    LATENCY_CAP = "latency_cap"
    EXPERT_UNAVAILABLE = "expert_unavailable"
    CUSTOM_DEVICE_ENVELOPE = "custom_device_envelope"


class ContractRole(_Axis):
    SOURCE = "source"
    TARGET = "target"
    DIAGNOSTIC = "diagnostic"


class AccessRegime(_Axis):
    DG_NO_TARGET = "DG_NO_TARGET"
    TTA_UNLABELLED_TARGET = "TTA_UNLABELLED_TARGET"
    FEW_LABEL_TARGET = "FEW_LABEL_TARGET"


@dataclass(frozen=True)
class TimeSpec:
    mode: TimeAxis
    start: Optional[float] = None
    end: Optional[float] = None
    window: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("time start and end must be provided together")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("time start cannot exceed end")
        if self.mode is TimeAxis.CUSTOM_BOUNDED_INTERVAL and self.start is None:
            raise ValueError("custom bounded interval requires start and end")
        if self.window is not None and self.window <= 0:
            raise ValueError("time window must be positive")


@dataclass(frozen=True)
class ConstructionSpec:
    mode: ConstructionAxis
    recent_window: Optional[int] = None
    degree_cap: Optional[int] = None
    custom_transform_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.recent_window is not None and self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.degree_cap is not None and self.degree_cap <= 0:
            raise ValueError("degree_cap must be positive")
        if self.mode is ConstructionAxis.RECENT_WINDOW and self.recent_window is None:
            raise ValueError("recent-window construction requires recent_window")
        if self.mode is ConstructionAxis.DEGREE_CAPPED and self.degree_cap is None:
            raise ValueError("degree-capped construction requires degree_cap")
        if self.mode is ConstructionAxis.TASK_SPECIFIC_CUSTOM_TRANSFORM:
            if self.custom_transform_id is None:
                raise ValueError("custom construction requires custom_transform_id")
            validate_identifier(self.custom_transform_id, "custom_transform_id")


@dataclass(frozen=True)
class BudgetSpec:
    mode: BudgetAxis
    value: Optional[float] = None
    cost_matrix: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if self.mode is BudgetAxis.FRACTIONAL_REVIEW_CAPACITY:
            if self.value is None or not 0 < self.value <= 1:
                raise ValueError("fractional review capacity must be in (0, 1]")
        elif self.mode in {
            BudgetAxis.FIXED_K,
            BudgetAxis.ABSTENTION_CAPACITY,
            BudgetAxis.LATENCY_BUDGET,
        }:
            if self.value is None or self.value < 0:
                raise ValueError(f"{self.mode.value} requires a non-negative value")
        elif self.mode is BudgetAxis.COST_MATRIX:
            if self.cost_matrix is None:
                raise ValueError("cost-matrix budget requires a 2x2 cost matrix")
        elif self.value is not None and self.value < 0:
            raise ValueError("budget value cannot be negative")


@dataclass(frozen=True)
class ResourceSpec:
    mode: ResourceAxis
    memory_gb: Optional[float] = None
    latency_ms: Optional[float] = None
    unavailable_experts: Tuple[str, ...] = ()
    custom_envelope_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.mode is ResourceAxis.MEMORY_CAP and self.memory_gb is None:
            raise ValueError("memory-cap resource requires memory_gb")
        if self.mode is ResourceAxis.LATENCY_CAP and self.latency_ms is None:
            raise ValueError("latency-cap resource requires latency_ms")
        if self.mode is ResourceAxis.EXPERT_UNAVAILABLE and not self.unavailable_experts:
            raise ValueError("expert-unavailable resource requires at least one expert")
        for expert in self.unavailable_experts:
            validate_identifier(expert, "unavailable_expert")
        if self.mode is ResourceAxis.CUSTOM_DEVICE_ENVELOPE:
            if self.custom_envelope_id is None:
                raise ValueError("custom resource requires custom_envelope_id")
            validate_identifier(self.custom_envelope_id, "custom_envelope_id")
