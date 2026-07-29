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


# V2 enums are retained only as migration/API compatibility tokens. V3
# serialisation uses the structured specifications below.
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


class NodeVisibility(_Axis):
    UNKNOWN = "unknown"
    SOURCE_ONLY = "source_only"
    OBSERVED_BY_CUTOFF = "observed_by_cutoff"
    ALL_NODES = "all_nodes"
    NONE = "none"


class EdgeVisibility(_Axis):
    UNKNOWN = "unknown"
    SOURCE_ONLY = "source_only"
    HISTORICAL_BY_CUTOFF = "historical_by_cutoff"
    ALL_EDGES = "all_edges"
    NONE = "none"


class HistoryPolicy(_Axis):
    FULL_HISTORY = "full_history"
    RECENT_WINDOW = "recent_window"
    NONE = "none"


class Orientation(_Axis):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    PRESERVE_PROVIDER = "preserve_provider"


class EdgeFeaturePolicy(_Axis):
    PRESERVE = "preserve"
    DROP = "drop"
    SHUFFLE = "shuffle"


class TopologyTransform(_Axis):
    NONE = "none"
    DEGREE_CAPPED = "degree_capped"
    DEGREE_ONLY = "degree_only"
    SHUFFLED = "shuffled"
    NO_GRAPH = "no_graph"
    CUSTOM = "custom"


class ReviewMode(_Axis):
    UNCONSTRAINED_RANKING = "unconstrained_ranking"
    FRACTION = "fraction"
    FIXED_K = "fixed_k"


class DeviceClass(_Axis):
    UNKNOWN = "unknown"
    CPU = "cpu"
    SINGLE_T4 = "single_t4"
    DUAL_T4 = "dual_t4"
    CUDA = "cuda"
    MPS = "mps"
    CUSTOM = "custom"


class MeasurementStatus(_Axis):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


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
class VisibilitySpec:
    node_visibility: NodeVisibility
    edge_visibility: EdgeVisibility
    target_node_availability: bool
    target_edge_availability: bool
    label_free_target_covariates: bool
    test_time_graph_access: bool
    historical_only: bool

    @property
    def graph_available(self) -> bool:
        return self.edge_visibility not in {EdgeVisibility.NONE, EdgeVisibility.UNKNOWN}

    @classmethod
    def strict_inductive(cls) -> "VisibilitySpec":
        return cls(
            node_visibility=NodeVisibility.OBSERVED_BY_CUTOFF,
            edge_visibility=EdgeVisibility.HISTORICAL_BY_CUTOFF,
            target_node_availability=True,
            target_edge_availability=False,
            label_free_target_covariates=True,
            test_time_graph_access=False,
            historical_only=False,
        )

    @classmethod
    def from_v2(cls, value: VisibilityAxis | str) -> "VisibilitySpec":
        mode = value if isinstance(value, VisibilityAxis) else VisibilityAxis(value)
        mapping = {
            VisibilityAxis.UNKNOWN: cls(
                NodeVisibility.UNKNOWN,
                EdgeVisibility.UNKNOWN,
                False,
                False,
                False,
                False,
                False,
            ),
            VisibilityAxis.TRANSDUCTIVE_STRUCTURE: cls(
                NodeVisibility.ALL_NODES,
                EdgeVisibility.ALL_EDGES,
                True,
                True,
                True,
                True,
                False,
            ),
            VisibilityAxis.STRICT_INDUCTIVE: cls.strict_inductive(),
            VisibilityAxis.ISOLATED_INDUCTIVE: cls(
                NodeVisibility.OBSERVED_BY_CUTOFF,
                EdgeVisibility.NONE,
                True,
                False,
                True,
                False,
                True,
            ),
            VisibilityAxis.HISTORICAL_ONLY: cls(
                NodeVisibility.OBSERVED_BY_CUTOFF,
                EdgeVisibility.HISTORICAL_BY_CUTOFF,
                True,
                False,
                True,
                False,
                True,
            ),
            VisibilityAxis.TEST_TIME_GRAPH_AVAILABLE: cls(
                NodeVisibility.OBSERVED_BY_CUTOFF,
                EdgeVisibility.HISTORICAL_BY_CUTOFF,
                True,
                True,
                True,
                True,
                False,
            ),
            VisibilityAxis.LABEL_FREE_TARGET_COVARIATES: cls(
                NodeVisibility.OBSERVED_BY_CUTOFF,
                EdgeVisibility.NONE,
                True,
                False,
                True,
                False,
                False,
            ),
            VisibilityAxis.MISSING_GRAPH: cls(
                NodeVisibility.OBSERVED_BY_CUTOFF,
                EdgeVisibility.NONE,
                True,
                False,
                True,
                False,
                True,
            ),
        }
        return mapping[mode]


@dataclass(frozen=True, init=False)
class ConstructionSpec:
    history_policy: HistoryPolicy
    recent_window: Optional[int]
    degree_cap: Optional[int]
    orientation: Orientation
    edge_feature_policy: EdgeFeaturePolicy
    topology_transform: TopologyTransform
    custom_transform_identifier: Optional[str]

    def __init__(
        self,
        legacy_mode: ConstructionAxis | None = None,
        *,
        history_policy: HistoryPolicy = HistoryPolicy.FULL_HISTORY,
        recent_window: Optional[int] = None,
        degree_cap: Optional[int] = None,
        orientation: Orientation = Orientation.PRESERVE_PROVIDER,
        edge_feature_policy: EdgeFeaturePolicy = EdgeFeaturePolicy.PRESERVE,
        topology_transform: TopologyTransform = TopologyTransform.NONE,
        custom_transform_identifier: Optional[str] = None,
        custom_transform_id: Optional[str] = None,
    ) -> None:
        if legacy_mode is not None:
            (
                history_policy,
                orientation,
                edge_feature_policy,
                topology_transform,
            ) = {
                ConstructionAxis.UNKNOWN: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.FULL_GRAPH: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.RECENT_WINDOW: (
                    HistoryPolicy.RECENT_WINDOW,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.DEGREE_CAPPED: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.DEGREE_CAPPED,
                ),
                ConstructionAxis.DEGREE_ONLY: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.DROP,
                    TopologyTransform.DEGREE_ONLY,
                ),
                ConstructionAxis.NO_EDGE_FEATURES: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.DROP,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.SHUFFLED_EDGE_FEATURES: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.SHUFFLE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.ORIGINAL_EDGE_ATTRIBUTES: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.DIRECTED: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.DIRECTED,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.UNDIRECTED: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.UNDIRECTED,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.NONE,
                ),
                ConstructionAxis.TASK_SPECIFIC_CUSTOM_TRANSFORM: (
                    HistoryPolicy.FULL_HISTORY,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.PRESERVE,
                    TopologyTransform.CUSTOM,
                ),
                ConstructionAxis.NO_GRAPH: (
                    HistoryPolicy.NONE,
                    Orientation.PRESERVE_PROVIDER,
                    EdgeFeaturePolicy.DROP,
                    TopologyTransform.NO_GRAPH,
                ),
            }[ConstructionAxis(legacy_mode)]
        custom = custom_transform_identifier or custom_transform_id
        object.__setattr__(self, "history_policy", HistoryPolicy(history_policy))
        object.__setattr__(self, "recent_window", recent_window)
        object.__setattr__(self, "degree_cap", degree_cap)
        object.__setattr__(self, "orientation", Orientation(orientation))
        object.__setattr__(self, "edge_feature_policy", EdgeFeaturePolicy(edge_feature_policy))
        object.__setattr__(self, "topology_transform", TopologyTransform(topology_transform))
        object.__setattr__(self, "custom_transform_identifier", custom)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.recent_window is not None and self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.degree_cap is not None and self.degree_cap <= 0:
            raise ValueError("degree_cap must be positive")
        if self.history_policy is HistoryPolicy.RECENT_WINDOW and self.recent_window is None:
            raise ValueError("recent-window history requires recent_window")
        if (
            self.topology_transform is TopologyTransform.DEGREE_CAPPED
            and self.degree_cap is None
        ):
            raise ValueError("degree-capped topology requires degree_cap")
        if self.topology_transform is TopologyTransform.CUSTOM:
            if self.custom_transform_identifier is None:
                raise ValueError("custom construction requires custom_transform_identifier")
            validate_identifier(
                self.custom_transform_identifier,
                "custom_transform_identifier",
            )
        elif self.custom_transform_identifier is not None:
            raise ValueError("custom transform identifier requires custom topology transform")
        if (
            self.topology_transform is TopologyTransform.NO_GRAPH
            and self.history_policy is not HistoryPolicy.NONE
        ):
            raise ValueError("no-graph topology requires no history")

    @property
    def mode(self) -> ConstructionAxis:
        """Lossy V2 display adapter; never used for V3 scientific semantics."""
        if self.topology_transform is TopologyTransform.NO_GRAPH:
            return ConstructionAxis.NO_GRAPH
        if self.topology_transform is TopologyTransform.DEGREE_CAPPED:
            return ConstructionAxis.DEGREE_CAPPED
        if self.topology_transform is TopologyTransform.DEGREE_ONLY:
            return ConstructionAxis.DEGREE_ONLY
        if self.topology_transform is TopologyTransform.CUSTOM:
            return ConstructionAxis.TASK_SPECIFIC_CUSTOM_TRANSFORM
        if self.history_policy is HistoryPolicy.RECENT_WINDOW:
            return ConstructionAxis.RECENT_WINDOW
        if self.edge_feature_policy is EdgeFeaturePolicy.DROP:
            return ConstructionAxis.NO_EDGE_FEATURES
        if self.edge_feature_policy is EdgeFeaturePolicy.SHUFFLE:
            return ConstructionAxis.SHUFFLED_EDGE_FEATURES
        if self.orientation is Orientation.DIRECTED:
            return ConstructionAxis.DIRECTED
        if self.orientation is Orientation.UNDIRECTED:
            return ConstructionAxis.UNDIRECTED
        return ConstructionAxis.FULL_GRAPH


@dataclass(frozen=True, init=False)
class BudgetSpec:
    review_mode: ReviewMode
    review_fraction: Optional[float]
    fixed_k: Optional[int]
    cost_matrix: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    abstention_capacity: Optional[float]
    latency_allowance_ms: Optional[float]

    def __init__(
        self,
        legacy_mode: BudgetAxis | None = None,
        *,
        review_mode: ReviewMode = ReviewMode.UNCONSTRAINED_RANKING,
        review_fraction: Optional[float] = None,
        fixed_k: Optional[int] = None,
        cost_matrix: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        abstention_capacity: Optional[float] = None,
        latency_allowance_ms: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        if legacy_mode is not None:
            mode = BudgetAxis(legacy_mode)
            if mode is BudgetAxis.FRACTIONAL_REVIEW_CAPACITY:
                review_mode, review_fraction = ReviewMode.FRACTION, value
            elif mode is BudgetAxis.FIXED_K:
                review_mode, fixed_k = ReviewMode.FIXED_K, (
                    None if value is None else int(value)
                )
            elif mode is BudgetAxis.COST_MATRIX:
                review_mode = ReviewMode.UNCONSTRAINED_RANKING
            elif mode is BudgetAxis.ABSTENTION_CAPACITY:
                abstention_capacity = value
            elif mode is BudgetAxis.LATENCY_BUDGET:
                latency_allowance_ms = value
        object.__setattr__(self, "review_mode", ReviewMode(review_mode))
        object.__setattr__(self, "review_fraction", review_fraction)
        object.__setattr__(self, "fixed_k", fixed_k)
        object.__setattr__(self, "cost_matrix", cost_matrix)
        object.__setattr__(self, "abstention_capacity", abstention_capacity)
        object.__setattr__(self, "latency_allowance_ms", latency_allowance_ms)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.review_mode is ReviewMode.FRACTION:
            if self.review_fraction is None or not 0 < self.review_fraction <= 1:
                raise ValueError("fractional review capacity must be in (0, 1]")
        elif self.review_fraction is not None:
            raise ValueError("review_fraction requires fraction review mode")
        if self.review_mode is ReviewMode.FIXED_K:
            if self.fixed_k is None or self.fixed_k < 0:
                raise ValueError("fixed-K review mode requires non-negative fixed_k")
        elif self.fixed_k is not None:
            raise ValueError("fixed_k requires fixed-K review mode")
        if self.cost_matrix is not None:
            if len(self.cost_matrix) != 2 or any(len(row) != 2 for row in self.cost_matrix):
                raise ValueError("cost matrix must be 2x2")
            if any(value < 0 for row in self.cost_matrix for value in row):
                raise ValueError("cost matrix entries cannot be negative")
        if self.abstention_capacity is not None and not 0 <= self.abstention_capacity <= 1:
            raise ValueError("abstention capacity must lie in [0,1]")
        if self.latency_allowance_ms is not None and self.latency_allowance_ms < 0:
            raise ValueError("latency allowance cannot be negative")

    @property
    def mode(self) -> BudgetAxis:
        if self.review_mode is ReviewMode.FRACTION:
            return BudgetAxis.FRACTIONAL_REVIEW_CAPACITY
        if self.review_mode is ReviewMode.FIXED_K:
            return BudgetAxis.FIXED_K
        return BudgetAxis.UNCONSTRAINED_RANKING

    @property
    def value(self) -> Optional[float]:
        if self.review_mode is ReviewMode.FRACTION:
            return self.review_fraction
        if self.review_mode is ReviewMode.FIXED_K:
            return None if self.fixed_k is None else float(self.fixed_k)
        return None


@dataclass(frozen=True, init=False)
class ResourceSpec:
    device_class: DeviceClass
    memory_cap_gb: Optional[float]
    latency_cap_ms: Optional[float]
    unavailable_experts: Tuple[str, ...]
    custom_envelope_id: Optional[str]
    measurement_status: MeasurementStatus

    def __init__(
        self,
        legacy_mode: ResourceAxis | None = None,
        *,
        device_class: DeviceClass = DeviceClass.UNKNOWN,
        memory_cap_gb: Optional[float] = None,
        latency_cap_ms: Optional[float] = None,
        unavailable_experts: Tuple[str, ...] = (),
        custom_envelope_id: Optional[str] = None,
        measurement_status: MeasurementStatus = MeasurementStatus.UNKNOWN,
        memory_gb: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        memory_cap_gb = memory_cap_gb if memory_cap_gb is not None else memory_gb
        latency_cap_ms = latency_cap_ms if latency_cap_ms is not None else latency_ms
        if legacy_mode is not None:
            mode = ResourceAxis(legacy_mode)
            device_class = {
                ResourceAxis.CPU: DeviceClass.CPU,
                ResourceAxis.SINGLE_T4: DeviceClass.SINGLE_T4,
                ResourceAxis.DUAL_T4: DeviceClass.DUAL_T4,
                ResourceAxis.CUSTOM_DEVICE_ENVELOPE: DeviceClass.CUSTOM,
            }.get(mode, device_class)
        object.__setattr__(self, "device_class", DeviceClass(device_class))
        object.__setattr__(self, "memory_cap_gb", memory_cap_gb)
        object.__setattr__(self, "latency_cap_ms", latency_cap_ms)
        object.__setattr__(self, "unavailable_experts", tuple(unavailable_experts))
        object.__setattr__(self, "custom_envelope_id", custom_envelope_id)
        object.__setattr__(self, "measurement_status", MeasurementStatus(measurement_status))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.memory_cap_gb is not None and self.memory_cap_gb <= 0:
            raise ValueError("memory_cap_gb must be positive")
        if self.latency_cap_ms is not None and self.latency_cap_ms < 0:
            raise ValueError("latency_cap_ms cannot be negative")
        for expert in self.unavailable_experts:
            validate_identifier(expert, "unavailable_expert")
        if self.device_class is DeviceClass.CUSTOM:
            if self.custom_envelope_id is None:
                raise ValueError("custom device class requires custom_envelope_id")
            validate_identifier(self.custom_envelope_id, "custom_envelope_id")
        elif self.custom_envelope_id is not None:
            raise ValueError("custom envelope requires custom device class")

    @property
    def mode(self) -> ResourceAxis:
        if self.device_class is DeviceClass.CPU:
            return ResourceAxis.CPU
        if self.device_class is DeviceClass.SINGLE_T4:
            return ResourceAxis.SINGLE_T4
        if self.device_class is DeviceClass.DUAL_T4:
            return ResourceAxis.DUAL_T4
        if self.device_class is DeviceClass.CUSTOM:
            return ResourceAxis.CUSTOM_DEVICE_ENVELOPE
        return ResourceAxis.UNKNOWN

    @property
    def memory_gb(self) -> Optional[float]:
        return self.memory_cap_gb

    @property
    def latency_ms(self) -> Optional[float]:
        return self.latency_cap_ms
