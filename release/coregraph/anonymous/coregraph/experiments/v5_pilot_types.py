"""Strict identities and label-safe data interfaces for the V5 saved-output pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from coregraph.contracts.serialization import stable_sha256


PRIMARY_METHODS = (
    "coregraph",
    "uniform_average",
    "best_fixed_expert",
    "source_logistic_gate",
)
EXPERT_ORDER = ("feature_mlp", "gcn", "graphsage")
METHOD_REGISTRY_VERSION = "coregraph_v5_primary_methods_v1"
METRIC_SCHEMA_VERSION = "coregraph_v5_metric_schema_v2"


def _sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class V5BaseArtifact:
    dataset: str
    protocol: str
    expert: str
    provider_seed: int
    fold: str
    base_coordinate_id: str
    base_artifact_hash: str
    archive_name: str
    archive_sha256: str
    member_name: str
    member_sha256: str
    schema_version: str
    score_semantics: str
    row_key_semantics: tuple[str, ...]
    split_token_semantics: Mapping[str, str]
    label_known_semantics: str
    provenance_status: str
    row_count: int
    label_known_count: int

    def __post_init__(self) -> None:
        if self.expert not in EXPERT_ORDER:
            raise ValueError(f"unknown V5 expert {self.expert!r}")
        if self.provider_seed < 0 or self.row_count < 0 or self.label_known_count < 0:
            raise ValueError("V5 artifact counts and seed must be non-negative")
        for value, name in (
            (self.base_coordinate_id, "base_coordinate_id"),
            (self.base_artifact_hash, "base_artifact_hash"),
            (self.archive_sha256, "archive_sha256"),
            (self.member_sha256, "member_sha256"),
        ):
            _sha256(value, name)
        if self.score_semantics != "probability_[0,1]":
            raise ValueError("V5 pilot requires probability scores")
        if self.label_known_count > self.row_count:
            raise ValueError("label-known count cannot exceed row count")

    @property
    def logical_key(self) -> tuple[str, str, str, int]:
        return self.dataset, self.protocol, self.expert, self.provider_seed

    def identity_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "expert": self.expert,
            "provider_seed": self.provider_seed,
            "fold": self.fold,
            "base_coordinate_id": self.base_coordinate_id,
            "base_artifact_hash": self.base_artifact_hash,
            "archive_name": self.archive_name,
            "archive_sha256": self.archive_sha256,
            "member_name": self.member_name,
            "member_sha256": self.member_sha256,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class V5ScenarioDefinition:
    scenario_id: str
    dataset: str
    provider_seed: int
    fold: str
    target_protocol: str
    source_protocols: tuple[str, str]
    access_regime: str
    feasible_experts: tuple[str, ...]
    resource_profile: str
    review_fraction: float
    methods: tuple[str, ...] = PRIMARY_METHODS
    target_evaluation_policy: str = "offline_after_policy_freeze"

    def __post_init__(self) -> None:
        if len(self.source_protocols) != 2 or len(set(self.source_protocols)) != 2:
            raise ValueError("V5 scenario requires exactly two source protocols")
        if self.target_protocol in self.source_protocols:
            raise ValueError("target protocol cannot also be a source protocol")
        if tuple(self.methods) != PRIMARY_METHODS:
            raise ValueError("V5 primary method set or order differs from preregistration")
        if tuple(self.feasible_experts) != EXPERT_ORDER:
            raise ValueError("V5 expert set or order differs from preregistration")
        if not 0 < self.review_fraction <= 1:
            raise ValueError("review fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class V5ScenarioBinding:
    binding_id: str
    scenario_id: str
    base_coordinate_id: str
    protocol: str
    expert: str
    provider_seed: int
    role: str
    source_environment_id: str | None
    permitted_splits: tuple[str, ...]
    label_access_policy: str

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise ValueError("binding role must be source or target")
        expected = ("train", "validation") if self.role == "source" else ("test",)
        if self.permitted_splits != expected:
            raise ValueError(f"{self.role} binding has invalid split scope")
        if self.role == "source" and not self.source_environment_id:
            raise ValueError("source binding requires an environment identity")
        if self.role == "target" and self.source_environment_id is not None:
            raise ValueError("target binding cannot have a source environment identity")


@dataclass(frozen=True, slots=True)
class V5ScenarioMaterialization:
    definition: V5ScenarioDefinition
    source_bindings: tuple[V5ScenarioBinding, ...]
    target_bindings: tuple[V5ScenarioBinding, ...]
    artifacts_by_coordinate: Mapping[str, V5BaseArtifact]
    scenario_fingerprint: str

    def __post_init__(self) -> None:
        if len(self.source_bindings) != 6 or len(self.target_bindings) != 3:
            raise ValueError("materialization requires six source and three target bindings")
        bindings = (*self.source_bindings, *self.target_bindings)
        if any(item.scenario_id != self.definition.scenario_id for item in bindings):
            raise ValueError("materialization contains a foreign scenario binding")
        roles: dict[str, set[str]] = {}
        for binding in bindings:
            roles.setdefault(binding.base_coordinate_id, set()).add(binding.role)
        if any(value == {"source", "target"} for value in roles.values()):
            raise ValueError("one artifact cannot have both roles within a scenario")
        expected = {
            (protocol, expert, "source")
            for protocol in self.definition.source_protocols
            for expert in EXPERT_ORDER
        } | {
            (self.definition.target_protocol, expert, "target") for expert in EXPERT_ORDER
        }
        observed = {(item.protocol, item.expert, item.role) for item in bindings}
        if observed != expected:
            raise ValueError("scenario bindings do not form the exact 6+3 identity grid")
        _sha256(self.scenario_fingerprint, "scenario_fingerprint")


@dataclass(frozen=True, slots=True)
class SourceEnvironmentBundle:
    environment_id: str
    dataset: str
    protocol: str
    provider_seed: int
    row_keys: tuple[str, ...]
    scores: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    availability: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.row_keys)
        if self.scores.shape != (rows, len(EXPERT_ORDER)):
            raise ValueError("source score matrix shape is invalid")
        if self.labels.shape != (rows,) or self.splits.shape != (rows,):
            raise ValueError("source labels/splits do not align")
        if self.availability.shape != self.scores.shape:
            raise ValueError("source availability does not align")
        if len(set(self.row_keys)) != rows:
            raise ValueError("source environment contains duplicate row keys")
        if not set(np.unique(self.splits)).issubset({"train", "validation"}):
            raise ValueError("source bundle contains a forbidden split")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("source scores must be finite probabilities")


@dataclass(frozen=True, slots=True)
class TargetUnlabeledBundle:
    """The only target interface accepted by fit/inference code; labels do not exist here."""

    dataset: str
    protocol: str
    provider_seed: int
    row_keys: tuple[str, ...]
    scores: np.ndarray
    availability: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.row_keys)
        if self.scores.shape != (rows, len(EXPERT_ORDER)):
            raise ValueError("target score matrix shape is invalid")
        if self.availability.shape != self.scores.shape:
            raise ValueError("target availability does not align")
        if len(set(self.row_keys)) != rows:
            raise ValueError("target bundle contains duplicate row keys")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("target scores must be finite probabilities")

    def to_serializable(self) -> Mapping[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "provider_seed": self.provider_seed,
            "row_count": len(self.row_keys),
            "row_key_sha256": stable_sha256(self.row_keys),
            "score_sha256": stable_sha256(self.scores.astype(np.float32).tolist()),
            "target_labels_present": False,
        }


@dataclass(frozen=True, slots=True)
class TargetEvaluationBundle:
    dataset: str
    protocol: str
    provider_seed: int
    row_keys: tuple[str, ...]
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.shape != (len(self.row_keys),):
            raise ValueError("target evaluation labels do not align")
        if len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError("target evaluation contains duplicate row keys")


@dataclass(frozen=True, slots=True)
class FrozenPilotPolicy:
    method: str
    scenario_fingerprint: str
    provider_seed: int
    fit_seed: int
    state: Mapping[str, Any]
    preprocessing_state: Mapping[str, Any]
    threshold_state: Mapping[str, Any]
    fit_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.method not in PRIMARY_METHODS:
            raise ValueError("unknown primary method")
        _sha256(self.scenario_fingerprint, "scenario_fingerprint")

    @property
    def policy_state_hash(self) -> str:
        return stable_sha256(
            {
                "method": self.method,
                "scenario_fingerprint": self.scenario_fingerprint,
                "provider_seed": self.provider_seed,
                "fit_seed": self.fit_seed,
                "state": self.state,
                "preprocessing_state": self.preprocessing_state,
                "threshold_state": self.threshold_state,
                "fit_report": self.fit_report,
            }
        )


@dataclass(frozen=True, slots=True)
class PilotCoordinate:
    dataset: str
    target_protocol: str
    provider_seed: int
    method: str
    pilot_specification_version: str
    scenario_id: str
    scenario_fingerprint: str
    effective_execution_config_sha256: str

    def __post_init__(self) -> None:
        if self.method not in PRIMARY_METHODS:
            raise ValueError("coordinate method is outside the frozen primary set")
        _sha256(
            self.effective_execution_config_sha256,
            "effective_execution_config_sha256",
        )

    @property
    def key(self) -> str:
        return stable_sha256(asdict(self))


class PilotStage(str, Enum):
    PLANNED = "PLANNED"
    INPUTS_VALIDATED = "INPUTS_VALIDATED"
    SOURCE_ASSEMBLED = "SOURCE_ASSEMBLED"
    POLICY_FITTED = "POLICY_FITTED"
    POLICY_FROZEN = "POLICY_FROZEN"
    TARGET_SCORED = "TARGET_SCORED"
    EVALUATED = "EVALUATED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PilotCheckpoint:
    coordinate_key: str
    identity_hash: str
    stage: PilotStage
    output_schema_version: str
    metric_schema_version: str
    effective_execution_config_sha256: str
    checksums: Mapping[str, str] = field(default_factory=dict)
    retry_count: int = 0


@dataclass(frozen=True, slots=True)
class PilotResultRecord:
    coordinate: PilotCoordinate
    execution_status: str
    metrics: Mapping[str, Any]
    route_summary: Mapping[str, Any]
    policy_freeze_sha256: str
    target_score_sha256: str
    evaluation_sha256: str


@dataclass(frozen=True, slots=True)
class PilotGateRecord:
    outcome: str
    reasons: tuple[str, ...]
    coordinate_count: int
    preregistration_sha256: str
    effective_execution_config_sha256: str
    metric_schema_version: str

    def __post_init__(self) -> None:
        if self.outcome not in {"GO", "NO_GO", "INCONCLUSIVE"}:
            raise ValueError("gate outcome must be GO, NO_GO, or INCONCLUSIVE")
