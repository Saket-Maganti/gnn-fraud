"""Role-neutral V5 prediction artifacts and scenario-bound no-training audits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from coregraph.contracts.axes import AccessRegime, ContractRole, VisibilityAxis, VisibilitySpec
from coregraph.contracts.contract import DeploymentContract
from coregraph.contracts.serialization import stable_sha256, to_primitive
from coregraph.data.leakage import (
    ScenarioPredictionScope,
    audit_evaluation_scenario_scopes,
)
from coregraph.objectives.scores import ScoreType, validate_numpy_scores

BASE_MANIFEST_SCHEMA = "coregraph_base_prediction_manifest_v5"
SCENARIO_BINDING_SCHEMA = "coregraph_evaluation_scenario_binding_v5"
SCENARIO_INDEX_SCHEMA = "coregraph_scenario_binding_index_v5"
NO_TRAINING_SCHEMA = "coregraph_saved_output_no_training_validation_v5"

ALLOWED_SPLITS = frozenset({"train", "validation", "test", "unscored"})
SOURCE_SPLITS = ("train", "validation")
TARGET_SPLITS = ("test",)


class CodeProvenanceType(str, Enum):
    GIT_COMMIT = "GIT_COMMIT"
    RUNTIME_BUNDLE_SHA256 = "RUNTIME_BUNDLE_SHA256"
    SOURCE_ARCHIVE_SHA256 = "SOURCE_ARCHIVE_SHA256"
    CONTAINER_IMAGE_DIGEST = "CONTAINER_IMAGE_DIGEST"
    UNRESOLVED_LEGACY_CODE = "UNRESOLVED_LEGACY_CODE"


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_neutral_contract_coordinates(
    contract: DeploymentContract,
) -> dict[str, Any]:
    """Return exactly the payload used by DeploymentContract.coordinate_hash."""

    return {
        "schema_version": contract.schema_version,
        "time": to_primitive(contract.time),
        "visibility": to_primitive(contract.visibility),
        "construction": to_primitive(contract.construction),
        "selection": to_primitive(contract.selection),
        "budget": to_primitive(contract.budget),
        "resource": to_primitive(contract.resource),
        "access_regime": to_primitive(contract.access_regime),
    }


def make_scenario_id(
    *,
    dataset: str,
    target_protocol_id: str,
    expert_prediction_seed: int,
    fold: str,
    access_regime: str,
) -> str:
    digest = stable_sha256(
        {
            "dataset": dataset,
            "target_protocol_id": target_protocol_id,
            "expert_prediction_seed": expert_prediction_seed,
            "fold": fold,
            "access_regime": access_regime,
        }
    )
    return f"scenario-{digest[:24]}"


@dataclass(frozen=True)
class BasePredictionArtifact:
    """One immutable role-neutral prediction export."""

    dataset: str
    task: str
    prediction_unit: str
    protocol_id: str
    contract_coordinate_hash: str
    role_neutral_contract_coordinates: Mapping[str, Any]
    expert_id: str
    expert_prediction_seed: int
    fold: str
    path: Path
    checksum: str
    row_schema: tuple[str, ...]
    provider_split_mapping: Mapping[str, str]
    label_mapping: Mapping[str, str]
    positive_label_id: int
    config_payload: Mapping[str, Any]
    config_sha256: str
    config_provenance_type: str
    config_provenance_path: str
    code_provenance_type: CodeProvenanceType
    code_provenance_value: str
    code_provenance_path: str
    routing_cost_value: float | None
    routing_cost_unit: str
    routing_cost_provenance: str
    measured_compute_available: bool
    measured_compute_record: Mapping[str, Any]
    validation_evidence: tuple[Mapping[str, Any], ...]
    artifact_family: str
    source_package: str
    source_archive_path: str
    source_archive_sha256: str
    alias_lineage: tuple[str, ...] = ()
    score_type: ScoreType = ScoreType.PROBABILITY

    def __post_init__(self) -> None:
        required_identity = (
            self.dataset,
            self.task,
            self.prediction_unit,
            self.protocol_id,
            self.expert_id,
            self.fold,
            self.artifact_family,
            self.source_package,
        )
        if any(not value for value in required_identity):
            raise ValueError("base artifact identity fields cannot be empty")
        if self.expert_prediction_seed < 0:
            raise ValueError("base artifact seed must be non-negative")
        if not _valid_sha256(self.checksum):
            raise ValueError("base prediction checksum must be lowercase SHA-256")
        if stable_sha256(self.role_neutral_contract_coordinates) != (
            self.contract_coordinate_hash
        ):
            raise ValueError("base artifact contract coordinate hash mismatch")
        if "contract_role" in self.role_neutral_contract_coordinates:
            raise ValueError("base artifact coordinates cannot contain contract_role")
        if not self.row_schema or len(self.row_schema) != len(set(self.row_schema)):
            raise ValueError("base artifact row schema must be non-empty and unique")
        required_columns = {
            f"{self.prediction_unit}_id",
            "score",
            "y_true",
            "split",
            "label_known",
        }
        if not required_columns.issubset(self.row_schema):
            raise ValueError("base artifact row schema is incomplete")
        if (
            not self.provider_split_mapping
            or set(self.provider_split_mapping.values()) - ALLOWED_SPLITS
        ):
            raise ValueError("base artifact provider split mapping is invalid")
        if str(self.positive_label_id) not in self.label_mapping:
            raise ValueError("positive label ID is absent from the label mapping")
        if self.config_payload:
            if stable_sha256(self.config_payload) != self.config_sha256:
                raise ValueError("base artifact config hash mismatch")
        elif self.config_sha256:
            raise ValueError("empty config payload cannot have a config hash")
        if not self.config_provenance_type or not self.config_provenance_path:
            raise ValueError("config provenance type and path are required")
        if not self.code_provenance_value or not self.code_provenance_path:
            raise ValueError("code provenance value and path are required")
        if self.routing_cost_value is not None and self.routing_cost_value < 0:
            raise ValueError("routing cost cannot be negative")
        if self.routing_cost_value is None:
            if self.routing_cost_unit or self.routing_cost_provenance not in {
                "UNRESOLVED",
                "NOT_APPLICABLE",
            }:
                raise ValueError("unresolved routing cost must be explicit")
        elif not self.routing_cost_unit or not self.routing_cost_provenance:
            raise ValueError("resolved routing cost requires unit and provenance")
        if self.measured_compute_available != bool(self.measured_compute_record):
            raise ValueError("measured compute availability and record disagree")
        if self.source_archive_sha256 and not _valid_sha256(
            self.source_archive_sha256
        ):
            raise ValueError("source archive checksum must be lowercase SHA-256")

    @property
    def logical_key(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.dataset,
            self.task,
            self.protocol_id,
            self.expert_id,
            self.expert_prediction_seed,
            self.fold,
        )

    @property
    def base_artifact_hash(self) -> str:
        return stable_sha256(
            {
                "schema_version": BASE_MANIFEST_SCHEMA,
                "logical_key": self.logical_key,
                "prediction_checksum": self.checksum,
                "contract_coordinate_hash": self.contract_coordinate_hash,
                "artifact_family": self.artifact_family,
                "source_package": self.source_package,
                "alias_lineage": self.alias_lineage,
            }
        )

    def to_manifest(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        prediction_path = self.path
        if relative_to is not None:
            try:
                prediction_path = self.path.relative_to(relative_to)
            except ValueError:
                pass
        return {
            "schema_version": BASE_MANIFEST_SCHEMA,
            "base_artifact_hash": self.base_artifact_hash,
            "dataset": self.dataset,
            "task": self.task,
            "prediction_unit": self.prediction_unit,
            "protocol_id": self.protocol_id,
            "contract_coordinate_hash": self.contract_coordinate_hash,
            "role_neutral_contract_coordinates": to_primitive(
                self.role_neutral_contract_coordinates
            ),
            "expert_id": self.expert_id,
            "expert_prediction_seed": self.expert_prediction_seed,
            "fold": self.fold,
            "prediction_path": str(prediction_path),
            "prediction_checksum": self.checksum,
            "row_schema": list(self.row_schema),
            "provider_split_mapping": dict(self.provider_split_mapping),
            "label_mapping": dict(self.label_mapping),
            "positive_label_id": self.positive_label_id,
            "config_payload": to_primitive(self.config_payload),
            "config_sha256": self.config_sha256,
            "config_provenance_type": self.config_provenance_type,
            "config_provenance_path": self.config_provenance_path,
            "code_provenance_type": self.code_provenance_type.value,
            "code_provenance_value": self.code_provenance_value,
            "code_provenance_path": self.code_provenance_path,
            "routing_cost_value": self.routing_cost_value,
            "routing_cost_unit": self.routing_cost_unit,
            "routing_cost_provenance": self.routing_cost_provenance,
            "measured_compute_available": self.measured_compute_available,
            "measured_compute_record": to_primitive(self.measured_compute_record),
            "validation_evidence": to_primitive(self.validation_evidence),
            "artifact_family": self.artifact_family,
            "source_package": self.source_package,
            "source_archive_path": self.source_archive_path,
            "source_archive_sha256": self.source_archive_sha256,
            "alias_lineage": list(self.alias_lineage),
            "score_type": self.score_type.value,
        }


@dataclass(frozen=True)
class ScenarioArtifactBinding:
    """A source or target role assigned inside exactly one scenario."""

    scenario_id: str
    base_artifact_hash: str
    base_protocol_id: str
    bound_protocol_id: str
    expert_id: str
    role: str
    permitted_splits: tuple[str, ...]
    evaluation_split: str
    require_label_known: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"source", "target"}:
            raise ValueError("scenario binding role must be source or target")
        if self.role == "source":
            if self.permitted_splits != SOURCE_SPLITS:
                raise ValueError("source binding must permit train and validation")
            if self.evaluation_split != "validation":
                raise ValueError("source binding evaluation split must be validation")
        else:
            if self.permitted_splits != TARGET_SPLITS:
                raise ValueError("target binding must permit only test")
            if self.evaluation_split != "test":
                raise ValueError("target binding evaluation split must be test")
        if not self.require_label_known:
            raise ValueError("scenario bindings must require label-known rows")

    @property
    def role_binding_id(self) -> str:
        return stable_sha256(
            {
                "scenario_id": self.scenario_id,
                "base_artifact_hash": self.base_artifact_hash,
                "base_protocol_id": self.base_protocol_id,
                "bound_protocol_id": self.bound_protocol_id,
                "expert_id": self.expert_id,
                "role": self.role,
                "permitted_splits": self.permitted_splits,
                "evaluation_split": self.evaluation_split,
                "require_label_known": self.require_label_known,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "permitted_splits": list(self.permitted_splits),
            "role_binding_id": self.role_binding_id,
        }


@dataclass(frozen=True)
class EvaluationScenarioBinding:
    """Nine artifact-role bindings for one held-out target protocol."""

    scenario_id: str
    dataset: str
    target_protocol_id: str
    source_protocol_ids: tuple[str, str]
    expert_prediction_seed: int
    fold: str
    access_regime: AccessRegime
    target_operational_contract: DeploymentContract
    bindings: tuple[ScenarioArtifactBinding, ...]
    no_target_labels_during_fitting: bool = True

    def __post_init__(self) -> None:
        expected_id = make_scenario_id(
            dataset=self.dataset,
            target_protocol_id=self.target_protocol_id,
            expert_prediction_seed=self.expert_prediction_seed,
            fold=self.fold,
            access_regime=self.access_regime.value,
        )
        if self.scenario_id != expected_id:
            raise ValueError("scenario ID does not match its immutable coordinates")
        if len(set(self.source_protocol_ids)) != 2:
            raise ValueError("scenario requires exactly two unique source protocols")
        if self.target_protocol_id in self.source_protocol_ids:
            raise ValueError("target protocol cannot also be a source protocol")
        if self.target_operational_contract.role is not ContractRole.TARGET:
            raise ValueError("scenario target operational contract must have target role")
        if self.target_operational_contract.access_regime is not self.access_regime:
            raise ValueError("scenario access regime and target contract disagree")
        if not self.no_target_labels_during_fitting:
            raise ValueError("target-label fitting access is forbidden")
        if len(self.bindings) != 9:
            raise ValueError("scenario requires exactly nine artifact bindings")
        if any(binding.scenario_id != self.scenario_id for binding in self.bindings):
            raise ValueError("scenario contains a binding with another scenario ID")
        roles = Counter(binding.role for binding in self.bindings)
        if roles != {"source": 6, "target": 3}:
            raise ValueError("scenario requires six source and three target bindings")
        expected_protocol_roles = {
            **{protocol: "source" for protocol in self.source_protocol_ids},
            self.target_protocol_id: "target",
        }
        for protocol, role in expected_protocol_roles.items():
            selected = [
                binding
                for binding in self.bindings
                if binding.bound_protocol_id == protocol and binding.role == role
            ]
            if len(selected) != 3 or len({item.expert_id for item in selected}) != 3:
                raise ValueError(
                    "scenario requires exactly three distinct experts per protocol"
                )
        roles_by_hash: dict[str, set[str]] = defaultdict(set)
        for binding in self.bindings:
            roles_by_hash[binding.base_artifact_hash].add(binding.role)
        if any(roles == {"source", "target"} for roles in roles_by_hash.values()):
            raise ValueError("one base artifact cannot hold both roles in one scenario")

    def validate_artifact_references(
        self,
        artifacts_by_hash: Mapping[str, BasePredictionArtifact],
    ) -> None:
        for binding in self.bindings:
            artifact = artifacts_by_hash.get(binding.base_artifact_hash)
            if artifact is None:
                raise ValueError(
                    f"scenario binding references missing base artifact "
                    f"{binding.base_artifact_hash}"
                )
            if (
                artifact.dataset != self.dataset
                or artifact.protocol_id != binding.base_protocol_id
                or artifact.protocol_id != binding.bound_protocol_id
                or artifact.expert_id != binding.expert_id
                or artifact.expert_prediction_seed != self.expert_prediction_seed
                or artifact.fold != self.fold
            ):
                raise ValueError("scenario binding conflicts with base artifact identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCENARIO_BINDING_SCHEMA,
            "scenario_id": self.scenario_id,
            "dataset": self.dataset,
            "target_protocol_id": self.target_protocol_id,
            "source_protocol_ids": list(self.source_protocol_ids),
            "expert_prediction_seed": self.expert_prediction_seed,
            "fold": self.fold,
            "access_regime": self.access_regime.value,
            "target_operational_contract": self.target_operational_contract.to_dict(),
            "no_target_labels_during_fitting": self.no_target_labels_during_fitting,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }


@dataclass(frozen=True)
class BaseRowAudit:
    base_artifact_hash: str
    row_count: int
    identifier_count: int
    duplicate_identifier_count: int
    split_counts: Mapping[str, int]
    label_known_counts: Mapping[str, int]
    provider_unknown_code: tuple[int, ...]
    unknown_label_counts: Mapping[str, int]
    score_min: float
    score_max: float
    timestamp_min: float | None
    timestamp_max: float | None
    identifiers: tuple[str, ...] = field(repr=False)
    labels: tuple[int, ...] = field(repr=False)
    splits: tuple[str, ...] = field(repr=False)
    label_known: tuple[bool, ...] = field(repr=False)
    scores: tuple[float, ...] = field(repr=False)
    timestamps: tuple[float | None, ...] = field(repr=False)

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "identifiers",
            "labels",
            "splits",
            "label_known",
            "scores",
            "timestamps",
        ):
            payload.pop(name)
        return payload


def load_base_prediction_artifacts(
    manifests: Sequence[Path],
) -> list[BasePredictionArtifact]:
    artifacts: list[BasePredictionArtifact] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != BASE_MANIFEST_SCHEMA:
            raise ValueError(f"{manifest_path} is not a V5 base prediction manifest")
        if "contract_role" in payload:
            raise ValueError("V5 base prediction manifest cannot contain contract_role")
        prediction_path = Path(str(payload["prediction_path"]))
        if not prediction_path.is_absolute():
            prediction_path = (manifest_path.parent / prediction_path).resolve()
        if not prediction_path.is_file():
            raise ValueError(
                f"{manifest_path} prediction file is missing: {prediction_path}"
            )
        checksum = _sha256_path(prediction_path)
        if checksum != str(payload["prediction_checksum"]):
            raise ValueError(f"{manifest_path} prediction checksum mismatch")
        artifact = BasePredictionArtifact(
            dataset=str(payload["dataset"]),
            task=str(payload["task"]),
            prediction_unit=str(payload["prediction_unit"]),
            protocol_id=str(payload["protocol_id"]),
            contract_coordinate_hash=str(payload["contract_coordinate_hash"]),
            role_neutral_contract_coordinates=dict(
                payload["role_neutral_contract_coordinates"]
            ),
            expert_id=str(payload["expert_id"]),
            expert_prediction_seed=int(payload["expert_prediction_seed"]),
            fold=str(payload["fold"]),
            path=prediction_path,
            checksum=checksum,
            row_schema=tuple(str(value) for value in payload["row_schema"]),
            provider_split_mapping={
                str(key): str(value)
                for key, value in payload["provider_split_mapping"].items()
            },
            label_mapping={
                str(key): str(value) for key, value in payload["label_mapping"].items()
            },
            positive_label_id=int(payload["positive_label_id"]),
            config_payload=dict(payload["config_payload"]),
            config_sha256=str(payload["config_sha256"]),
            config_provenance_type=str(payload["config_provenance_type"]),
            config_provenance_path=str(payload["config_provenance_path"]),
            code_provenance_type=CodeProvenanceType(
                payload["code_provenance_type"]
            ),
            code_provenance_value=str(payload["code_provenance_value"]),
            code_provenance_path=str(payload["code_provenance_path"]),
            routing_cost_value=(
                None
                if payload["routing_cost_value"] is None
                else float(payload["routing_cost_value"])
            ),
            routing_cost_unit=str(payload["routing_cost_unit"]),
            routing_cost_provenance=str(payload["routing_cost_provenance"]),
            measured_compute_available=bool(payload["measured_compute_available"]),
            measured_compute_record=dict(payload["measured_compute_record"]),
            validation_evidence=tuple(
                dict(value) for value in payload["validation_evidence"]
            ),
            artifact_family=str(payload["artifact_family"]),
            source_package=str(payload["source_package"]),
            source_archive_path=str(payload["source_archive_path"]),
            source_archive_sha256=str(payload["source_archive_sha256"]),
            alias_lineage=tuple(str(value) for value in payload["alias_lineage"]),
            score_type=ScoreType(payload["score_type"]),
        )
        if str(payload.get("base_artifact_hash", "")) != artifact.base_artifact_hash:
            raise ValueError(f"{manifest_path} base artifact hash mismatch")
        artifacts.append(artifact)
    keys = [artifact.logical_key for artifact in artifacts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate V5 base artifact logical identity")
    hashes = [artifact.base_artifact_hash for artifact in artifacts]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate V5 base artifact hash")
    return artifacts


def load_evaluation_scenarios(path: Path) -> list[EvaluationScenarioBinding]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCENARIO_INDEX_SCHEMA:
        raise ValueError("scenario index must use the V5 schema")
    scenarios = []
    for record in payload.get("scenarios", ()):
        contract_payload = record.get("target_operational_contract")
        if not isinstance(contract_payload, Mapping):
            raise ValueError(
                f"scenario {record.get('scenario_id', '<unknown>')} is blocked: "
                "target operational contract is unresolved"
            )
        contract = DeploymentContract.from_dict(contract_payload)
        bindings = tuple(
            ScenarioArtifactBinding(
                scenario_id=str(item["scenario_id"]),
                base_artifact_hash=str(item["base_artifact_hash"]),
                base_protocol_id=str(item["base_protocol_id"]),
                bound_protocol_id=str(item["bound_protocol_id"]),
                expert_id=str(item["expert_id"]),
                role=str(item["role"]),
                permitted_splits=tuple(
                    str(value) for value in item["permitted_splits"]
                ),
                evaluation_split=str(item["evaluation_split"]),
                require_label_known=bool(item["require_label_known"]),
            )
            for item in record["bindings"]
        )
        scenarios.append(
            EvaluationScenarioBinding(
                scenario_id=str(record["scenario_id"]),
                dataset=str(record["dataset"]),
                target_protocol_id=str(record["target_protocol_id"]),
                source_protocol_ids=tuple(record["source_protocol_ids"]),
                expert_prediction_seed=int(record["expert_prediction_seed"]),
                fold=str(record["fold"]),
                access_regime=AccessRegime(record["access_regime"]),
                target_operational_contract=contract,
                no_target_labels_during_fitting=bool(
                    record["no_target_labels_during_fitting"]
                ),
                bindings=bindings,
            )
        )
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate V5 evaluation scenario ID")
    return scenarios


def _parse_known(value: str | None) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid label_known value {value!r}")


def _parse_timestamp(row: Mapping[str, str]) -> float | None:
    value = row.get("timestamp", row.get("timestep", ""))
    if value in {None, ""}:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("prediction timestamp must be finite")
    return parsed


def audit_base_artifact_rows(artifact: BasePredictionArtifact) -> BaseRowAudit:
    with artifact.path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("prediction CSV has no header")
        if tuple(reader.fieldnames) != artifact.row_schema:
            raise ValueError("prediction CSV row schema differs from base manifest")
        rows = list(reader)
    id_column = f"{artifact.prediction_unit}_id"
    identifiers = tuple(str(row[id_column]) for row in rows)
    labels = tuple(int(row["y_true"]) for row in rows)
    splits = tuple(
        artifact.provider_split_mapping.get(str(row["split"]), str(row["split"]))
        for row in rows
    )
    if set(splits) - ALLOWED_SPLITS:
        raise ValueError("prediction CSV contains an unmapped provider split")
    known = tuple(_parse_known(row.get("label_known")) for row in rows)
    scores_array = validate_numpy_scores(
        np.asarray([float(row["score"]) for row in rows]),
        artifact.score_type,
    )
    scores = tuple(float(value) for value in scores_array)
    timestamps = tuple(_parse_timestamp(row) for row in rows)
    if any(
        value not in {artifact.expert_id, ""}
        for value in (
            str(row.get("expert_id") or row.get("model") or "") for row in rows
        )
    ):
        raise ValueError("prediction row expert conflicts with base artifact")
    declared_labels = {int(value) for value in artifact.label_mapping}
    if not set(labels).issubset(declared_labels):
        raise ValueError("prediction CSV contains an undeclared provider label")
    unknown_codes = tuple(
        sorted(
            int(identifier)
            for identifier, meaning in artifact.label_mapping.items()
            if str(meaning).strip().lower()
            in {"unknown", "unlabeled", "unlabelled"}
        )
    )
    if any(is_known and label in unknown_codes for label, is_known in zip(labels, known)):
        raise ValueError("provider-unknown label cannot be declared label-known")
    known_times = [value for value in timestamps if value is not None]
    return BaseRowAudit(
        base_artifact_hash=artifact.base_artifact_hash,
        row_count=len(rows),
        identifier_count=len(set(identifiers)),
        duplicate_identifier_count=len(identifiers) - len(set(identifiers)),
        split_counts=dict(Counter(splits)),
        label_known_counts={
            "known": int(sum(known)),
            "unknown": int(len(known) - sum(known)),
        },
        provider_unknown_code=unknown_codes,
        unknown_label_counts={
            split: sum(
                1
                for label, value_split, is_known in zip(labels, splits, known)
                if value_split == split and (not is_known or label in unknown_codes)
            )
            for split in sorted(set(splits))
        },
        score_min=float(np.min(scores_array)) if len(scores_array) else float("nan"),
        score_max=float(np.max(scores_array)) if len(scores_array) else float("nan"),
        timestamp_min=min(known_times) if known_times else None,
        timestamp_max=max(known_times) if known_times else None,
        identifiers=identifiers,
        labels=labels,
        splits=splits,
        label_known=known,
        scores=scores,
        timestamps=timestamps,
    )


def _registry_consistent(
    artifact: BasePredictionArtifact,
    registry: Mapping[str, Any],
) -> bool:
    records = {
        str(record["protocol_id"]): record
        for record in registry.get("protocols", ())
        if isinstance(record, Mapping)
    }
    record = records.get(artifact.protocol_id)
    if record is None:
        return False
    expected_visibility = to_primitive(
        VisibilitySpec.from_v2(VisibilityAxis(record["visibility_profile"]))
    )
    return (
        artifact.role_neutral_contract_coordinates.get("visibility")
        == expected_visibility
    )


def materialize_evaluation_scenario(
    artifacts_by_hash: Mapping[str, BasePredictionArtifact],
    scenario: EvaluationScenarioBinding,
    *,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter rows and audit a scenario without fitting or target scoring."""

    scenario.validate_artifact_references(artifacts_by_hash)
    target_coordinate = next(
        artifacts_by_hash[binding.base_artifact_hash].contract_coordinate_hash
        for binding in scenario.bindings
        if binding.role == "target"
    )
    if scenario.target_operational_contract.coordinate_hash != target_coordinate:
        raise ValueError("target operational contract conflicts with base coordinate")
    audits: dict[str, BaseRowAudit] = {}
    scopes: list[ScenarioPredictionScope] = []
    row_scope_reports: list[dict[str, Any]] = []
    alignment: dict[str, tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], tuple[bool, ...], tuple[float | None, ...]]] = {}
    target_known_identifiers: set[str] = set()
    target_excluded_unknown_identifiers: set[str] = set()
    for binding in scenario.bindings:
        artifact = artifacts_by_hash[binding.base_artifact_hash]
        audit = audits.setdefault(
            artifact.base_artifact_hash,
            audit_base_artifact_rows(artifact),
        )
        semantic_key = (
            audit.identifiers,
            audit.labels,
            audit.splits,
            audit.label_known,
            audit.timestamps,
        )
        previous = alignment.setdefault(binding.bound_protocol_id, semantic_key)
        if previous != semantic_key:
            raise ValueError(
                "experts for one protocol disagree on identifiers or row semantics"
            )
        if binding.role == "source":
            selected = tuple(
                index
                for index, split in enumerate(audit.splits)
                if split in binding.permitted_splits
            )
            excluded_unknown = tuple(
                index for index in selected if not audit.label_known[index]
            )
        else:
            target_split = tuple(
                index
                for index, split in enumerate(audit.splits)
                if split == binding.evaluation_split
            )
            selected = tuple(
                index for index in target_split if audit.label_known[index]
            )
            excluded_unknown = tuple(
                index for index in target_split if not audit.label_known[index]
            )
            target_known_identifiers.update(
                audit.identifiers[index] for index in selected
            )
            target_excluded_unknown_identifiers.update(
                audit.identifiers[index] for index in excluded_unknown
            )
        scopes.append(
            ScenarioPredictionScope(
                scenario_id=scenario.scenario_id,
                dataset=scenario.dataset,
                base_artifact_hash=artifact.base_artifact_hash,
                expert_id=artifact.expert_id,
                base_protocol_id=artifact.protocol_id,
                bound_protocol_id=binding.bound_protocol_id,
                expert_prediction_seed=artifact.expert_prediction_seed,
                fold=artifact.fold,
                role=binding.role,
                contract_coordinate_hash=artifact.contract_coordinate_hash,
                path=str(artifact.path),
                checksum=artifact.checksum,
                selected_identifiers=tuple(
                    audit.identifiers[index] for index in selected
                ),
                selected_splits=tuple(audit.splits[index] for index in selected),
                selected_label_known=tuple(
                    audit.label_known[index] for index in selected
                ),
                selected_timestamps=tuple(
                    audit.timestamps[index] for index in selected
                ),
                target_labels_used_for_fitting=False,
                selection_metadata_fields=(),
                registry_consistent=_registry_consistent(artifact, registry),
            )
        )
        row_scope_reports.append(
            {
                **audit.public_dict(),
                "scenario_id": scenario.scenario_id,
                "role_binding_id": binding.role_binding_id,
                "role": binding.role,
                "base_protocol_id": artifact.protocol_id,
                "bound_protocol_id": binding.bound_protocol_id,
                "expert_id": artifact.expert_id,
                "selected_row_count": len(selected),
                "excluded_unknown_target_row_count": len(excluded_unknown)
                if binding.role == "target"
                else 0,
                "permitted_splits": list(binding.permitted_splits),
                "evaluation_split": binding.evaluation_split,
                "target_label_values_exposed": False,
            }
        )
    leakage = audit_evaluation_scenario_scopes(
        scopes,
        scenario_id=scenario.scenario_id,
        dataset=scenario.dataset,
        target_protocol_id=scenario.target_protocol_id,
        source_protocol_ids=scenario.source_protocol_ids,
        expert_prediction_seed=scenario.expert_prediction_seed,
        fold=scenario.fold,
    )
    if not leakage.passed:
        codes = sorted(
            finding.code
            for finding in leakage.findings
            if finding.severity == "ATOMIC"
        )
        raise RuntimeError(f"atomic scenario leakage blocks validation: {codes}")
    return {
        "scenario_id": scenario.scenario_id,
        "dataset": scenario.dataset,
        "target_protocol_id": scenario.target_protocol_id,
        "source_protocol_ids": list(scenario.source_protocol_ids),
        "expert_prediction_seed": scenario.expert_prediction_seed,
        "fold": scenario.fold,
        "target_contract_coordinate_hash": target_coordinate,
        "target_contract_id": scenario.target_operational_contract.contract_id,
        "row_scope_reports": row_scope_reports,
        "leakage_report": leakage.to_dict(),
        "target_known_test_identifiers": sorted(target_known_identifiers),
        "excluded_unknown_target_identifiers": sorted(
            target_excluded_unknown_identifiers
        ),
        "target_labels_accessed_before_scoring": False,
        "target_label_values_exposed": False,
        "training_performed": False,
        "metric_computation_performed": False,
        "oracle_computation_performed": False,
    }


def validate_no_training_scenarios(
    artifacts: Sequence[BasePredictionArtifact],
    scenarios: Sequence[EvaluationScenarioBinding],
    *,
    registry: Mapping[str, Any],
    expected_datasets: Sequence[str],
    expected_protocols: Sequence[str],
    expected_experts: Sequence[str],
    expected_seeds: Sequence[int],
    expected_folds: Sequence[str] = ("fold0",),
) -> dict[str, Any]:
    artifacts_by_hash = {artifact.base_artifact_hash: artifact for artifact in artifacts}
    expected_base = (
        len(expected_datasets)
        * len(expected_protocols)
        * len(expected_experts)
        * len(expected_seeds)
        * len(expected_folds)
    )
    expected_scenarios = (
        len(expected_datasets)
        * len(expected_protocols)
        * len(expected_seeds)
        * len(expected_folds)
    )
    expected_bindings = expected_scenarios * len(expected_protocols) * len(
        expected_experts
    )
    if len(artifacts) != expected_base:
        raise ValueError(
            f"base artifact completeness failed: expected {expected_base}, "
            f"actual {len(artifacts)}"
        )
    logical = {artifact.logical_key for artifact in artifacts}
    expected_logical = {
        (dataset, "node_classification", protocol, expert, seed, fold)
        for dataset in expected_datasets
        for protocol in expected_protocols
        for expert in expected_experts
        for seed in expected_seeds
        for fold in expected_folds
    }
    if logical != expected_logical:
        raise ValueError("base artifact logical completeness surface is not exact")
    if len(scenarios) != expected_scenarios:
        raise ValueError(
            f"scenario completeness failed: expected {expected_scenarios}, "
            f"actual {len(scenarios)}"
        )
    if sum(len(scenario.bindings) for scenario in scenarios) != expected_bindings:
        raise ValueError("scenario binding completeness surface is not exact")
    materializations = [
        materialize_evaluation_scenario(
            artifacts_by_hash,
            scenario,
            registry=registry,
        )
        for scenario in scenarios
    ]
    return {
        "schema_version": NO_TRAINING_SCHEMA,
        "status": "VALIDATED_NO_TRAINING_V5",
        "base_artifact_count": len(artifacts),
        "scenario_count": len(scenarios),
        "scenario_binding_count": sum(
            len(scenario.bindings) for scenario in scenarios
        ),
        "expected_base_artifact_count": expected_base,
        "expected_scenario_count": expected_scenarios,
        "expected_scenario_binding_count": expected_bindings,
        "training_performed": False,
        "fitting_path_reachable": False,
        "metric_computation_performed": False,
        "oracle_computation_performed": False,
        "target_labels_accessed_before_scoring": False,
        "materializations": materializations,
    }
