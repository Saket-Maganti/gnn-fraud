"""Join V5 role-neutral registries to checksum-addressed archive members."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from coregraph.contracts.serialization import stable_sha256
from coregraph.evidence.archive_store import ArchiveStore
from coregraph.evidence.member_index import MemberIndex
from coregraph.experiments.v5_pilot_types import (
    EXPERT_ORDER,
    METRIC_SCHEMA_VERSION,
    PRIMARY_METHODS,
    PilotCoordinate,
    V5BaseArtifact,
    V5ScenarioBinding,
    V5ScenarioDefinition,
    V5ScenarioMaterialization,
)


CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "pilot_specification_version",
        "preregistration_path",
        "preregistration_sha256",
        "base_artifact_registry",
        "scenario_registry",
        "binding_registry",
        "member_index",
        "archive_hashes",
        "required_datasets",
        "required_protocols",
        "required_experts",
        "required_provider_seeds",
        "primary_methods",
        "access_policy",
        "split_normalization",
        "contract_encoder",
        "diagnostics",
        "optimization",
        "validation_selection",
        "abstention",
        "review_fraction",
        "expert_relative_costs",
        "resource_profiles",
        "streaming",
        "numerics",
        "metric_schema_version",
        "output_schemas",
        "gate",
        "resume",
        "determinism",
        "authorization",
    }
)


@dataclass(frozen=True, slots=True)
class V5PilotConfig:
    path: Path
    payload: Mapping[str, Any]
    config_sha256: str
    preregistration_sha256: str

    @property
    def specification_version(self) -> str:
        return str(self.payload["pilot_specification_version"])

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["primary_methods"])

    @property
    def experts(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["required_experts"])

    @property
    def split_normalization(self) -> Mapping[str, str]:
        return {
            str(key): str(value)
            for key, value in self.payload["split_normalization"].items()
        }

    def resolve(self, field: str) -> Path:
        value = Path(str(self.payload[field])).expanduser()
        if not value.is_absolute():
            value = (self.path.parents[3] / value).resolve()
        return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required V5 registry is empty: {path}")
    return rows


def load_v5_config(path: Path) -> V5PilotConfig:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V5 pilot config must decode to a mapping")
    unknown = set(payload) - CONFIG_FIELDS
    missing = CONFIG_FIELDS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"V5 config field mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if payload["schema_version"] != "coregraph_saved_output_pilot_config_v5.2":
        raise ValueError("V5 config schema version is invalid")
    if payload["metric_schema_version"] != METRIC_SCHEMA_VERSION:
        raise ValueError("V5 metric schema version is invalid or superseded")
    if tuple(payload["primary_methods"]) != PRIMARY_METHODS:
        raise ValueError("V5 primary method set/order is not frozen")
    if tuple(payload["required_experts"]) != EXPERT_ORDER:
        raise ValueError("V5 expert set/order is not frozen")
    preregistration = Path(str(payload["preregistration_path"])).expanduser()
    if not preregistration.is_absolute():
        preregistration = (resolved.parents[3] / preregistration).resolve()
    observed_preregistration = stable_file_sha256(preregistration)
    declared = str(payload["preregistration_sha256"])
    if observed_preregistration != declared:
        raise ValueError(
            "V5 preregistration hash mismatch: "
            f"expected {declared}, observed {observed_preregistration}"
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return V5PilotConfig(
        path=resolved,
        payload=dict(payload),
        config_sha256=stable_sha256(json.loads(canonical)),
        preregistration_sha256=declared,
    )


def stable_file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(row: Mapping[str, str], key: str) -> int:
    value = str(row.get(key, ""))
    if not value:
        raise ValueError(f"V5 registry field {key!r} is empty")
    return int(value)


def load_v5_surface(
    config: V5PilotConfig,
    *,
    code_sha: str,
    evidence_cache: Path,
) -> tuple[tuple[V5BaseArtifact, ...], tuple[V5ScenarioMaterialization, ...]]:
    """Load and fully validate the exact committed 180/60/540 identity surface."""

    base_rows = _read_csv(config.resolve("base_artifact_registry"))
    scenario_rows = _read_csv(config.resolve("scenario_registry"))
    binding_rows = _read_csv(config.resolve("binding_registry"))
    member_index_path = Path(str(config.payload["member_index"])).expanduser()
    if not member_index_path.is_absolute():
        member_index_path = (evidence_cache.resolve() / member_index_path).resolve()
    index = MemberIndex.from_csv(member_index_path)
    required_datasets = tuple(str(value) for value in config.payload["required_datasets"])
    required_protocols = tuple(str(value) for value in config.payload["required_protocols"])
    required_seeds = tuple(int(value) for value in config.payload["required_provider_seeds"])
    expected_base = len(required_datasets) * len(required_protocols) * len(EXPERT_ORDER) * len(
        required_seeds
    )
    expected_scenarios = len(required_datasets) * len(required_protocols) * len(
        required_seeds
    )
    if (len(base_rows), len(scenario_rows), len(binding_rows)) != (
        expected_base,
        expected_scenarios,
        expected_scenarios * 9,
    ):
        raise ValueError(
            "V5 registry cardinality mismatch: "
            f"{len(base_rows)}/{len(scenario_rows)}/{len(binding_rows)}"
        )
    archive_hashes = {
        str(key): str(value) for key, value in config.payload["archive_hashes"].items()
    }
    artifacts: list[V5BaseArtifact] = []
    for row in base_rows:
        member = index.locate(row["dataset"], row["protocol"], row["expert"], int(row["seed"]))
        if (
            member.archive_name != row["archive"]
            or member.member_name != row["member"]
            or member.member_sha256 != row["member_sha256"]
        ):
            raise ValueError("V5 base registry conflicts with the canonical member index")
        if archive_hashes.get(member.archive_name) != row["archive_expected_sha256"]:
            raise ValueError("V5 base registry conflicts with the frozen archive hash map")
        checks = (
            member.coordinate_verified,
            member.row_order_verified,
            member.chronology_verified,
            member.provider_alignment_verified,
        )
        if checks != (True, True, True, True):
            raise ValueError("V5 member lacks required structural verification")
        if member.duplicate_identifier_count not in {None, 0}:
            raise ValueError("V5 member index reports duplicate identifiers")
        artifacts.append(
            V5BaseArtifact(
                dataset=row["dataset"],
                protocol=row["protocol"],
                expert=row["expert"],
                provider_seed=int(row["seed"]),
                fold=row["fold"],
                base_coordinate_id=row["base_coordinate_id"],
                base_artifact_hash=row["base_artifact_hash"],
                archive_name=member.archive_name,
                archive_sha256=row["archive_expected_sha256"],
                member_name=member.member_name,
                member_sha256=member.member_sha256,
                schema_version=member.schema_version or "RB09V3_PREDICTION_CSV_V1",
                score_semantics="probability_[0,1]",
                row_key_semantics=("dataset", "provider_seed", "protocol", "split", "node_id"),
                split_token_semantics=config.split_normalization,
                label_known_semantics="provider_boolean_unknown_excluded",
                provenance_status=row["status"],
                row_count=member.row_count or _integer(row, "row_count"),
                label_known_count=member.label_known_count
                if member.label_known_count is not None
                else _integer(row, "label_known_count"),
            )
        )
    artifact_by_coordinate = {item.base_coordinate_id: item for item in artifacts}
    if len(artifact_by_coordinate) != expected_base:
        raise ValueError("duplicate V5 base coordinate identity")
    expected_grid = {
        (dataset, protocol, expert, seed)
        for dataset in required_datasets
        for protocol in required_protocols
        for expert in EXPERT_ORDER
        for seed in required_seeds
    }
    if {item.logical_key for item in artifacts} != expected_grid:
        raise ValueError("V5 base artifacts do not form the frozen identity grid")

    bindings_by_scenario: dict[str, list[V5ScenarioBinding]] = defaultdict(list)
    seen_bindings: set[str] = set()
    for row in binding_rows:
        if row["binding_id"] in seen_bindings:
            raise ValueError("duplicate V5 binding ID")
        seen_bindings.add(row["binding_id"])
        role = row["role"]
        policy = row["label_access"]
        expected_policy = (
            "SOURCE_LABELS_ALLOWED" if role == "source" else "EVALUATION_ONLY_AFTER_FREEZE"
        )
        if policy != expected_policy:
            raise ValueError("V5 binding label policy is invalid")
        binding = V5ScenarioBinding(
            binding_id=row["binding_id"],
            scenario_id=row["scenario_id"],
            base_coordinate_id=row["base_coordinate_id"],
            protocol=row["protocol"],
            expert=row["expert"],
            provider_seed=int(row["seed"]),
            role=role,
            source_environment_id=(
                f"{row['dataset'] if 'dataset' in row else artifact_by_coordinate[row['base_coordinate_id']].dataset}"
                f":{row['protocol']}:seed{row['seed']}" if role == "source" else None
            ),
            permitted_splits=tuple(row["permitted_splits"].split(";")),
            label_access_policy=policy,
        )
        artifact = artifact_by_coordinate.get(binding.base_coordinate_id)
        if artifact is None:
            raise ValueError("V5 binding references an absent base coordinate")
        if (
            artifact.protocol != binding.protocol
            or artifact.expert != binding.expert
            or artifact.provider_seed != binding.provider_seed
        ):
            raise ValueError("V5 binding conflicts with its base artifact")
        bindings_by_scenario[binding.scenario_id].append(binding)

    definitions: list[V5ScenarioDefinition] = []
    seen_scenarios: set[str] = set()
    for row in scenario_rows:
        if row["scenario_id"] in seen_scenarios:
            raise ValueError("duplicate V5 scenario ID")
        seen_scenarios.add(row["scenario_id"])
        definitions.append(
            V5ScenarioDefinition(
                scenario_id=row["scenario_id"],
                dataset=row["dataset"],
                provider_seed=int(row["seed"]),
                fold=row["fold"],
                target_protocol=row["target_protocol"],
                source_protocols=tuple(row["source_protocols"].split(";")),  # type: ignore[arg-type]
                access_regime=row["access_regime"],
                feasible_experts=tuple(row["feasible_experts"].split(";")),
                resource_profile=row["resource_mask"],
                review_fraction=float(config.payload["review_fraction"]),
            )
        )
    materializations: list[V5ScenarioMaterialization] = []
    for definition in definitions:
        scenario_bindings = tuple(bindings_by_scenario.get(definition.scenario_id, ()))
        source = tuple(item for item in scenario_bindings if item.role == "source")
        target = tuple(item for item in scenario_bindings if item.role == "target")
        identities = [
            artifact_by_coordinate[item.base_coordinate_id].identity_dict()
            for item in sorted(
                scenario_bindings,
                key=lambda value: (value.role, value.protocol, EXPERT_ORDER.index(value.expert)),
            )
        ]
        fingerprint = stable_sha256(
            {
                "scenario": {
                    "scenario_id": definition.scenario_id,
                    "dataset": definition.dataset,
                    "provider_seed": definition.provider_seed,
                    "fold": definition.fold,
                    "target_protocol": definition.target_protocol,
                    "source_protocols": definition.source_protocols,
                    "access_regime": definition.access_regime,
                    "resource_profile": definition.resource_profile,
                    "methods": definition.methods,
                },
                "artifacts": identities,
                "config_sha256": config.config_sha256,
                "code_sha": code_sha,
                "preregistration_sha256": config.preregistration_sha256,
            }
        )
        materializations.append(
            V5ScenarioMaterialization(
                definition=definition,
                source_bindings=source,
                target_bindings=target,
                artifacts_by_coordinate=artifact_by_coordinate,
                scenario_fingerprint=fingerprint,
            )
        )
    expected_scenario_grid = {
        (dataset, protocol, seed)
        for dataset in required_datasets
        for protocol in required_protocols
        for seed in required_seeds
    }
    if {
        (item.definition.dataset, item.definition.target_protocol, item.definition.provider_seed)
        for item in materializations
    } != expected_scenario_grid:
        raise ValueError("V5 scenarios do not form the frozen 60-cell grid")
    if Counter(item.definition.dataset for item in materializations) != Counter(
        {dataset: len(required_protocols) * len(required_seeds) for dataset in required_datasets}
    ):
        raise ValueError("V5 dataset scenario balance is invalid")
    return tuple(artifacts), tuple(materializations)


def build_pilot_coordinates(
    materializations: Sequence[V5ScenarioMaterialization],
    config: V5PilotConfig,
    *,
    effective_execution_config_sha256: str,
) -> tuple[PilotCoordinate, ...]:
    coordinates = tuple(
        PilotCoordinate(
            dataset=scenario.definition.dataset,
            target_protocol=scenario.definition.target_protocol,
            provider_seed=scenario.definition.provider_seed,
            method=method,
            pilot_specification_version=config.specification_version,
            scenario_id=scenario.definition.scenario_id,
            scenario_fingerprint=scenario.scenario_fingerprint,
            effective_execution_config_sha256=effective_execution_config_sha256,
        )
        for scenario in materializations
        for method in PRIMARY_METHODS
    )
    keys = [item.key for item in coordinates]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot coordinate primary keys are not unique")
    return coordinates


def validate_archive_surface(
    artifacts: Sequence[V5BaseArtifact],
    *,
    evidence_cache: Path,
    verify_members: bool,
) -> Mapping[str, Any]:
    records = {item.archive_name: item.archive_sha256 for item in artifacts}
    store = ArchiveStore(evidence_cache, records)
    archives = []
    for archive_name in sorted(records):
        members = store.list_members(archive_name)
        expected_members = {item.member_name for item in artifacts if item.archive_name == archive_name}
        if not expected_members.issubset(set(members)):
            raise ValueError(f"canonical archive {archive_name} is missing indexed members")
        archives.append({"archive": archive_name, "member_count": len(expected_members), "status": "VERIFIED"})
    verified_members = 0
    if verify_members:
        for artifact in artifacts:
            store.verify_member(
                artifact.archive_name,
                artifact.member_name,
                expected_sha256=artifact.member_sha256,
            )
            verified_members += 1
    return {
        "schema": "coregraph_v5_archive_surface_validation_v1",
        "archive_count": len(archives),
        "member_identity_count": len(artifacts),
        "member_checksum_verified": verified_members,
        "archives": archives,
        "permanent_extractions": 0,
        "training_performed": False,
        "target_labels_loaded": False,
    }
