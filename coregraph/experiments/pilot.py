"""Seed-bound saved-output pilot and honest baseline adapters."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from coregraph.contracts.axes import ContractRole, ReviewMode
from coregraph.contracts.contract import DeploymentContract
from coregraph.evaluation.metrics import (
    binary_metrics,
    budget_curve_auc,
    recall_at_k,
)
from coregraph.method import CoReGraph
from coregraph.objectives.classification import binary_cross_entropy_values
from coregraph.objectives.composite import CompositeObjective, ObjectiveWeights
from coregraph.objectives.scores import ScoreType, validate_numpy_scores
from coregraph.routing.abstention import (
    apply_frozen_abstention_decision,
    area_under_risk_coverage_curve,
    select_grouped_abstention_threshold,
)
from coregraph.routing.stability import consistency_penalty
from coregraph.tasks.base import align_prediction_rows
from coregraph.utils.seeding import seed_everything


@dataclass(frozen=True)
class PredictionArtifact:
    expert_id: str
    dataset: str
    task: str
    prediction_unit: str
    contract_coordinate_hash: str
    environment_id: str
    seed: int
    fold: str
    path: Path
    checksum: str
    config_hash: str
    code_hash: str
    contract_role: str
    deployment_contract: DeploymentContract
    expert_alias_of: str = ""
    expert_available: bool = True
    availability_reason_codes: tuple[str, ...] = ("available",)
    compute_cost: float = 0.0
    score_type: ScoreType = ScoreType.PROBABILITY

    def __post_init__(self) -> None:
        def valid_hex(value: str, lengths: tuple[int, ...]) -> bool:
            return len(value) in lengths and all(
                character in "0123456789abcdef" for character in value
            )

        if self.contract_coordinate_hash != self.deployment_contract.coordinate_hash:
            raise ValueError("artifact contract coordinate hash mismatch")
        if self.environment_id != self.deployment_contract.environment_id:
            raise ValueError("artifact environment ID mismatch")
        if self.contract_role != self.deployment_contract.role.value:
            raise ValueError("artifact contract role mismatch")
        if self.seed < 0:
            raise ValueError("artifact seed must be non-negative")
        if self.compute_cost < 0:
            raise ValueError("artifact compute cost cannot be negative")
        if not self.expert_id or not self.dataset or not self.task or not self.fold:
            raise ValueError("artifact identity fields cannot be empty")
        if not valid_hex(self.checksum, (64,)):
            raise ValueError("artifact checksum must be lowercase SHA-256")
        if not valid_hex(self.config_hash, (64,)):
            raise ValueError("artifact config hash must be lowercase SHA-256")
        if not valid_hex(self.code_hash, (40, 64)):
            raise ValueError("artifact code hash must be lowercase Git/SHA hash")
        if not self.availability_reason_codes:
            raise ValueError("artifact availability reason codes are required")
        if self.expert_available and self.availability_reason_codes != (
            "available",
        ):
            raise ValueError("available artifact must have only the available reason")
        if (
            not self.expert_available
            and "available" in self.availability_reason_codes
        ):
            raise ValueError("unavailable artifact cannot have the available reason")
        if self.expert_alias_of == self.expert_id:
            raise ValueError("artifact cannot alias itself")

    @property
    def canonical_expert_id(self) -> str:
        return self.expert_alias_of or self.expert_id

    @property
    def expert_prediction_seed(self) -> int:
        return self.seed

    @property
    def group_key(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.dataset,
            self.task,
            self.contract_coordinate_hash,
            self.environment_id,
            self.seed,
            self.fold,
        )


@dataclass(frozen=True)
class SavedSourceGroup:
    contract: DeploymentContract
    scores: Mapping[str, np.ndarray]
    labels: np.ndarray
    splits: np.ndarray
    availability: Mapping[str, np.ndarray]
    expert_costs: Mapping[str, float]

    def __post_init__(self) -> None:
        experts = set(self.scores)
        if experts != set(self.availability) or experts != set(self.expert_costs):
            raise ValueError(
                "source scores, availability, and expert costs need identical experts"
            )
        n = len(self.labels)
        if len(self.splits) != n:
            raise ValueError("source labels and split rows must align")
        for expert in experts:
            if len(self.scores[expert]) != n or len(self.availability[expert]) != n:
                raise ValueError(f"source expert {expert} rows do not align")
            validate_numpy_scores(
                np.asarray(self.scores[expert]),
                ScoreType.PROBABILITY,
            )
            if self.expert_costs[expert] < 0:
                raise ValueError("source expert costs cannot be negative")
        allowed = {"train", "validation"}
        if not set(np.unique(self.splits)).issubset(allowed):
            raise ValueError("source router groups may contain train/validation rows only")


@dataclass(frozen=True)
class BaselinePrediction:
    scores: np.ndarray
    abstention_probability: np.ndarray
    abstain: np.ndarray
    forced_abstention: np.ndarray
    expected_compute: np.ndarray
    abstention_threshold: float | None
    abstention_threshold_provenance: str
    abstention_capacity: float | None
    abstention_cost: float
    execution_status: MethodExecutionStatus | str
    learned: bool = False
    adapter: str = ""
    offline_oracle: bool = False
    diagnostic_only: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = validate_numpy_scores(
            np.asarray(self.scores),
            ScoreType.PROBABILITY,
        )
        abstention = np.asarray(self.abstention_probability, dtype=float)
        decision = np.asarray(self.abstain, dtype=bool)
        forced = np.asarray(self.forced_abstention, dtype=bool)
        compute = np.asarray(self.expected_compute, dtype=float)
        if (
            scores.ndim != 1
            or abstention.shape != scores.shape
            or decision.shape != scores.shape
            or forced.shape != scores.shape
            or compute.shape != scores.shape
        ):
            raise ValueError(
                "baseline score, abstention decision, and compute rows must align"
            )
        if (
            not np.isfinite(abstention).all()
            or np.any(abstention < 0)
            or np.any(abstention > 1)
        ):
            raise ValueError("baseline abstention probabilities must lie in [0,1]")
        if not np.isfinite(compute).all() or np.any(compute < 0):
            raise ValueError("baseline compute must be finite and non-negative")
        if self.abstention_threshold is not None and np.isnan(
            self.abstention_threshold
        ):
            raise ValueError("abstention thresholds cannot be NaN")
        if not self.abstention_threshold_provenance:
            raise ValueError("abstention threshold provenance is required")
        if (
            self.abstention_capacity is not None
            and not 0 <= self.abstention_capacity <= 1
        ):
            raise ValueError("baseline abstention capacity must lie in [0,1]")
        if self.abstention_cost < 0:
            raise ValueError("baseline abstention cost cannot be negative")
        status = MethodExecutionStatus(self.execution_status)
        object.__setattr__(self, "execution_status", status)
        if np.any(forced & ~decision):
            raise ValueError("forced abstentions must be present in frozen decision")
        if status is MethodExecutionStatus.ABSTAIN_ONLY and not decision.all():
            raise ValueError("ABSTAIN_ONLY requires every decision to abstain")
        if self.diagnostic_only and not self.offline_oracle:
            raise ValueError("diagnostic-only predictions must be offline oracles")

    @property
    def ranking_eligible(self) -> bool:
        return self.execution_status in {
            MethodExecutionStatus.EXECUTABLE,
            MethodExecutionStatus.EXECUTABLE_WITH_FALLBACK,
        }


class MethodExecutionStatus(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    EXECUTABLE_WITH_FALLBACK = "EXECUTABLE_WITH_FALLBACK"
    ABSTAIN_ONLY = "ABSTAIN_ONLY"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PilotAblation(str, Enum):
    FULL = "full"
    NO_CONTRACT = "no_contract"
    ATOMIC_CONTRACT = "atomic_contract"
    NO_REGRET = "no_regret"
    NO_BUDGET = "no_budget"
    NO_RESOURCE_MASK = "no_resource_mask"
    NO_STABILITY = "no_stability"
    NO_ABSTENTION = "no_abstention"
    NO_DIAGNOSTICS = "no_diagnostics"


@dataclass(frozen=True)
class SavedRouterPrediction:
    scores: np.ndarray
    selected_experts: np.ndarray
    routing_weights: np.ndarray
    abstention_probability: np.ndarray
    abstain: np.ndarray
    expected_compute: np.ndarray
    forced_abstention: np.ndarray
    perturbation_flip_rate: float
    ablation: PilotAblation
    source_train_examples: int
    source_validation_examples: int
    abstention_threshold: float
    abstention_threshold_fitted_on: str
    source_abstention_capacities: Mapping[int, float | None]
    target_abstention_capacity: float | None
    abstention_cost: float
    expert_prediction_seed: int
    router_training_seed: int
    source_fit_hash: str
    early_stopping_source_only: bool = True


def derive_router_seed(expert_prediction_seed: int, method: str) -> int:
    if expert_prediction_seed < 0 or not method:
        raise ValueError("router seed derivation requires a non-negative seed and method")
    digest = hashlib.sha256(
        f"coregraph-router-v3:{expert_prediction_seed}:{method}".encode()
    ).digest()
    derived = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return derived if derived != expert_prediction_seed else (derived + 1) & 0x7FFFFFFF


def source_review_k_by_group(
    contracts: Sequence[DeploymentContract],
    group_indices: np.ndarray,
) -> dict[int, int | None]:
    groups = tuple(int(value) for value in np.unique(group_indices))
    if groups != tuple(range(len(contracts))):
        raise ValueError("source contracts and group indices must align exactly")
    constrained_modes = {
        contract.budget.review_mode
        for contract in contracts
        if contract.budget.review_mode is not ReviewMode.UNCONSTRAINED_RANKING
    }
    if len(constrained_modes) > 1:
        raise ValueError("mixed constrained review modes are unsupported")
    output: dict[int, int | None] = {}
    for group, contract in enumerate(contracts):
        count = int(np.sum(group_indices == group))
        if contract.budget.review_mode is ReviewMode.UNCONSTRAINED_RANKING:
            output[group] = None
        elif contract.budget.review_mode is ReviewMode.FRACTION:
            fraction = contract.budget.review_fraction
            if fraction is None:
                raise ValueError("fractional source budget is missing its fraction")
            output[group] = min(count, max(1, int(np.ceil(fraction * count))))
        elif contract.budget.review_mode is ReviewMode.FIXED_K:
            fixed_k = contract.budget.fixed_k
            if fixed_k is None:
                raise ValueError("fixed source budget is missing K")
            output[group] = min(count, fixed_k)
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError("unsupported source review mode")
    return output


def source_abstention_capacity_by_group(
    contracts: Sequence[DeploymentContract],
    group_indices: np.ndarray,
) -> dict[int, float | None]:
    groups = tuple(int(value) for value in np.unique(group_indices))
    if groups != tuple(range(len(contracts))):
        raise ValueError("source contracts and group indices must align exactly")
    return {
        group: contract.budget.abstention_capacity
        for group, contract in enumerate(contracts)
    }


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def discover_prediction_manifests(roots: Sequence[str | Path]) -> list[Path]:
    manifests: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.name.endswith(".json"):
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(root.rglob("prediction_manifest.json"))
    return sorted(set(path.resolve() for path in manifests))


def load_prediction_artifacts(manifests: Sequence[Path]) -> list[PredictionArtifact]:
    artifacts: list[PredictionArtifact] = []
    required = {
        "expert_id",
        "dataset",
        "task",
        "prediction_unit",
        "contract_coordinate_hash",
        "environment_id",
        "expert_prediction_seed",
        "fold",
        "prediction_path",
        "prediction_checksum",
        "config_hash",
        "code_hash",
        "contract_role",
        "deployment_contract",
        "expert_available",
        "availability_reason_codes",
        "compute_cost",
        "score_type",
    }
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"{path} missing prediction manifest keys {missing}")
        prediction_path = Path(payload["prediction_path"])
        if not prediction_path.is_absolute():
            prediction_path = (path.parent / prediction_path).resolve()
        if not prediction_path.is_file():
            raise ValueError(f"{path} prediction file is missing: {prediction_path}")
        actual_checksum = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if actual_checksum != str(payload["prediction_checksum"]):
            raise ValueError(f"{path} prediction checksum mismatch")
        contract = DeploymentContract.from_dict(payload["deployment_contract"])
        if str(payload["contract_coordinate_hash"]) != contract.coordinate_hash:
            raise ValueError(f"{path} contract coordinate hash mismatch")
        if str(payload["environment_id"]) != contract.environment_id:
            raise ValueError(f"{path} environment ID mismatch")
        artifacts.append(
            PredictionArtifact(
                expert_id=str(payload["expert_id"]),
                expert_alias_of=str(payload.get("expert_alias_of", "")),
                dataset=str(payload["dataset"]),
                task=str(payload["task"]),
                prediction_unit=str(payload["prediction_unit"]),
                contract_coordinate_hash=str(
                    payload["contract_coordinate_hash"]
                ),
                environment_id=str(payload["environment_id"]),
                seed=int(payload["expert_prediction_seed"]),
                fold=str(payload["fold"]),
                path=prediction_path,
                checksum=str(payload["prediction_checksum"]),
                config_hash=str(payload["config_hash"]),
                code_hash=str(payload["code_hash"]),
                contract_role=str(payload["contract_role"]),
                deployment_contract=contract,
                expert_available=bool(payload["expert_available"]),
                availability_reason_codes=tuple(
                    str(value)
                    for value in payload["availability_reason_codes"]
                ),
                compute_cost=float(payload["compute_cost"]),
                score_type=ScoreType(payload["score_type"]),
            )
        )
    return artifacts


def validate_artifact_groups(
    artifacts: Sequence[PredictionArtifact],
    *,
    expected_experts: Sequence[str],
    expected_seeds: Sequence[int],
    expected_datasets: Sequence[str] | None = None,
    expected_target_contracts: Sequence[str] | None = None,
) -> dict[
    tuple[str, str, str, str, int, str],
    tuple[PredictionArtifact, ...],
]:
    if not artifacts:
        raise ValueError("no prediction artifacts supplied")
    expected_expert_set = set(expected_experts)
    expected_seed_set = set(expected_seeds)
    if len(expected_expert_set) != len(tuple(expected_experts)):
        raise ValueError("expected experts must be unique canonical IDs")
    groups: dict[
        tuple[str, str, str, str, int, str],
        list[PredictionArtifact],
    ] = defaultdict(list)
    for artifact in artifacts:
        if artifact.seed not in expected_seed_set:
            raise ValueError(f"unexpected seed {artifact.seed}")
        if artifact.expert_alias_of:
            raise ValueError(
                "expert alias cannot be counted as an independent expert"
            )
        groups[artifact.group_key].append(artifact)
    for key, group in groups.items():
        canonical = [artifact.canonical_expert_id for artifact in group]
        if len(canonical) != len(set(canonical)):
            raise ValueError(f"duplicate expert-seed artifacts in group {key}")
        if set(canonical) != expected_expert_set:
            missing = sorted(expected_expert_set - set(canonical))
            extra = sorted(set(canonical) - expected_expert_set)
            raise ValueError(
                f"group {key} does not contain exactly one artifact per expert; "
                f"missing={missing} extra={extra}"
            )
    by_contract_fold: dict[
        tuple[str, str, str, str, str],
        set[int],
    ] = defaultdict(set)
    for dataset, task, coordinate, environment, seed, fold in groups:
        by_contract_fold[
            (dataset, task, coordinate, environment, fold)
        ].add(seed)
    for contract_fold_key, seeds in by_contract_fold.items():
        if seeds != expected_seed_set:
            raise ValueError(
                f"missing seeds for {contract_fold_key}: "
                f"{sorted(expected_seed_set - seeds)}"
            )
    if expected_datasets is not None:
        expected_dataset_set = set(expected_datasets)
        actual_dataset_set = {artifact.dataset for artifact in artifacts}
        if actual_dataset_set != expected_dataset_set:
            raise ValueError(
                "prediction manifests do not contain exactly the required "
                f"datasets; expected={sorted(expected_dataset_set)} "
                f"actual={sorted(actual_dataset_set)}"
            )
    if expected_target_contracts is not None:
        required_contracts = set(expected_target_contracts)
        target_coverage: dict[str, set[str]] = defaultdict(set)
        for artifact in artifacts:
            if artifact.contract_role == ContractRole.TARGET.value:
                target_coverage[artifact.dataset].add(
                    artifact.deployment_contract.contract_id
                )
        datasets = (
            set(expected_datasets)
            if expected_datasets is not None
            else set(target_coverage)
        )
        for dataset in datasets:
            if target_coverage[dataset] != required_contracts:
                raise ValueError(
                    f"target contract coverage for {dataset} is incomplete; "
                    f"expected={sorted(required_contracts)} "
                    f"actual={sorted(target_coverage[dataset])}"
                )
    return {
        key: tuple(sorted(group, key=lambda artifact: artifact.expert_id))
        for key, group in sorted(groups.items())
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def align_artifact_group(
    artifacts: Sequence[PredictionArtifact],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not artifacts:
        raise ValueError("no prediction artifacts supplied")
    if len({artifact.group_key for artifact in artifacts}) != 1:
        raise ValueError(
            "prediction alignment cannot pool dataset task contract seed or fold"
        )
    if len({artifact.prediction_unit for artifact in artifacts}) != 1:
        raise ValueError("prediction alignment cannot pool prediction units")
    if {artifact.score_type for artifact in artifacts} != {
        ScoreType.PROBABILITY
    }:
        raise ValueError(
            "saved-output probability pilot requires PROBABILITY artifacts"
        )
    expert_ids = [artifact.canonical_expert_id for artifact in artifacts]
    if len(expert_ids) != len(set(expert_ids)):
        raise ValueError("duplicate expert-seed artifacts")
    unit = artifacts[0].prediction_unit
    id_column = f"{unit}_id"
    rows_by_expert: dict[str, list[dict[str, str]]] = {}
    for artifact in artifacts:
        rows = _read_rows(artifact.path)
        for row in rows:
            if row.get("expert_id") not in {artifact.expert_id, ""}:
                raise ValueError("prediction row expert ID mismatch")
        rows_by_expert[artifact.canonical_expert_id] = rows
    ids, scores, labels = align_prediction_rows(
        rows_by_expert,
        id_column=id_column,
    )
    scores = {
        expert: validate_numpy_scores(values, ScoreType.PROBABILITY)
        for expert, values in scores.items()
    }
    reference = rows_by_expert[expert_ids[0]]
    reference_split = {
        str(row[id_column]): str(row["split"]) for row in reference
    }
    for expert, rows in rows_by_expert.items():
        split_map = {str(row[id_column]): str(row["split"]) for row in rows}
        if split_map != reference_split:
            raise ValueError(f"split mismatch after alignment for expert {expert}")
    splits = np.asarray([reference_split[str(identifier)] for identifier in ids])
    return ids, scores, labels, splits


def _stack_source_validation(
    source_groups: Sequence[SavedSourceGroup],
    experts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_masks: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    offset = 0
    for group in source_groups:
        mask = np.asarray(group.splits) == "validation"
        if not mask.any():
            raise ValueError(
                f"source contract {group.contract.contract_id} has no validation rows"
            )
        matrix = np.column_stack([group.scores[name] for name in experts])[mask]
        matrices.append(matrix)
        labels.append((np.asarray(group.labels)[mask] == 1).astype(float))
        availability.append(
            np.column_stack(
                [group.availability[name] for name in experts]
            )[mask].astype(bool)
        )
        group_mask = np.arange(offset, offset + len(matrix))
        group_masks.append(group_mask)
        offset += len(matrix)
    return (
        np.concatenate(matrices),
        np.concatenate(labels),
        group_masks,
        np.concatenate(availability),
    )


def _balanced_brier(
    prediction: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[np.ndarray],
) -> float:
    return float(
        np.mean(
            [
                np.mean((prediction[index] - labels[index]) ** 2)
                for index in groups
            ]
        )
    )


def _convex_validation_weights(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[np.ndarray],
    availability: np.ndarray,
) -> np.ndarray:
    experts = matrix.shape[1]

    def objective(weights: np.ndarray) -> float:
        masked = availability * weights[None, :]
        denominator = masked.sum(axis=1)
        prediction = np.divide(
            (matrix * masked).sum(axis=1),
            denominator,
            out=np.zeros(len(matrix)),
            where=denominator > 0,
        )
        return _balanced_brier(prediction, labels, groups)

    result = minimize(
        objective,
        np.full(experts, 1 / experts),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * experts,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"source-validation convex mixture failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0, 1)
    return weights / weights.sum()


def _learned_gate_baseline(
    source_groups: Sequence[SavedSourceGroup],
    experts: Sequence[str],
    target_matrix: np.ndarray,
    target_mask: np.ndarray,
    *,
    atomic: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    trust: list[np.ndarray] = []
    contract_ids = sorted(
        {group.contract.contract_id for group in source_groups}
    )
    for group in source_groups:
        mask = np.asarray(group.splits) == "train"
        matrix = np.column_stack([group.scores[name] for name in experts])[mask]
        availability = np.column_stack(
            [group.availability[name] for name in experts]
        )[mask].astype(bool)
        labels = (np.asarray(group.labels)[mask] == 1).astype(float)
        feasible_rows = availability.any(axis=1)
        if not feasible_rows.any():
            continue
        matrix = matrix[feasible_rows]
        availability = availability[feasible_rows]
        labels = labels[feasible_rows]
        features = [
            matrix,
            np.std(matrix, axis=1, keepdims=True),
        ]
        if atomic:
            one_hot = np.zeros((len(matrix), len(contract_ids)))
            one_hot[:, contract_ids.index(group.contract.contract_id)] = 1
            features.append(one_hot)
        rows.append(np.column_stack(features))
        errors = np.where(
            availability,
            np.abs(matrix - labels[:, None]),
            np.inf,
        )
        trust.append(errors.argmin(axis=1))
    if not rows:
        raise ValueError("learned gate baseline has no feasible source rows")
    features = np.concatenate(rows)
    target_features = [
        target_matrix,
        np.std(target_matrix, axis=1, keepdims=True),
    ]
    if atomic:
        target_features.append(
            np.zeros((len(target_matrix), len(contract_ids)))
        )
    target_features_array = np.column_stack(target_features)
    target_values = np.concatenate(trust)
    if len(np.unique(target_values)) < 2:
        weights = np.full((len(target_matrix), len(experts)), 1 / len(experts))
    else:
        gate = LogisticRegression(
            max_iter=1000,
            random_state=seed,
        )
        gate.fit(features, target_values)
        probabilities = gate.predict_proba(target_features_array)
        weights = np.zeros((len(target_matrix), len(experts)))
        weights[:, gate.classes_.astype(int)] = probabilities
    masked_weights = weights * target_mask
    denominator = masked_weights.sum(axis=1, keepdims=True)
    masked_weights = np.divide(
        masked_weights,
        denominator,
        out=np.zeros_like(masked_weights),
        where=denominator > 0,
    )
    return (masked_weights * target_matrix).sum(axis=1), masked_weights


def _group_vector(
    group_rows: Sequence[np.ndarray],
    total: int,
) -> np.ndarray:
    output = np.empty(total, dtype=int)
    for group, rows in enumerate(group_rows):
        output[rows] = group
    return output


def _execution_status(
    executable_rows: np.ndarray,
    abstain: np.ndarray,
    *,
    fallback_used: bool,
) -> MethodExecutionStatus:
    if not executable_rows.all():
        return MethodExecutionStatus.RESOURCE_BLOCKED
    if abstain.all():
        return MethodExecutionStatus.ABSTAIN_ONLY
    if fallback_used:
        return MethodExecutionStatus.EXECUTABLE_WITH_FALLBACK
    return MethodExecutionStatus.EXECUTABLE


def _baseline_prediction(
    *,
    scores: np.ndarray,
    abstention_probability: np.ndarray,
    abstain: np.ndarray,
    forced_abstention: np.ndarray,
    expected_compute: np.ndarray,
    target_capacity: float | None,
    abstention_cost: float,
    executable_rows: np.ndarray,
    fallback_used: bool = False,
    threshold: float | None = None,
    threshold_provenance: str = "forced_unavailability_only",
    learned: bool = False,
    adapter: str = "",
    offline_oracle: bool = False,
    diagnostic_only: bool = False,
    details: Mapping[str, Any] | None = None,
) -> BaselinePrediction:
    return BaselinePrediction(
        scores=scores,
        abstention_probability=abstention_probability,
        abstain=abstain,
        forced_abstention=forced_abstention,
        expected_compute=expected_compute,
        abstention_threshold=threshold,
        abstention_threshold_provenance=threshold_provenance,
        abstention_capacity=target_capacity,
        abstention_cost=abstention_cost,
        execution_status=_execution_status(
            executable_rows,
            abstain,
            fallback_used=fallback_used,
        ),
        learned=learned,
        adapter=adapter,
        offline_oracle=offline_oracle,
        diagnostic_only=diagnostic_only,
        details={} if details is None else details,
    )


def _select_mowst_routing_threshold(
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    validation_groups: Sequence[np.ndarray],
    validation_availability: np.ndarray,
    *,
    feature_index: int,
    graph_index: int,
) -> float:
    confidence = np.abs(validation_matrix[:, feature_index] - 0.5)
    upper = np.nextafter(np.max(confidence), np.inf)
    lower = np.nextafter(np.min(confidence), -np.inf)
    candidates = np.concatenate(
        ([upper], np.unique(confidence), [lower])
    )
    best: tuple[float, float] | None = None
    rows = np.arange(len(validation_matrix))
    for threshold in candidates:
        choice = np.where(
            confidence >= threshold,
            feature_index,
            graph_index,
        )
        alternative = np.where(
            choice == feature_index,
            graph_index,
            feature_index,
        )
        chosen_available = validation_availability[rows, choice]
        alternative_available = validation_availability[rows, alternative]
        choice = np.where(
            chosen_available,
            choice,
            np.where(alternative_available, alternative, choice),
        )
        feasible = validation_availability[rows, choice]
        group_risks: list[float] = []
        for group_rows in validation_groups:
            keep = group_rows[feasible[group_rows]]
            if not len(keep):
                group_risks = []
                break
            prediction = validation_matrix[keep, choice[keep]]
            group_risks.append(
                float(np.mean((prediction - validation_labels[keep]) ** 2))
            )
        if not group_risks:
            continue
        record = (float(np.mean(group_risks)), float(threshold))
        if best is None or record < best:
            best = record
    if best is None:
        raise ValueError("Mowst-inspired source validation has no feasible threshold")
    return best[1]


def baseline_scores(
    source_groups: Sequence[SavedSourceGroup],
    *,
    target_contract: DeploymentContract,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
    target_expert_costs: Mapping[str, float],
    expert_prediction_seed: int,
    abstention_cost: float = 0.2,
) -> dict[str, BaselinePrediction]:
    """Fit every comparator on source contracts and freeze target decisions."""

    if not source_groups:
        raise ValueError("baseline fitting requires source contracts")
    experts = sorted(target_scores)
    if any(set(group.scores) != set(experts) for group in source_groups):
        raise ValueError("all source and target groups need the same experts")
    if (
        set(target_availability) != set(experts)
        or set(target_expert_costs) != set(experts)
    ):
        raise ValueError(
            "target availability and costs must declare every expert"
        )
    if abstention_cost < 0:
        raise ValueError("abstention cost cannot be negative")
    target_matrix = np.column_stack([target_scores[name] for name in experts])
    validate_numpy_scores(target_matrix, ScoreType.PROBABILITY)
    target_mask = np.column_stack(
        [np.asarray(target_availability[name], dtype=bool) for name in experts]
    )
    if target_matrix.shape != target_mask.shape:
        raise ValueError("target scores and availability must align")
    target_capacity = target_contract.budget.abstention_capacity
    (
        validation_matrix,
        validation_labels,
        validation_groups,
        validation_availability,
    ) = _stack_source_validation(source_groups, experts)
    validation_group_indices = _group_vector(
        validation_groups,
        len(validation_matrix),
    )
    validation_risks = np.asarray(
        [
            np.mean(
                [
                    (
                        np.mean(
                            (
                                validation_matrix[index, expert_index]
                                - validation_labels[index]
                            )
                            ** 2
                        )
                        if validation_availability[index, expert_index].all()
                        else np.inf
                    )
                    for index in validation_groups
                ]
            )
            for expert_index in range(len(experts))
        ]
    )
    if not np.isfinite(validation_risks).any():
        raise ValueError(
            "source-validation best expert requires one expert feasible "
            "across every source contract"
        )
    best = int(np.argmin(validation_risks))
    costs = np.asarray(
        [target_expert_costs[name] for name in experts],
        dtype=float,
    )
    output: dict[str, BaselinePrediction] = {}
    for index, name in enumerate(experts):
        forced = ~target_mask[:, index]
        method = f"expert:{name}"
        output[method] = _baseline_prediction(
            scores=target_matrix[:, index],
            abstention_probability=forced.astype(float),
            abstain=forced,
            forced_abstention=forced,
            expected_compute=target_mask[:, index].astype(float) * costs[index],
            target_capacity=target_capacity,
            abstention_cost=abstention_cost,
            executable_rows=target_mask[:, index],
            details={
                "expert_id": name,
                "router_training_seed": derive_router_seed(
                    expert_prediction_seed,
                    method,
                ),
            },
        )

    feasible_count = target_mask.sum(axis=1)
    executable_rows = feasible_count > 0
    forced = ~executable_rows
    feasible_sum = np.where(target_mask, target_matrix, 0.0).sum(axis=1)
    average = np.divide(
        feasible_sum,
        feasible_count,
        out=np.zeros(len(target_matrix)),
        where=executable_rows,
    )
    output["average_all_feasible"] = _baseline_prediction(
        scores=average,
        abstention_probability=forced.astype(float),
        abstain=forced,
        forced_abstention=forced,
        expected_compute=(target_mask * costs).sum(axis=1),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=executable_rows,
        fallback_used=not target_mask.all(),
        details={
            "router_training_seed": derive_router_seed(
                expert_prediction_seed,
                "average_all_feasible",
            )
        },
    )
    best_forced = ~target_mask[:, best]
    output["best_source_validation"] = _baseline_prediction(
        scores=target_matrix[:, best],
        abstention_probability=best_forced.astype(float),
        abstain=best_forced,
        forced_abstention=best_forced,
        expected_compute=target_mask[:, best].astype(float) * costs[best],
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=target_mask[:, best],
        details={
            "selected_expert": experts[best],
            "router_training_seed": derive_router_seed(
                expert_prediction_seed,
                "best_source_validation",
            ),
        },
    )
    convex_weights = _convex_validation_weights(
        validation_matrix,
        validation_labels,
        validation_groups,
        validation_availability,
    )
    masked_weights = target_mask * convex_weights[None, :]
    denominator = masked_weights.sum(axis=1, keepdims=True)
    masked_weights = np.divide(
        masked_weights,
        denominator,
        out=np.zeros_like(masked_weights),
        where=denominator > 0,
    )
    output["source_validation_convex_mixture"] = _baseline_prediction(
        scores=(masked_weights * target_matrix).sum(axis=1),
        abstention_probability=forced.astype(float),
        abstain=forced,
        forced_abstention=forced,
        expected_compute=(masked_weights * costs).sum(axis=1),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=executable_rows,
        fallback_used=not target_mask.all(),
        details={
            "weights": dict(zip(experts, convex_weights, strict=True)),
            "router_training_seed": derive_router_seed(
                expert_prediction_seed,
                "source_validation_convex_mixture",
            ),
        },
    )

    from models.graphsafe_v2 import confidence_scores

    validation_graphsafe_confidence = np.clip(
        confidence_scores(validation_matrix[:, best]),
        0,
        1,
    )
    source_capacities = {
        index: group.contract.budget.abstention_capacity
        for index, group in enumerate(source_groups)
    }
    graphsafe_threshold = select_grouped_abstention_threshold(
        torch.tensor(
            (validation_matrix[:, best] - validation_labels) ** 2,
            dtype=torch.float32,
        ),
        torch.tensor(1 - validation_graphsafe_confidence, dtype=torch.float32),
        torch.tensor(validation_group_indices, dtype=torch.long),
        capacities=source_capacities,
        abstention_cost_value=abstention_cost,
        forced_abstention=torch.tensor(
            ~validation_availability[:, best],
            dtype=torch.bool,
        ),
    )
    graphsafe_confidence = np.clip(
        confidence_scores(target_matrix[:, best]),
        0,
        1,
    )
    graphsafe_probability = np.maximum(
        best_forced.astype(float),
        1 - graphsafe_confidence,
    )
    graphsafe_decision = apply_frozen_abstention_decision(
        torch.tensor(graphsafe_probability, dtype=torch.float32),
        threshold=graphsafe_threshold.threshold,
        capacity=target_capacity,
        forced_abstention=torch.tensor(best_forced),
    ).numpy()
    graphsafe_name = "graphsafe_confidence_abstention_component"
    output[graphsafe_name] = _baseline_prediction(
        scores=target_matrix[:, best],
        abstention_probability=graphsafe_probability,
        abstain=graphsafe_decision,
        forced_abstention=best_forced,
        expected_compute=target_mask[:, best].astype(float) * costs[best],
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=target_mask[:, best],
        threshold=graphsafe_threshold.threshold,
        threshold_provenance=graphsafe_threshold.fitted_on,
        adapter="models.graphsafe_v2.confidence_scores",
        details={
            "parity_status": "PARTIAL_CONFIDENCE_COMPONENT_NOT_FULL_GRAPHSAFE",
            "source_validation_expert": experts[best],
            "router_training_seed": derive_router_seed(
                expert_prediction_seed,
                graphsafe_name,
            ),
        },
    )

    feature_index = next(
        (
            index
            for index, name in enumerate(experts)
            if any(
                token in name.lower()
                for token in ("feature", "mlp", "logistic")
            )
        ),
        0,
    )
    graph_index = next(
        (
            index
            for index, name in enumerate(experts)
            if index != feature_index
            and any(
                token in name.lower()
                for token in ("graph", "gcn", "sage", "gat")
            )
        ),
        1 if len(experts) > 1 else 0,
    )
    from models.graph_feature_gating import GraphFeatureGate

    gate_training_rows = (
        validation_availability[:, graph_index]
        & validation_availability[:, feature_index]
    )
    graph_gate_name = "current_graph_feature_gate_adapter"
    graph_gate_seed = derive_router_seed(
        expert_prediction_seed,
        graph_gate_name,
    )
    gate = GraphFeatureGate(
        mode="logistic",
        min_validation=4,
        seed=graph_gate_seed,
    ).fit(
        validation_matrix[gate_training_rows, graph_index],
        validation_matrix[gate_training_rows, feature_index],
        validation_labels[gate_training_rows],
    )
    gate_scores = gate.predict(
        target_matrix[:, graph_index],
        target_matrix[:, feature_index],
    )
    graph_available = target_mask[:, graph_index]
    feature_available = target_mask[:, feature_index]
    gate_scores = np.where(
        graph_available & ~feature_available,
        target_matrix[:, graph_index],
        gate_scores,
    )
    gate_scores = np.where(
        feature_available & ~graph_available,
        target_matrix[:, feature_index],
        gate_scores,
    )
    gate_executable = graph_available | feature_available
    gate_forced = ~gate_executable
    gate_scores = np.where(gate_executable, gate_scores, 0.0)
    output[graph_gate_name] = _baseline_prediction(
        scores=gate_scores,
        abstention_probability=gate_forced.astype(float),
        abstain=gate_forced,
        forced_abstention=gate_forced,
        expected_compute=(
            graph_available.astype(float) * costs[graph_index]
            + feature_available.astype(float) * costs[feature_index]
        ),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=gate_executable,
        fallback_used=bool(np.any(graph_available != feature_available)),
        learned=True,
        adapter="models.graph_feature_gating.GraphFeatureGate",
        details={
            "mode": gate.mode,
            "fallback_used": gate.fallback_used,
            "router_training_seed": graph_gate_seed,
        },
    )
    no_contract_name = "learned_no_contract_router"
    no_contract_seed = derive_router_seed(
        expert_prediction_seed,
        no_contract_name,
    )
    no_contract_scores, no_contract_weights = _learned_gate_baseline(
        source_groups,
        experts,
        target_matrix,
        target_mask,
        atomic=False,
        seed=no_contract_seed,
    )
    output[no_contract_name] = _baseline_prediction(
        scores=no_contract_scores,
        abstention_probability=forced.astype(float),
        abstain=forced,
        forced_abstention=forced,
        expected_compute=(no_contract_weights * costs).sum(axis=1),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=executable_rows,
        fallback_used=not target_mask.all(),
        learned=True,
        adapter="coregraph.learned_no_contract_router",
        details={"router_training_seed": no_contract_seed},
    )
    atomic_name = "learned_atomic_contract_router"
    atomic_seed = derive_router_seed(expert_prediction_seed, atomic_name)
    atomic_scores, atomic_weights = _learned_gate_baseline(
        source_groups,
        experts,
        target_matrix,
        target_mask,
        atomic=True,
        seed=atomic_seed,
    )
    output[atomic_name] = _baseline_prediction(
        scores=atomic_scores,
        abstention_probability=forced.astype(float),
        abstain=forced,
        forced_abstention=forced,
        expected_compute=(atomic_weights * costs).sum(axis=1),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=executable_rows,
        fallback_used=not target_mask.all(),
        learned=True,
        adapter="coregraph.learned_atomic_contract_router",
        details={
            "unseen_target_atomic_id": True,
            "router_training_seed": atomic_seed,
        },
    )
    mowst_threshold = _select_mowst_routing_threshold(
        validation_matrix,
        validation_labels,
        validation_groups,
        validation_availability,
        feature_index=feature_index,
        graph_index=graph_index,
    )
    feature_confidence = np.abs(target_matrix[:, feature_index] - 0.5)
    mowst_choice = np.where(
        feature_confidence >= mowst_threshold,
        feature_index,
        graph_index,
    )
    alternative = np.where(
        mowst_choice == feature_index,
        graph_index,
        feature_index,
    )
    rows = np.arange(len(target_matrix))
    chosen_available = target_mask[rows, mowst_choice]
    alternative_available = target_mask[rows, alternative]
    mowst_choice = np.where(
        chosen_available,
        mowst_choice,
        np.where(alternative_available, alternative, mowst_choice),
    )
    mowst_feasible = target_mask[rows, mowst_choice]
    mowst_forced = ~mowst_feasible
    mowst_scores = np.where(
        mowst_feasible,
        target_matrix[rows, mowst_choice],
        0.0,
    )
    mowst_name = "MOWST_INSPIRED_REIMPLEMENTATION"
    output[mowst_name] = _baseline_prediction(
        scores=mowst_scores,
        abstention_probability=mowst_forced.astype(float),
        abstain=mowst_forced,
        forced_abstention=mowst_forced,
        expected_compute=np.where(
            mowst_feasible,
            costs[mowst_choice],
            0.0,
        ),
        target_capacity=target_capacity,
        abstention_cost=abstention_cost,
        executable_rows=mowst_feasible,
        fallback_used=bool(np.any(chosen_available != mowst_feasible)),
        adapter=mowst_name,
        details={
            "official_baseline": False,
            "routing_threshold": mowst_threshold,
            "routing_threshold_fitted_on": (
                "source_validation_balanced_contracts"
            ),
            "router_training_seed": derive_router_seed(
                expert_prediction_seed,
                mowst_name,
            ),
        },
    )
    return output


def contract_feasible_oracle(
    *,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
    target_expert_costs: Mapping[str, float],
    target_labels: np.ndarray,
) -> BaselinePrediction:
    """Select one best feasible expert for the entire target contract."""

    experts = sorted(target_scores)
    if (
        set(target_availability) != set(experts)
        or set(target_expert_costs) != set(experts)
    ):
        raise ValueError("contract oracle requires aligned scores, masks, and costs")
    matrix = np.column_stack([target_scores[name] for name in experts])
    validate_numpy_scores(matrix, ScoreType.PROBABILITY)
    availability = np.column_stack(
        [np.asarray(target_availability[name], dtype=bool) for name in experts]
    )
    if matrix.shape != availability.shape:
        raise ValueError("contract oracle scores and availability must align")
    labels = (np.asarray(target_labels).reshape(-1) == 1).astype(float)
    if len(labels) != len(matrix):
        raise ValueError("contract oracle labels must align with target scores")
    costs = np.asarray([target_expert_costs[name] for name in experts])
    contract_available = availability.all(axis=0)
    if not contract_available.any():
        raise ValueError(
            "contract feasible oracle requires one expert executable "
            "for the whole contract"
        )
    risks = np.where(
        contract_available,
        np.mean((matrix - labels[:, None]) ** 2, axis=0),
        np.inf,
    )
    chosen = int(np.argmin(risks))
    forced = np.zeros(len(matrix), dtype=bool)
    return _baseline_prediction(
        scores=matrix[:, chosen],
        abstention_probability=np.zeros(len(matrix)),
        abstain=forced,
        forced_abstention=forced,
        expected_compute=np.full(len(matrix), costs[chosen]),
        target_capacity=None,
        abstention_cost=0.0,
        executable_rows=np.ones(len(matrix), dtype=bool),
        offline_oracle=True,
        diagnostic_only=False,
        adapter="contract_feasible_oracle",
        details={
            "oracle_kind": "one_best_feasible_expert_per_contract",
            "selected_expert": experts[chosen],
            "target_labels_used_for_offline_headline_oracle_only": True,
        },
    )


def instance_clairvoyant_oracle_ceiling(
    *,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
    target_expert_costs: Mapping[str, float],
    target_labels: np.ndarray,
) -> BaselinePrediction:
    """Construct the non-deployable per-instance diagnostic ceiling."""

    experts = sorted(target_scores)
    if (
        set(target_availability) != set(experts)
        or set(target_expert_costs) != set(experts)
    ):
        raise ValueError("instance oracle requires aligned scores, masks, and costs")
    matrix = np.column_stack([target_scores[name] for name in experts])
    validate_numpy_scores(matrix, ScoreType.PROBABILITY)
    availability = np.column_stack(
        [np.asarray(target_availability[name], dtype=bool) for name in experts]
    )
    if matrix.shape != availability.shape:
        raise ValueError("instance oracle scores and availability must align")
    labels = (np.asarray(target_labels).reshape(-1) == 1).astype(float)
    if len(labels) != len(matrix):
        raise ValueError("instance oracle labels must align with target scores")
    costs = np.asarray([target_expert_costs[name] for name in experts])
    errors = np.where(availability, np.abs(matrix - labels[:, None]), np.inf)
    chosen = errors.argmin(axis=1)
    no_feasible = ~availability.any(axis=1)
    rows = np.arange(len(matrix))
    scores = matrix[rows, chosen]
    scores[no_feasible] = 0.0
    compute = costs[chosen]
    compute[no_feasible] = 0.0
    return _baseline_prediction(
        scores=scores,
        abstention_probability=no_feasible.astype(float),
        abstain=no_feasible,
        forced_abstention=no_feasible,
        expected_compute=compute,
        target_capacity=None,
        abstention_cost=0.0,
        executable_rows=~no_feasible,
        offline_oracle=True,
        diagnostic_only=True,
        adapter="instance_clairvoyant_oracle_ceiling",
        details={
            "oracle_kind": "per_instance_clairvoyant_diagnostic_only",
            "target_labels_used_for_offline_diagnostic_only": True,
        },
    )


def _diagnostics(
    values: np.ndarray,
    costs: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return (
            np.zeros((len(values), 3)),
            np.zeros((len(values), values.shape[1], 3)),
        )
    shared = np.column_stack(
        [
            values.mean(axis=1),
            values.std(axis=1),
            values.max(axis=1) - values.min(axis=1),
        ]
    )
    per_expert = np.stack(
        [
            np.abs(values - 0.5) * 2,
            np.abs(values - values.mean(axis=1, keepdims=True)),
            np.broadcast_to(
                costs / max(float(np.max(costs)), 1e-8),
                values.shape,
            ),
        ],
        axis=-1,
    )
    return shared, per_expert


def _assemble_source(
    source_groups: Sequence[SavedSourceGroup],
    experts: Sequence[str],
    split: str,
) -> tuple[
    list[DeploymentContract],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    contracts: list[DeploymentContract] = []
    score_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    cost_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    for group_index, group in enumerate(source_groups):
        keep = np.asarray(group.splits) == split
        if not keep.any():
            raise ValueError(
                f"source contract {group.contract.contract_id} has no {split} rows"
            )
        count = int(keep.sum())
        contracts.extend([group.contract] * count)
        score_rows.append(
            np.column_stack([group.scores[name] for name in experts])[keep]
        )
        label_rows.append((np.asarray(group.labels)[keep] == 1).astype(float))
        mask_rows.append(
            np.column_stack(
                [group.availability[name] for name in experts]
            )[keep]
        )
        costs = np.asarray([group.expert_costs[name] for name in experts])
        cost_rows.append(np.broadcast_to(costs, (count, len(experts))).copy())
        group_rows.append(np.full(count, group_index))
    return (
        contracts,
        np.concatenate(score_rows),
        np.concatenate(label_rows),
        np.concatenate(mask_rows).astype(bool),
        np.concatenate(cost_rows),
        np.concatenate(group_rows),
    )


def fit_saved_output_corerouter(
    source_groups: Sequence[SavedSourceGroup],
    *,
    target_contract: DeploymentContract,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
    target_expert_costs: Mapping[str, float],
    expert_prediction_seed: int,
    steps: int = 100,
    ablation: PilotAblation = PilotAblation.FULL,
    abstention_cost: float = 0.2,
) -> SavedRouterPrediction:
    """Fit on source-train, stop on source-validation, then freeze for target."""

    if len(source_groups) < 2:
        raise ValueError("full CoReRouter pilot requires at least two source contracts")
    experts = sorted(target_scores)
    if len(experts) < 2:
        raise ValueError("full CoReRouter pilot requires two aligned experts")
    for group in source_groups:
        if set(group.scores) != set(experts):
            raise ValueError("source and target expert sets must align")
    if set(target_availability) != set(experts):
        raise ValueError("target availability must declare every expert")
    if set(target_expert_costs) != set(experts):
        raise ValueError("target expert costs must declare every expert")
    if steps < 1:
        raise ValueError("router steps must be positive")
    if abstention_cost < 0:
        raise ValueError("abstention cost cannot be negative")
    train = _assemble_source(source_groups, experts, "train")
    validation = _assemble_source(source_groups, experts, "validation")
    (
        train_contracts,
        train_matrix,
        train_labels,
        train_availability,
        train_costs,
        train_groups,
    ) = train
    (
        validation_contracts,
        validation_matrix,
        validation_labels,
        validation_availability,
        validation_costs,
        validation_groups,
    ) = validation
    use_resource_mask = ablation is not PilotAblation.NO_RESOURCE_MASK
    if not use_resource_mask:
        train_availability = np.ones_like(train_availability)
        validation_availability = np.ones_like(validation_availability)
    diagnostics_enabled = ablation is not PilotAblation.NO_DIAGNOSTICS
    train_shared, train_per_expert = _diagnostics(
        train_matrix,
        train_costs.mean(axis=0),
        enabled=diagnostics_enabled,
    )
    validation_shared, validation_per_expert = _diagnostics(
        validation_matrix,
        validation_costs.mean(axis=0),
        enabled=diagnostics_enabled,
    )
    encoder_kind = (
        "none"
        if ablation is PilotAblation.NO_CONTRACT
        else "atomic"
        if ablation is PilotAblation.ATOMIC_CONTRACT
        else "factorised"
    )
    method_name = (
        "full_corerouter"
        if ablation is PilotAblation.FULL
        else f"ablation:{ablation.value}"
    )
    router_training_seed = derive_router_seed(
        expert_prediction_seed,
        method_name,
    )
    seed_everything(router_training_seed)
    model = CoReGraph(
        num_experts=len(experts),
        diagnostic_dim=3,
        per_expert_diagnostic_dim=3,
        axis_dropout=0.05,
        contract_noise_std=0.0,
        contract_encoder_kind=encoder_kind,
        seen_contract_ids=[
            group.contract.contract_id for group in source_groups
        ],
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    weights = ObjectiveWeights(
        average=1.0,
        ranking=0.1,
        robust_regret=(
            0.0 if ablation is PilotAblation.NO_REGRET else 1.0
        ),
        budget=0.0 if ablation is PilotAblation.NO_BUDGET else 0.1,
        stability=(
            0.0 if ablation is PilotAblation.NO_STABILITY else 0.1
        ),
        compute=0.1,
        calibration=0.1,
        abstention=(
            0.0 if ablation is PilotAblation.NO_ABSTENTION else 0.2
        ),
    )
    objective = CompositeObjective(weights, cvar_alpha=0.8)

    def tensor(value: np.ndarray, *, boolean: bool = False) -> torch.Tensor:
        return torch.tensor(
            value,
            dtype=torch.bool if boolean else torch.float32,
        )

    train_score = tensor(train_matrix)
    train_target = tensor(train_labels)
    train_mask = tensor(train_availability, boolean=True)
    train_cost = tensor(train_costs)
    train_shared_tensor = tensor(train_shared)
    train_per_tensor = tensor(train_per_expert)
    train_group_tensor = torch.tensor(train_groups, dtype=torch.long)
    validation_score = tensor(validation_matrix)
    validation_mask = tensor(validation_availability, boolean=True)
    validation_cost = tensor(validation_costs)
    validation_shared_tensor = tensor(validation_shared)
    validation_per_tensor = tensor(validation_per_expert)
    validation_target = tensor(validation_labels)
    validation_group_tensor = torch.tensor(
        validation_groups,
        dtype=torch.long,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    patience = max(2, min(10, steps // 4 or 2))
    source_contracts = tuple(group.contract for group in source_groups)
    train_review_k = source_review_k_by_group(
        source_contracts,
        train_groups,
    )
    validation_review_k = source_review_k_by_group(
        source_contracts,
        validation_groups,
    )
    train_capacities = source_abstention_capacity_by_group(
        source_contracts,
        train_groups,
    )
    validation_capacities = source_abstention_capacity_by_group(
        source_contracts,
        validation_groups,
    )
    model.train()
    for _ in range(steps):
        output = model(
            contracts=train_contracts,
            expert_scores=train_score,
            score_type=ScoreType.PROBABILITY,
            shared_diagnostics=train_shared_tensor,
            per_expert_diagnostics=train_per_tensor,
            availability_mask=train_mask,
            expert_costs=train_cost,
            expert_names=experts,
        )
        stability = train_score.sum() * 0
        if weights.stability > 0:
            perturbation = torch.linspace(
                -1e-3,
                1e-3,
                train_score.numel(),
            ).reshape_as(train_score)
            perturbed_scores = (train_score + perturbation).clamp(0, 1)
            perturbed_shared, perturbed_per = _diagnostics(
                perturbed_scores.detach().numpy(),
                train_costs.mean(axis=0),
                enabled=diagnostics_enabled,
            )
            perturbed_output = model(
                contracts=train_contracts,
                expert_scores=perturbed_scores,
                score_type=ScoreType.PROBABILITY,
                shared_diagnostics=tensor(perturbed_shared),
                per_expert_diagnostics=tensor(perturbed_per),
                availability_mask=train_mask,
                expert_costs=train_cost,
                expert_names=experts,
            )
            stability = consistency_penalty(
                output.expert_weights,
                perturbed_output.expert_weights,
            )
        total, _ = objective(
            router_scores=output.blended_score,
            score_type=output.score_type,
            targets=train_target,
            group_indices=train_group_tensor,
            expert_scores=train_score,
            availability_mask=train_mask,
            expert_weights=output.expert_weights,
            expert_costs=train_cost,
            stability_penalty=stability,
            review_k_by_group=train_review_k,
            abstention_probability=(
                None
                if ablation is PilotAblation.NO_ABSTENTION
                else output.abstention_probability
            ),
            forced_abstention=~train_mask.any(dim=1),
            abstention_capacity_by_group=(
                None
                if ablation is PilotAblation.NO_ABSTENTION
                else train_capacities
            ),
            abstention_cost_value=abstention_cost,
        )
        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_output = model(
                contracts=validation_contracts,
                expert_scores=validation_score,
                score_type=ScoreType.PROBABILITY,
                shared_diagnostics=validation_shared_tensor,
                per_expert_diagnostics=validation_per_tensor,
                availability_mask=validation_mask,
                expert_costs=validation_cost,
                expert_names=experts,
            )
            validation_loss, _ = objective(
                router_scores=validation_output.blended_score,
                score_type=validation_output.score_type,
                targets=validation_target,
                group_indices=validation_group_tensor,
                expert_scores=validation_score,
                availability_mask=validation_mask,
                expert_weights=validation_output.expert_weights,
                expert_costs=validation_cost,
                review_k_by_group=validation_review_k,
                abstention_probability=(
                    None
                    if ablation is PilotAblation.NO_ABSTENTION
                    else validation_output.abstention_probability
                ),
                forced_abstention=~validation_mask.any(dim=1),
                abstention_capacity_by_group=(
                    None
                    if ablation is PilotAblation.NO_ABSTENTION
                    else validation_capacities
                ),
                abstention_cost_value=abstention_cost,
            )
            value = float(validation_loss.item())
        if value < best_validation - 1e-9:
            best_validation = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
        model.train()
    model.load_state_dict(best_state)
    model.eval()
    source_fit_hash = _model_state_hash(model)
    with torch.no_grad():
        validation_output = model(
            contracts=validation_contracts,
            expert_scores=validation_score,
            score_type=ScoreType.PROBABILITY,
            shared_diagnostics=validation_shared_tensor,
            per_expert_diagnostics=validation_per_tensor,
            availability_mask=validation_mask,
            expert_costs=validation_cost,
            expert_names=experts,
        )
    validation_losses = binary_cross_entropy_values(
        validation_output.blended_score,
        validation_target,
        score_type=ScoreType.PROBABILITY,
    )
    if ablation is PilotAblation.NO_ABSTENTION:
        threshold = float("inf")
        threshold_fitted_on = "source_validation_disabled_ablation"
    else:
        selected_threshold = select_grouped_abstention_threshold(
            validation_losses,
            validation_output.abstention_probability,
            validation_group_tensor,
            capacities=validation_capacities,
            abstention_cost_value=abstention_cost,
            forced_abstention=~validation_mask.any(dim=1),
        )
        threshold = selected_threshold.threshold
        threshold_fitted_on = selected_threshold.fitted_on

    target_matrix = np.column_stack([target_scores[name] for name in experts])
    target_mask_array = np.column_stack(
        [target_availability[name] for name in experts]
    ).astype(bool)
    if target_matrix.shape != target_mask_array.shape:
        raise ValueError("target scores and target availability must align")
    if not use_resource_mask:
        target_mask_array = np.ones_like(target_mask_array)
    target_cost_array = np.asarray(
        [target_expert_costs[name] for name in experts],
        dtype=float,
    )
    target_shared, target_per_expert = _diagnostics(
        target_matrix,
        target_cost_array,
        enabled=diagnostics_enabled,
    )
    with torch.no_grad():
        target_output = model(
            contracts=[target_contract] * len(target_matrix),
            expert_scores=tensor(target_matrix),
            score_type=ScoreType.PROBABILITY,
            shared_diagnostics=tensor(target_shared),
            per_expert_diagnostics=tensor(target_per_expert),
            availability_mask=tensor(target_mask_array, boolean=True),
            expert_costs=tensor(target_cost_array),
            expert_names=experts,
        )
        target_perturbation = np.linspace(
            -1e-3,
            1e-3,
            target_matrix.size,
        ).reshape(target_matrix.shape)
        perturbed_matrix = np.clip(
            target_matrix + target_perturbation,
            0,
            1,
        )
        perturbed_shared, perturbed_per = _diagnostics(
            perturbed_matrix,
            target_cost_array,
            enabled=diagnostics_enabled,
        )
        perturbed = model(
            contracts=[target_contract] * len(target_matrix),
            expert_scores=tensor(perturbed_matrix),
            score_type=ScoreType.PROBABILITY,
            shared_diagnostics=tensor(perturbed_shared),
            per_expert_diagnostics=tensor(perturbed_per),
            availability_mask=tensor(target_mask_array, boolean=True),
            expert_costs=tensor(target_cost_array),
            expert_names=experts,
        )
    forced_abstention = ~target_mask_array.any(axis=1)
    target_capacity = target_contract.budget.abstention_capacity
    if ablation is PilotAblation.NO_ABSTENTION:
        abstain = forced_abstention
    else:
        abstain = apply_frozen_abstention_decision(
            target_output.abstention_probability,
            threshold=threshold,
            capacity=target_capacity,
            forced_abstention=target_output.all_experts_unavailable,
        )
    selected = target_output.selected_expert.cpu().numpy()
    perturbed_selected = perturbed.selected_expert.cpu().numpy()
    return SavedRouterPrediction(
        scores=target_output.blended_score.cpu().numpy(),
        selected_experts=selected,
        routing_weights=target_output.expert_weights.cpu().numpy(),
        abstention_probability=(
            target_output.abstention_probability.cpu().numpy()
        ),
        abstain=abstain.cpu().numpy() if isinstance(abstain, torch.Tensor) else abstain,
        expected_compute=target_output.expected_compute.cpu().numpy(),
        forced_abstention=forced_abstention,
        perturbation_flip_rate=float(
            np.mean(selected != perturbed_selected)
        ),
        ablation=ablation,
        source_train_examples=len(train_labels),
        source_validation_examples=len(validation_labels),
        abstention_threshold=threshold,
        abstention_threshold_fitted_on=threshold_fitted_on,
        source_abstention_capacities=validation_capacities,
        target_abstention_capacity=target_capacity,
        abstention_cost=abstention_cost,
        expert_prediction_seed=expert_prediction_seed,
        router_training_seed=router_training_seed,
        source_fit_hash=source_fit_hash,
    )


def evaluate_saved_output_pilot(
    target_labels: np.ndarray,
    candidates: Mapping[str, BaselinePrediction],
    *,
    dataset: str,
    target_contract: str,
    expert_prediction_seed: int,
    router_training_seeds: Mapping[str, int],
    fold: str,
) -> dict[str, Any]:
    """Score frozen predictions after every fit/hyperparameter is frozen."""

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    binary = (np.asarray(target_labels) == 1).astype(float)
    oracle = candidates.get("contract_feasible_oracle")
    if oracle is None:
        raise ValueError("pilot evaluation requires the contract feasible oracle")
    if not oracle.offline_oracle or oracle.diagnostic_only:
        raise ValueError(
            "contract feasible oracle must be an offline headline reference"
        )
    oracle_losses = (oracle.scores - binary) ** 2
    oracle_effective = np.where(
        oracle.abstain,
        oracle.abstention_cost,
        oracle_losses,
    )
    oracle_risk = float(np.mean(oracle_effective))
    measured_methods = {
        method
        for method, prediction in candidates.items()
        if not prediction.diagnostic_only
        and method != "contract_feasible_oracle"
    }
    if set(router_training_seeds) != measured_methods:
        raise ValueError(
            "router training seed map must cover measured methods exactly"
        )
    for method, prediction in sorted(candidates.items()):
        if method == "contract_feasible_oracle":
            continue
        if prediction.diagnostic_only:
            diagnostics.append(
                {
                    "dataset": dataset,
                    "target_contract": target_contract,
                    "expert_prediction_seed": int(expert_prediction_seed),
                    "fold": fold,
                    "name": method,
                    "risk": float(
                        np.mean(
                            np.where(
                                prediction.abstain,
                                prediction.abstention_cost,
                                (prediction.scores - binary) ** 2,
                            )
                        )
                    ),
                    "excluded_from_significance": True,
                    "excluded_from_deployable_methods": True,
                }
            )
            continue
        router_training_seed = int(router_training_seeds[method])
        if router_training_seed == expert_prediction_seed:
            raise ValueError(
                "router training seed must differ from expert prediction seed"
            )
        rank_eligible = prediction.ranking_eligible
        metrics = (
            binary_metrics(target_labels, prediction.scores)
            if rank_eligible
            else {"auprc": float("nan")}
        )
        decision = np.asarray(prediction.abstain, dtype=bool)
        point_losses = (
            (prediction.scores >= 0.5).astype(float) - binary
        ) ** 2
        accepted = ~decision
        selective = (
            float(point_losses[accepted].mean())
            if accepted.any()
            else float("nan")
        )
        contract_losses = (prediction.scores - binary) ** 2
        contract_risk = float(
            np.mean(
                np.where(
                    decision,
                    prediction.abstention_cost,
                    contract_losses,
                )
            )
        )
        valid_headline_risk = (
            rank_eligible and accepted.any()
        )

        def ranking_value(function: Any, *args: Any) -> float:
            return float(function(*args)) if rank_eligible else float("nan")

        values = {
            "auprc": metrics["auprc"],
            "recall_at_0.5pct": ranking_value(
                recall_at_k,
                target_labels,
                prediction.scores,
                0.005,
            ),
            "recall_at_1pct": ranking_value(
                recall_at_k,
                target_labels,
                prediction.scores,
                0.01,
            ),
            "recall_at_2pct": ranking_value(
                recall_at_k,
                target_labels,
                prediction.scores,
                0.02,
            ),
            "budget_curve_area": ranking_value(
                budget_curve_auc,
                target_labels,
                prediction.scores,
                (0.005, 0.01, 0.02),
            ),
            "contract_regret": (
                contract_risk - oracle_risk
                if valid_headline_risk
                else float("nan")
            ),
            "selective_risk": selective,
            "coverage": float(accepted.mean()),
            "aurc": float(
                area_under_risk_coverage_curve(
                    torch.tensor(point_losses, dtype=torch.float32),
                    torch.tensor(
                        prediction.abstention_probability,
                        dtype=torch.float32,
                    ),
                ).item()
            )
            if rank_eligible
            else float("nan"),
            "abstention_cost": float(
                decision.mean() * prediction.abstention_cost
            ),
            "compute": float(np.mean(prediction.expected_compute)),
        }
        for metric, value in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "target_contract": target_contract,
                    "seed": int(expert_prediction_seed),
                    "expert_prediction_seed": int(expert_prediction_seed),
                    "router_training_seed": router_training_seed,
                    "fold": fold,
                    "method": method,
                    "metric": metric,
                    "value": float(value),
                    "execution_status": MethodExecutionStatus(
                        prediction.execution_status
                    ).value,
                    "abstention_threshold": prediction.abstention_threshold,
                    "abstention_threshold_provenance": (
                        prediction.abstention_threshold_provenance
                    ),
                    "routing_threshold": prediction.details.get(
                        "routing_threshold"
                    ),
                    "routing_threshold_provenance": prediction.details.get(
                        "routing_threshold_fitted_on"
                    ),
                    "abstention_decision_sha256": hashlib.sha256(
                        decision.astype(np.uint8).tobytes()
                    ).hexdigest(),
                    "accepted_count": int(accepted.sum()),
                    "abstained_count": int(decision.sum()),
                    "forced_abstention": bool(
                        prediction.forced_abstention.any()
                    ),
                    "forced_abstention_count": int(
                        prediction.forced_abstention.sum()
                    ),
                    "abstention_capacity": prediction.abstention_capacity,
                    "abstention_cost_per_decision": prediction.abstention_cost,
                    "offline_oracle": prediction.offline_oracle,
                }
            )
    return {
        "schema": "coregraph_saved_output_pilot_v3",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": rows,
        "headline_oracle_reference": {
            "dataset": dataset,
            "target_contract": target_contract,
            "expert_prediction_seed": int(expert_prediction_seed),
            "fold": fold,
            "name": "contract_feasible_oracle",
            "risk": oracle_risk,
            "selected_expert": oracle.details.get("selected_expert"),
            "used_for_headline_regret": True,
            "excluded_from_significance_as_a_method": True,
            "excluded_from_deployable_methods": True,
        },
        "diagnostic_oracles": diagnostics,
        "target_information": "labels_used_for_final_offline_scoring_only",
        "target_label_selection": False,
        "oracle_target_selection": False,
        "headline_oracle": "contract_feasible_oracle",
        "diagnostic_oracle": "instance_clairvoyant_oracle_ceiling",
        "inferential_block": "expert_prediction_seed",
    }
