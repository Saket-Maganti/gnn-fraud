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
    apply_abstention_capacity,
    area_under_risk_coverage_curve,
    select_abstention_threshold,
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
    expected_compute: np.ndarray
    learned: bool = False
    adapter: str = ""
    offline_oracle: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = validate_numpy_scores(
            np.asarray(self.scores),
            ScoreType.PROBABILITY,
        )
        abstention = np.asarray(self.abstention_probability, dtype=float)
        compute = np.asarray(self.expected_compute, dtype=float)
        if (
            scores.ndim != 1
            or abstention.shape != scores.shape
            or compute.shape != scores.shape
        ):
            raise ValueError("baseline score, abstention, and compute rows must align")
        if (
            not np.isfinite(abstention).all()
            or np.any(abstention < 0)
            or np.any(abstention > 1)
        ):
            raise ValueError("baseline abstention probabilities must lie in [0,1]")
        if not np.isfinite(compute).all() or np.any(compute < 0):
            raise ValueError("baseline compute must be finite and non-negative")


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
    perturbation_flip_rate: float
    ablation: PilotAblation
    source_train_examples: int
    source_validation_examples: int
    abstention_threshold: float
    abstention_threshold_fitted_on: str
    early_stopping_source_only: bool = True


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
        "seed",
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
                seed=int(payload["seed"]),
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
            random_state=0,
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


def baseline_scores(
    source_groups: Sequence[SavedSourceGroup],
    *,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
) -> dict[str, BaselinePrediction]:
    """Fit label-free baselines on the complete source-contract set."""

    if not source_groups:
        raise ValueError("baseline fitting requires source contracts")
    experts = sorted(target_scores)
    if any(set(group.scores) != set(experts) for group in source_groups):
        raise ValueError("all source and target groups need the same experts")
    if set(target_availability) != set(experts):
        raise ValueError("target availability must declare every expert")
    target_matrix = np.column_stack([target_scores[name] for name in experts])
    validate_numpy_scores(target_matrix, ScoreType.PROBABILITY)
    target_mask = np.column_stack(
        [np.asarray(target_availability[name], dtype=bool) for name in experts]
    )
    if target_matrix.shape != target_mask.shape:
        raise ValueError("target scores and availability must align")
    (
        validation_matrix,
        validation_labels,
        validation_groups,
        validation_availability,
    ) = (
        _stack_source_validation(source_groups, experts)
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
        [
            np.mean([group.expert_costs[name] for group in source_groups])
            for name in experts
        ]
    )
    output = {
        f"expert:{name}": BaselinePrediction(
            scores=target_matrix[:, index],
            abstention_probability=(~target_mask[:, index]).astype(float),
            expected_compute=target_mask[:, index].astype(float) * costs[index],
            details={"expert_id": name},
        )
        for index, name in enumerate(experts)
    }
    feasible_count = target_mask.sum(axis=1)
    feasible_sum = np.where(target_mask, target_matrix, 0.0).sum(axis=1)
    average = np.divide(
        feasible_sum,
        feasible_count,
        out=np.zeros(len(target_matrix)),
        where=feasible_count > 0,
    )
    output["average_all_feasible"] = BaselinePrediction(
        scores=average,
        abstention_probability=(feasible_count == 0).astype(float),
        expected_compute=(target_mask * costs).sum(axis=1),
    )
    output["best_source_validation"] = BaselinePrediction(
        scores=target_matrix[:, best],
        abstention_probability=(~target_mask[:, best]).astype(float),
        expected_compute=target_mask[:, best].astype(float) * costs[best],
        details={"selected_expert": experts[best]},
    )
    convex_weights = _convex_validation_weights(
        validation_matrix,
        validation_labels,
        validation_groups,
        validation_availability,
    )
    masked_weights = target_mask * convex_weights[None, :]
    masked_weights = np.divide(
        masked_weights,
        masked_weights.sum(axis=1, keepdims=True),
        out=np.zeros_like(masked_weights),
        where=masked_weights.sum(axis=1, keepdims=True) > 0,
    )
    output["source_validation_convex_mixture"] = BaselinePrediction(
        scores=(masked_weights * target_matrix).sum(axis=1),
        abstention_probability=(feasible_count == 0).astype(float),
        expected_compute=(masked_weights * costs).sum(axis=1),
        details={"weights": dict(zip(experts, convex_weights, strict=True))},
    )

    best_scores = target_matrix[:, best]
    from models.graphsafe_v2 import confidence_scores

    graphsafe_confidence = np.clip(
        confidence_scores(best_scores),
        0,
        1,
    )
    output["graphsafe_v2_adapter"] = BaselinePrediction(
        scores=best_scores,
        abstention_probability=np.maximum(
            (~target_mask[:, best]).astype(float),
            1 - graphsafe_confidence,
        ),
        expected_compute=target_mask[:, best].astype(float) * costs[best],
        adapter="models.graphsafe_v2",
        details={
            "source_validation_expert": experts[best],
            "confidence_adapter": "models.graphsafe_v2.confidence_scores",
        },
    )
    feature_index = next(
        (
            index
            for index, name in enumerate(experts)
            if any(token in name.lower() for token in ("feature", "mlp", "logistic"))
        ),
        0,
    )
    graph_index = next(
        (
            index
            for index, name in enumerate(experts)
            if index != feature_index
            and any(token in name.lower() for token in ("graph", "gcn", "sage", "gat"))
        ),
        1 if len(experts) > 1 else 0,
    )
    from models.graph_feature_gating import GraphFeatureGate

    gate_training_rows = (
        validation_availability[:, graph_index]
        & validation_availability[:, feature_index]
    )
    gate = GraphFeatureGate(mode="logistic", min_validation=4, seed=0).fit(
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
    gate_scores = np.where(
        graph_available | feature_available,
        gate_scores,
        0.0,
    )
    output["current_graph_feature_gate_adapter"] = BaselinePrediction(
        scores=gate_scores,
        abstention_probability=(
            ~(graph_available | feature_available)
        ).astype(float),
        expected_compute=(
            graph_available.astype(float) * costs[graph_index]
            + feature_available.astype(float) * costs[feature_index]
        ),
        learned=True,
        adapter="models.graph_feature_gating.GraphFeatureGate",
        details={"mode": gate.mode, "fallback_used": gate.fallback_used},
    )
    no_contract_scores, no_contract_weights = _learned_gate_baseline(
        source_groups,
        experts,
        target_matrix,
        target_mask,
        atomic=False,
    )
    output["learned_no_contract_router"] = BaselinePrediction(
        scores=no_contract_scores,
        abstention_probability=(feasible_count == 0).astype(float),
        expected_compute=(no_contract_weights * costs).sum(axis=1),
        learned=True,
        adapter="coregraph.learned_no_contract_router",
    )
    atomic_scores, atomic_weights = _learned_gate_baseline(
        source_groups,
        experts,
        target_matrix,
        target_mask,
        atomic=True,
    )
    output["learned_atomic_contract_router"] = BaselinePrediction(
        scores=atomic_scores,
        abstention_probability=(feasible_count == 0).astype(float),
        expected_compute=(atomic_weights * costs).sum(axis=1),
        learned=True,
        adapter="coregraph.learned_atomic_contract_router",
        details={"unseen_target_atomic_id": True},
    )
    feature_confidence = np.abs(target_matrix[:, feature_index] - 0.5)
    mowst_choice = np.where(
        feature_confidence >= 0.25,
        feature_index,
        graph_index,
    )
    alternative = np.where(mowst_choice == feature_index, graph_index, feature_index)
    rows = np.arange(len(target_matrix))
    chosen_available = target_mask[rows, mowst_choice]
    alternative_available = target_mask[rows, alternative]
    mowst_choice = np.where(
        chosen_available,
        mowst_choice,
        np.where(alternative_available, alternative, mowst_choice),
    )
    mowst_feasible = target_mask[rows, mowst_choice]
    mowst_scores = target_matrix[rows, mowst_choice]
    mowst_scores = np.where(mowst_feasible, mowst_scores, 0.0)
    output["MOWST_INSPIRED_REIMPLEMENTATION"] = BaselinePrediction(
        scores=mowst_scores,
        abstention_probability=(~mowst_feasible).astype(float),
        expected_compute=np.where(mowst_feasible, costs[mowst_choice], 0.0),
        adapter="MOWST_INSPIRED_REIMPLEMENTATION",
        details={"official_baseline": False},
    )
    return output


def offline_feasible_oracle_ceiling(
    *,
    target_scores: Mapping[str, np.ndarray],
    target_availability: Mapping[str, np.ndarray],
    target_expert_costs: Mapping[str, float],
    target_labels: np.ndarray,
) -> BaselinePrediction:
    """Construct the explicitly offline ceiling after all methods are frozen."""

    experts = sorted(target_scores)
    if (
        set(target_availability) != set(experts)
        or set(target_expert_costs) != set(experts)
    ):
        raise ValueError("offline oracle requires aligned scores, masks, and costs")
    matrix = np.column_stack([target_scores[name] for name in experts])
    validate_numpy_scores(matrix, ScoreType.PROBABILITY)
    availability = np.column_stack(
        [np.asarray(target_availability[name], dtype=bool) for name in experts]
    )
    if matrix.shape != availability.shape:
        raise ValueError("offline oracle scores and availability must align")
    labels = (np.asarray(target_labels).reshape(-1) == 1).astype(float)
    if len(labels) != len(matrix):
        raise ValueError("offline oracle labels must align with target scores")
    costs = np.asarray([target_expert_costs[name] for name in experts])
    errors = np.where(
        availability,
        np.abs(matrix - labels[:, None]),
        np.inf,
    )
    chosen = errors.argmin(axis=1)
    no_feasible = ~availability.any(axis=1)
    rows = np.arange(len(matrix))
    scores = matrix[rows, chosen]
    scores[no_feasible] = 0.0
    compute = costs[chosen]
    compute[no_feasible] = 0.0
    return BaselinePrediction(
        scores=scores,
        abstention_probability=no_feasible.astype(float),
        expected_compute=compute,
        offline_oracle=True,
        adapter="offline_feasible_oracle_ceiling",
        details={"target_labels_used_for_offline_ceiling_only": True},
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
    seed: int = 20260729,
    steps: int = 100,
    ablation: PilotAblation = PilotAblation.FULL,
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
    seed_everything(seed)
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
    review_fraction = next(
        (
            group.contract.budget.review_fraction
            for group in source_groups
            if group.contract.budget.review_fraction is not None
        ),
        0.01,
    )
    review_k = max(1, int(round(float(review_fraction) * len(train_target))))
    abstention_capacity = target_contract.budget.abstention_capacity
    if abstention_capacity is None:
        abstention_capacity = 0.1
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
            review_k=review_k,
            abstention_probability=(
                None
                if ablation is PilotAblation.NO_ABSTENTION
                else output.abstention_probability
            ),
            forced_abstention=~train_mask.any(dim=1),
            abstention_capacity=(
                None
                if ablation is PilotAblation.NO_ABSTENTION
                else abstention_capacity
            ),
            abstention_cost_value=0.2,
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
                review_k=max(
                    1,
                    int(round(float(review_fraction) * len(validation_target))),
                ),
                abstention_probability=(
                    None
                    if ablation is PilotAblation.NO_ABSTENTION
                    else validation_output.abstention_probability
                ),
                forced_abstention=~validation_mask.any(dim=1),
                abstention_capacity=(
                    None
                    if ablation is PilotAblation.NO_ABSTENTION
                    else abstention_capacity
                ),
                abstention_cost_value=0.2,
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
        selected_threshold = select_abstention_threshold(
            validation_losses,
            validation_output.abstention_probability,
            capacity=abstention_capacity,
            abstention_cost_value=0.2,
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
    if ablation is PilotAblation.NO_ABSTENTION:
        abstain = ~target_mask_array.any(axis=1)
    else:
        learned = target_output.abstention_probability >= threshold
        abstain = apply_abstention_capacity(
            target_output.abstention_probability,
            abstention_capacity,
            forced_abstention=target_output.all_experts_unavailable,
        )
        abstain = abstain | learned
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
        perturbation_flip_rate=float(
            np.mean(selected != perturbed_selected)
        ),
        ablation=ablation,
        source_train_examples=len(train_labels),
        source_validation_examples=len(validation_labels),
        abstention_threshold=threshold,
        abstention_threshold_fitted_on=threshold_fitted_on,
    )


def evaluate_saved_output_pilot(
    target_labels: np.ndarray,
    candidates: Mapping[str, BaselinePrediction],
    *,
    dataset: str,
    target_contract: str,
    seed: int,
    fold: str,
) -> dict[str, Any]:
    """Score frozen predictions after every fit/hyperparameter is frozen."""

    rows: list[dict[str, Any]] = []
    binary = (np.asarray(target_labels) == 1).astype(float)
    oracle = candidates.get("offline_feasible_oracle_ceiling")
    if oracle is None:
        raise ValueError("pilot evaluation requires the offline feasible oracle ceiling")
    oracle_risk = float(np.mean((oracle.scores - binary) ** 2))
    for method, prediction in sorted(candidates.items()):
        metrics = binary_metrics(target_labels, prediction.scores)
        decision = prediction.abstention_probability >= 0.5
        point_losses = (
            (prediction.scores >= 0.5).astype(float) - binary
        ) ** 2
        accepted = ~decision
        selective = (
            float(point_losses[accepted].mean()) if accepted.any() else 0.0
        )
        contract_risk = float(np.mean((prediction.scores - binary) ** 2))
        values = {
            "auprc": metrics["auprc"],
            "recall_at_0.5pct": recall_at_k(
                target_labels,
                prediction.scores,
                0.005,
            ),
            "recall_at_1pct": recall_at_k(
                target_labels,
                prediction.scores,
                0.01,
            ),
            "recall_at_2pct": recall_at_k(
                target_labels,
                prediction.scores,
                0.02,
            ),
            "budget_curve_area": budget_curve_auc(
                target_labels,
                prediction.scores,
                (0.005, 0.01, 0.02),
            ),
            "contract_regret": contract_risk - oracle_risk,
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
            ),
            "compute": float(np.mean(prediction.expected_compute)),
        }
        for metric, value in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "target_contract": target_contract,
                    "seed": int(seed),
                    "fold": fold,
                    "method": method,
                    "metric": metric,
                    "value": float(value),
                    "offline_oracle": prediction.offline_oracle,
                }
            )
    return {
        "schema": "coregraph_saved_output_pilot_v2",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": rows,
        "target_information": "labels_used_for_final_offline_scoring_only",
        "oracle_target_selection": False,
    }
