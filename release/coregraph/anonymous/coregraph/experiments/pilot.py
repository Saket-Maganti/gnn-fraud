"""Saved-output pilot without model retraining."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from coregraph.contracts.contract import DeploymentContract
from coregraph.evaluation.metrics import binary_metrics, recall_at_k
from coregraph.method import CoReGraph
from coregraph.tasks.base import align_prediction_rows
from coregraph.utils.seeding import seed_everything


@dataclass(frozen=True)
class PredictionArtifact:
    expert_id: str
    dataset: str
    contract_id: str
    task_type: str
    prediction_unit: str
    path: Path
    checksum: str
    config_hash: str
    contract_role: str
    deployment_contract: DeploymentContract


@dataclass(frozen=True)
class SavedRouterPrediction:
    scores: np.ndarray
    selected_experts: np.ndarray
    routing_weights: np.ndarray
    perturbation_flip_rate: float


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
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "expert_id",
            "dataset",
            "contract_id",
            "task_type",
            "prediction_unit",
            "prediction_path",
            "prediction_checksum",
            "config_hash",
            "contract_role",
            "deployment_contract",
        }
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
        artifacts.append(
            PredictionArtifact(
                expert_id=str(payload["expert_id"]),
                dataset=str(payload["dataset"]),
                contract_id=str(payload["contract_id"]),
                task_type=str(payload["task_type"]),
                prediction_unit=str(payload["prediction_unit"]),
                path=prediction_path,
                checksum=str(payload["prediction_checksum"]),
                config_hash=str(payload["config_hash"]),
                contract_role=str(payload["contract_role"]),
                deployment_contract=DeploymentContract.from_dict(
                    payload["deployment_contract"]
                ),
            )
        )
    return artifacts


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def align_artifact_group(
    artifacts: Sequence[PredictionArtifact],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not artifacts:
        raise ValueError("no prediction artifacts supplied")
    units = {artifact.prediction_unit for artifact in artifacts}
    datasets = {artifact.dataset for artifact in artifacts}
    contracts = {artifact.contract_id for artifact in artifacts}
    if len(units) != 1 or len(datasets) != 1 or len(contracts) != 1:
        raise ValueError("prediction alignment cannot pool task units datasets or contracts")
    unit = next(iter(units))
    id_column = f"{unit}_id"
    rows_by_expert = {
        artifact.expert_id: _read_rows(artifact.path) for artifact in artifacts
    }
    ids, scores, labels = align_prediction_rows(rows_by_expert, id_column=id_column)
    reference = rows_by_expert[artifacts[0].expert_id]
    split_map = {str(row[id_column]): str(row["split"]) for row in reference}
    splits = np.asarray([split_map[str(identifier)] for identifier in ids])
    return ids, scores, labels, splits


def baseline_scores(
    source_scores: Mapping[str, np.ndarray],
    source_labels: np.ndarray,
    source_split: np.ndarray,
    target_scores: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    experts = sorted(source_scores)
    matrix_source = np.column_stack([source_scores[name] for name in experts])
    matrix_target = np.column_stack([target_scores[name] for name in experts])
    source_binary = (np.asarray(source_labels) == 1).astype(int)
    validation = source_split == "validation"
    if not validation.any():
        raise ValueError("pilot requires source validation predictions")
    validation_risk = np.mean(
        (matrix_source[validation] - source_binary[validation, None]) ** 2,
        axis=0,
    )
    best = int(np.argmin(validation_risk))
    output = {
        f"expert:{name}": target_scores[name] for name in experts
    }
    output["simple_average"] = matrix_target.mean(axis=1)
    output["best_source_validation"] = matrix_target[:, best]
    feature_candidates = [
        index
        for index, name in enumerate(experts)
        if any(token in name.lower() for token in ("feature", "mlp", "logistic"))
    ]
    if feature_candidates:
        output["graphsafe_feature"] = matrix_target[:, feature_candidates[0]]
    if len(experts) >= 2:
        # Global validation weight is selected without target labels.
        best_weight, best_risk = 0.5, float("inf")
        for weight in np.linspace(0, 1, 21):
            blended = (
                weight * matrix_source[validation, 0]
                + (1 - weight) * matrix_source[validation, 1]
            )
            risk = float(np.mean((blended - source_binary[validation]) ** 2))
            if risk < best_risk:
                best_weight, best_risk = float(weight), risk
        output["global_validation_weight"] = (
            best_weight * matrix_target[:, 0]
            + (1 - best_weight) * matrix_target[:, 1]
        )
        output["no_contract_router"] = output["global_validation_weight"]
        output["atomic_contract_router"] = output["best_source_validation"]
        confidence = np.abs(matrix_target[:, 0] - 0.5)
        output["mowst_style_confidence_gate"] = np.where(
            confidence >= 0.25,
            matrix_target[:, 0],
            matrix_target[:, 1],
        )
        disagreement = np.abs(matrix_source[:, 0] - matrix_source[:, 1])
        features = np.column_stack([matrix_source, disagreement])
        trust = (
            np.abs(matrix_source[:, 0] - source_binary)
            < np.abs(matrix_source[:, 1] - source_binary)
        ).astype(int)
        train = np.isin(source_split, ["train", "validation"])
        if len(np.unique(trust[train])) == 2:
            gate = LogisticRegression(max_iter=1000, random_state=0)
            gate.fit(features[train], trust[train])
            target_features = np.column_stack(
                [matrix_target, np.abs(matrix_target[:, 0] - matrix_target[:, 1])]
            )
            weights = gate.predict_proba(target_features)[:, 1]
            output["logistic_gate"] = (
                weights * matrix_target[:, 0]
                + (1 - weights) * matrix_target[:, 1]
            )
    return output


def fit_saved_output_corerouter(
    source_groups: Sequence[
        tuple[DeploymentContract, Mapping[str, np.ndarray], np.ndarray, np.ndarray]
    ],
    *,
    target_contract: DeploymentContract,
    target_scores: Mapping[str, np.ndarray],
    seed: int = 20260729,
    steps: int = 100,
) -> SavedRouterPrediction:
    """Fit the pilot router on source labels and predict a label-free target.

    Expert predictions remain frozen. Target labels are absent from this
    signature by construction.
    """

    if len(source_groups) < 2:
        raise ValueError("full CoReRouter pilot requires at least two source contracts")
    expert_sets = [set(group_scores) for _, group_scores, _, _ in source_groups]
    experts = sorted(set.intersection(*expert_sets, set(target_scores)))
    if len(experts) < 2:
        raise ValueError("full CoReRouter pilot requires two aligned experts")
    contracts: list[DeploymentContract] = []
    score_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for contract, scores, labels, splits in source_groups:
        fit_mask = np.isin(splits, ("train", "validation"))
        if not fit_mask.any():
            raise ValueError(f"source contract {contract.contract_id} has no fit rows")
        matrix = np.column_stack([scores[name] for name in experts])
        score_rows.append(matrix[fit_mask])
        label_rows.append((labels[fit_mask] == 1).astype(float))
        contracts.extend([contract] * int(fit_mask.sum()))
    source_matrix = np.concatenate(score_rows)
    source_labels = np.concatenate(label_rows)

    def diagnostics(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [values.mean(axis=1), values.std(axis=1), values.max(axis=1) - values.min(axis=1)]
        )

    seed_everything(seed)
    model = CoReGraph(
        num_experts=len(experts),
        diagnostic_dim=3,
        axis_dropout=0.05,
        contract_noise_std=0.0,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    score_tensor = torch.tensor(source_matrix, dtype=torch.float32)
    diagnostic_tensor = torch.tensor(diagnostics(source_matrix), dtype=torch.float32)
    target_tensor = torch.tensor(source_labels, dtype=torch.float32)
    availability = torch.ones_like(score_tensor, dtype=torch.bool)
    model.train()
    for _ in range(steps):
        output = model(
            contracts=contracts,
            expert_scores=score_tensor,
            diagnostics=diagnostic_tensor,
            availability_mask=availability,
        )
        loss = torch.nn.functional.binary_cross_entropy(
            output.blended_score.clamp(1e-6, 1 - 1e-6),
            target_tensor,
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    target_matrix = np.column_stack([target_scores[name] for name in experts])
    model.eval()
    with torch.no_grad():
        output = model(
            contracts=[target_contract] * len(target_matrix),
            expert_scores=torch.tensor(target_matrix, dtype=torch.float32),
            diagnostics=torch.tensor(diagnostics(target_matrix), dtype=torch.float32),
            availability_mask=torch.ones(target_matrix.shape, dtype=torch.bool),
            expert_names=experts,
        )
        rng = np.random.default_rng(seed)
        perturbed_matrix = np.clip(
            target_matrix + rng.normal(0, 1e-3, target_matrix.shape),
            0,
            1,
        )
        perturbed = model(
            contracts=[target_contract] * len(target_matrix),
            expert_scores=torch.tensor(perturbed_matrix, dtype=torch.float32),
            diagnostics=torch.tensor(diagnostics(perturbed_matrix), dtype=torch.float32),
            availability_mask=torch.ones(target_matrix.shape, dtype=torch.bool),
            expert_names=experts,
        )
    selected = output.selected_expert.cpu().numpy()
    perturbed_selected = perturbed.selected_expert.cpu().numpy()
    return SavedRouterPrediction(
        scores=output.blended_score.cpu().numpy(),
        selected_experts=selected,
        routing_weights=output.expert_weights.cpu().numpy(),
        perturbation_flip_rate=float(np.mean(selected != perturbed_selected)),
    )


def evaluate_saved_output_pilot(
    target_labels: np.ndarray,
    candidate_scores: Mapping[str, np.ndarray],
    *,
    review_fraction: float = 0.01,
    dataset: str = "",
    contract_id: str = "",
) -> dict[str, Any]:
    rows = []
    for method, scores in sorted(candidate_scores.items()):
        metrics = binary_metrics(target_labels, scores)
        metrics["recall_at_1pct"] = recall_at_k(
            target_labels,
            scores,
            review_fraction,
        )
        rows.append(
            {
                "dataset": dataset,
                "contract_id": contract_id,
                "method": method,
                **metrics,
            }
        )
    return {
        "schema": "coregraph_saved_output_pilot_v1",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": rows,
        "target_information": "labels_used_for_final_offline_scoring_only",
        "oracle_target_selection": False,
    }
