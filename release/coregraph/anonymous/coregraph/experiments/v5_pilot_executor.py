"""Checksum-gated, target-blind, resumable V5 saved-output pilot execution."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import math
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from coregraph.contracts.axes import (
    AccessRegime,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    DeviceClass,
    MeasurementStatus,
    ResourceSpec,
    ReviewMode,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    VisibilityAxis,
    VisibilitySpec,
)
from coregraph.contracts.contract import DeploymentContract
from coregraph.evidence.archive_store import ArchiveIntegrityError, ArchiveStore
from coregraph.experiments.pilot import (
    PilotAblation,
    SavedSourceGroup,
    derive_router_seed,
    fit_saved_output_corerouter,
)
from coregraph.experiments.v5_pilot_outputs import (
    OUTPUT_SCHEMA_VERSION,
    atomic_write_npz,
    canonical_hash,
    coordinate_identity_hash,
    mark_complete,
    reusable_complete,
    sha256_path,
    write_checkpoint,
    write_failure,
)
from coregraph.experiments.v5_pilot_types import (
    EXPERT_ORDER,
    FrozenPilotPolicy,
    PilotCheckpoint,
    PilotCoordinate,
    PilotGateRecord,
    PilotStage,
    SourceEnvironmentBundle,
    TargetEvaluationBundle,
    TargetUnlabeledBundle,
    V5BaseArtifact,
    V5ScenarioMaterialization,
)
from coregraph.experiments.v5_scenario_loader import V5PilotConfig
from coregraph.utils.io import atomic_write_json


@dataclass(frozen=True, slots=True)
class _StreamRow:
    row_key: str
    score: float
    split: str
    label_known: bool
    label: int | None


@dataclass(frozen=True, slots=True)
class MethodInference:
    policy: FrozenPilotPolicy
    scores: np.ndarray
    routing_weights: np.ndarray
    abstain: np.ndarray
    selected_experts: np.ndarray
    expected_compute: np.ndarray
    route_summary: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", "", "nan", "unknown"}:
        return False
    raise ArchiveIntegrityError(f"invalid label_known value {value!r}")


def _stream_artifact(
    store: ArchiveStore,
    artifact: V5BaseArtifact,
    *,
    selected_splits: set[str],
    include_labels: bool,
) -> Iterator[_StreamRow]:
    """Read an archive member without permanent extraction or target-label exposure."""

    with store.open_member(
        artifact.archive_name,
        artifact.member_name,
        expected_sha256=artifact.member_sha256,
    ) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ArchiveIntegrityError(f"empty prediction member {artifact.member_name}") from exc
        positions = {name: index for index, name in enumerate(header)}
        required = {"dataset", "protocol", "seed", "split", "node_id", "score", "label_known"}
        if include_labels:
            required.add("y_true")
        missing = required - set(positions)
        if missing:
            raise ArchiveIntegrityError(
                f"prediction schema missing {sorted(missing)} in {artifact.member_name}"
            )
        split_mapping = dict(artifact.split_token_semantics)
        seen: set[str] = set()
        for values in reader:
            if len(values) != len(header):
                raise ArchiveIntegrityError(f"malformed CSV row in {artifact.member_name}")
            split_raw = values[positions["split"]]
            if split_raw not in split_mapping:
                raise ArchiveIntegrityError(f"unmapped provider split token {split_raw!r}")
            split = split_mapping[split_raw]
            if split not in selected_splits:
                continue
            known = _boolean(values[positions["label_known"]])
            if not known:
                continue
            provider_protocol = {
                "strict_inductive": "strict_inductive",
                "isolated_inductive": "inductive_isolated",
                "transductive_structure": "transductive",
            }[artifact.protocol]
            if (
                values[positions["dataset"]] != artifact.dataset
                or values[positions["protocol"]] != provider_protocol
                or int(values[positions["seed"]]) != artifact.provider_seed
            ):
                raise ArchiveIntegrityError("row coordinate conflicts with V5 artifact identity")
            raw_identifier = values[positions["node_id"]]
            row_key = ":".join(
                (
                    artifact.dataset,
                    str(artifact.provider_seed),
                    artifact.protocol,
                    split,
                    raw_identifier,
                )
            )
            if row_key in seen:
                raise ArchiveIntegrityError(f"duplicate composite row key {row_key}")
            seen.add(row_key)
            score = float(values[positions["score"]])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ArchiveIntegrityError("prediction score is not a finite probability")
            label = int(values[positions["y_true"]]) if include_labels else None
            yield _StreamRow(row_key, score, split, known, label)


def _artifacts_for_protocol(
    scenario: V5ScenarioMaterialization,
    protocol: str,
    role: str,
) -> tuple[V5BaseArtifact, ...]:
    bindings = scenario.source_bindings if role == "source" else scenario.target_bindings
    by_expert = {
        binding.expert: scenario.artifacts_by_coordinate[binding.base_coordinate_id]
        for binding in bindings
        if binding.protocol == protocol
    }
    if tuple(sorted(by_expert, key=EXPERT_ORDER.index)) != EXPERT_ORDER:
        raise ValueError("protocol binding does not contain the frozen expert order")
    return tuple(by_expert[expert] for expert in EXPERT_ORDER)


def _iter_aligned(
    store: ArchiveStore,
    artifacts: Sequence[V5BaseArtifact],
    *,
    selected_splits: set[str],
    include_labels: bool,
) -> Iterator[tuple[str, np.ndarray, str, int | None]]:
    streams = [
        _stream_artifact(
            store,
            artifact,
            selected_splits=selected_splits,
            include_labels=include_labels,
        )
        for artifact in artifacts
    ]
    for aligned in zip_longest(*streams):
        if any(item is None for item in aligned):
            raise ArchiveIntegrityError("expert members have different selected row counts")
        rows = tuple(item for item in aligned if item is not None)
        first = rows[0]
        if any(
            (row.row_key, row.split, row.label_known) !=
            (first.row_key, first.split, first.label_known)
            for row in rows[1:]
        ):
            raise ArchiveIntegrityError("expert members are not aligned by composite row key")
        if include_labels and any(row.label != first.label for row in rows[1:]):
            raise ArchiveIntegrityError("experts disagree on legally accessed provider labels")
        yield (
            first.row_key,
            np.asarray([row.score for row in rows], dtype=np.float32),
            first.split,
            first.label,
        )


def _availability(scenario: V5ScenarioMaterialization, config: V5PilotConfig, rows: int) -> np.ndarray:
    profiles = config.payload["resource_profiles"]
    profile = profiles.get(scenario.definition.resource_profile)
    if not isinstance(profile, Mapping):
        raise ValueError(f"unknown resource profile {scenario.definition.resource_profile!r}")
    unavailable = {str(value) for value in profile.get("unavailable_experts", ())}
    return np.broadcast_to(
        np.asarray([expert not in unavailable for expert in EXPERT_ORDER], dtype=bool),
        (rows, len(EXPERT_ORDER)),
    ).copy()


def assemble_source_environments(
    store: ArchiveStore,
    scenario: V5ScenarioMaterialization,
    config: V5PilotConfig,
) -> tuple[SourceEnvironmentBundle, ...]:
    """Take a stable hash sample per source environment/split with bounded memory."""

    cap = int(config.payload["streaming"]["source_rows_per_split_per_environment"])
    if cap <= 0:
        raise ValueError("source row cap must be positive")
    output: list[SourceEnvironmentBundle] = []
    for protocol in scenario.definition.source_protocols:
        artifacts = _artifacts_for_protocol(scenario, protocol, "source")
        heaps: dict[str, list[tuple[int, str, tuple[float, ...], int]]] = {
            "train": [],
            "validation": [],
        }
        for key, scores, split, label in _iter_aligned(
            store,
            artifacts,
            selected_splits={"train", "validation"},
            include_labels=True,
        ):
            if label is None:
                raise RuntimeError("source label reader returned no label")
            rank = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
            item = (-rank, key, tuple(float(value) for value in scores), int(label))
            heap = heaps[split]
            if len(heap) < cap:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        selected = sorted(heaps["train"], key=lambda item: (-item[0], item[1]))
        selected.extend(
            sorted(heaps["validation"], key=lambda item: (-item[0], item[1]))
        )
        train_keys = {item[1] for item in heaps["train"]}
        if not heaps["train"] or not heaps["validation"]:
            raise ValueError(f"source environment {protocol} lacks train or validation rows")
        row_keys = tuple(item[1] for item in selected)
        scores = np.asarray([item[2] for item in selected], dtype=np.float32)
        labels = np.asarray([item[3] for item in selected], dtype=np.int16)
        splits = np.asarray(
            ["train" if item[1] in train_keys else "validation" for item in selected],
            dtype="U10",
        )
        output.append(
            SourceEnvironmentBundle(
                environment_id=f"{scenario.definition.dataset}:{protocol}:seed{scenario.definition.provider_seed}",
                dataset=scenario.definition.dataset,
                protocol=protocol,
                provider_seed=scenario.definition.provider_seed,
                row_keys=row_keys,
                scores=scores,
                labels=labels,
                splits=splits,
                availability=_availability(scenario, config, len(row_keys)),
            )
        )
    return tuple(output)


def assemble_target_unlabeled(
    store: ArchiveStore,
    scenario: V5ScenarioMaterialization,
    config: V5PilotConfig,
) -> TargetUnlabeledBundle:
    """Build the target view while never indexing or returning the y_true column."""

    artifacts = _artifacts_for_protocol(scenario, scenario.definition.target_protocol, "target")
    keys: list[str] = []
    score_rows: list[np.ndarray] = []
    for key, scores, split, label in _iter_aligned(
        store,
        artifacts,
        selected_splits={"test"},
        include_labels=False,
    ):
        if label is not None or split != "test":
            raise RuntimeError("target-unlabelled reader crossed the label firewall")
        keys.append(key)
        score_rows.append(scores)
    matrix = np.asarray(score_rows, dtype=np.float32)
    return TargetUnlabeledBundle(
        dataset=scenario.definition.dataset,
        protocol=scenario.definition.target_protocol,
        provider_seed=scenario.definition.provider_seed,
        row_keys=tuple(keys),
        scores=matrix,
        availability=_availability(scenario, config, len(keys)),
    )


class TargetLabelVault:
    """Single-use evaluator that refuses to open labels before a valid freeze."""

    __slots__ = ("_store", "_scenario", "_opened")

    def __init__(self, store: ArchiveStore, scenario: V5ScenarioMaterialization) -> None:
        self._store = store
        self._scenario = scenario
        self._opened = False

    def __getstate__(self) -> Mapping[str, Any]:
        raise TypeError("TargetLabelVault cannot be serialized")

    def open_after_freeze(
        self,
        *,
        freeze_manifest_path: Path,
        target_scores_path: Path,
        expected_row_keys: tuple[str, ...],
    ) -> TargetEvaluationBundle:
        if self._opened:
            raise RuntimeError("TargetLabelVault is single-use")
        payload = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "coregraph_v5_policy_freeze_manifest_v1":
            raise RuntimeError("offline evaluation requires a V5 policy freeze manifest")
        if payload.get("target_labels_loaded") is not False:
            raise RuntimeError("policy freeze does not attest target-label blinding")
        if payload.get("target_score_sha256") != sha256_path(target_scores_path):
            raise RuntimeError("target score changed after policy freeze")
        artifacts = _artifacts_for_protocol(
            self._scenario,
            self._scenario.definition.target_protocol,
            "target",
        )
        keys: list[str] = []
        labels: list[int] = []
        for key, _, split, label in _iter_aligned(
            self._store,
            artifacts,
            selected_splits={"test"},
            include_labels=True,
        ):
            if split != "test" or label is None:
                raise RuntimeError("offline target label loader returned an invalid row")
            keys.append(key)
            labels.append(label)
        if tuple(keys) != expected_row_keys:
            raise RuntimeError("offline target labels do not align with frozen target scores")
        self._opened = True
        return TargetEvaluationBundle(
            dataset=self._scenario.definition.dataset,
            protocol=self._scenario.definition.target_protocol,
            provider_seed=self._scenario.definition.provider_seed,
            row_keys=tuple(keys),
            labels=np.asarray(labels, dtype=np.int16),
        )


def _deployment_contract(
    scenario: V5ScenarioMaterialization,
    protocol: str,
    role: ContractRole,
    config: V5PilotConfig,
) -> DeploymentContract:
    unavailable = tuple(
        str(value)
        for value in config.payload["resource_profiles"][scenario.definition.resource_profile].get(
            "unavailable_experts", ()
        )
    )
    return DeploymentContract(
        environment_id=(
            f"{scenario.definition.dataset}.{protocol}.seed{scenario.definition.provider_seed}.{role.value}"
        ),
        role=role,
        time=TimeSpec(TimeAxis.EARLY_TO_LATE),
        visibility=VisibilitySpec.from_v2(VisibilityAxis(protocol)),
        construction=ConstructionSpec(
            ConstructionAxis.NO_GRAPH
            if protocol == "isolated_inductive"
            else ConstructionAxis.FULL_GRAPH
        ),
        selection=SelectionAxis.NO_TARGET_ACCESS,
        budget=BudgetSpec(
            review_mode=ReviewMode.FRACTION,
            review_fraction=float(config.payload["review_fraction"]),
            abstention_capacity=float(config.payload["abstention"]["capacity"]),
        ),
        resource=ResourceSpec(
            device_class=DeviceClass.CPU,
            unavailable_experts=unavailable,
            measurement_status=MeasurementStatus.ESTIMATED,
        ),
        access_regime=AccessRegime.DG_NO_TARGET,
        dataset_id=scenario.definition.dataset,
        task_id="node_classification",
    )


def _source_fit_seed(scenario: V5ScenarioMaterialization, method: str) -> int:
    return derive_router_seed(scenario.definition.provider_seed, method)


def _method_inference(
    method: str,
    source: Sequence[SourceEnvironmentBundle],
    target: TargetUnlabeledBundle,
    scenario: V5ScenarioMaterialization,
    config: V5PilotConfig,
) -> MethodInference:
    if hasattr(target, "labels") or "label" in target.to_serializable():
        raise RuntimeError("target label firewall failed before fitting")
    costs = np.asarray(
        [float(config.payload["expert_relative_costs"][name]) for name in EXPERT_ORDER],
        dtype=np.float32,
    )
    available = target.availability.astype(bool)
    executable = available.any(axis=1)
    chunk_rows = int(config.payload["streaming"]["chunk_rows"])
    if chunk_rows <= 0:
        raise ValueError("target inference chunk size must be positive")
    slices = tuple(
        slice(start, min(start + chunk_rows, len(target.row_keys)))
        for start in range(0, len(target.row_keys), chunk_rows)
    )
    fit_seed = _source_fit_seed(scenario, method)
    state: dict[str, Any] = {}
    preprocessing: dict[str, Any] = {"fit_scope": "source_train_only"}
    threshold_state: dict[str, Any] = {"fit_scope": "source_validation_only"}
    fit_report: dict[str, Any] = {
        "source_train_rows": int(sum(np.sum(item.splits == "train") for item in source)),
        "source_validation_rows": int(
            sum(np.sum(item.splits == "validation") for item in source)
        ),
        "target_labels_available_to_fitter": False,
        "fit_seed": fit_seed,
    }
    if method == "uniform_average":
        weight_chunks = []
        score_chunks = []
        selected_chunks = []
        for chunk in slices:
            chunk_weights = available[chunk].astype(np.float32)
            denominator = chunk_weights.sum(axis=1, keepdims=True)
            chunk_weights = np.divide(
                chunk_weights,
                denominator,
                out=np.zeros_like(chunk_weights),
                where=denominator > 0,
            )
            weight_chunks.append(chunk_weights)
            score_chunks.append((chunk_weights * target.scores[chunk]).sum(axis=1))
            selected_chunks.append(
                np.where(executable[chunk], np.argmax(chunk_weights, axis=1), -1)
            )
        weights = np.concatenate(weight_chunks)
        scores = np.concatenate(score_chunks)
        selected = np.concatenate(selected_chunks)
        state = {"equal_weight_over_available_experts": True}
    elif method == "best_fixed_expert":
        risks = []
        for expert_index, expert in enumerate(EXPERT_ORDER):
            group_risks = []
            for group in source:
                keep = group.splits == "validation"
                feasible = group.availability[keep, expert_index]
                if not feasible.all():
                    group_risks.append(float("inf"))
                else:
                    labels = (group.labels[keep] == 1).astype(np.float32)
                    group_risks.append(float(np.mean((group.scores[keep, expert_index] - labels) ** 2)))
            risks.append(float(np.mean(group_risks)))
        if not np.isfinite(risks).any():
            raise RuntimeError("no expert is feasible across source validation environments")
        chosen = int(np.argmin(np.asarray(risks)))
        weights = np.zeros_like(target.scores, dtype=np.float32)
        score_chunks = []
        selected_chunks = []
        for chunk in slices:
            weights[chunk, chosen] = available[chunk, chosen]
            score_chunks.append(target.scores[chunk, chosen].copy())
            selected_chunks.append(
                np.where(available[chunk, chosen], chosen, -1)
            )
        scores = np.concatenate(score_chunks)
        selected = np.concatenate(selected_chunks)
        executable = available[:, chosen]
        state = {
            "selected_expert": EXPERT_ORDER[chosen],
            "selection_criterion": "mean_environment_balanced_validation_brier",
            "source_validation_risks": dict(zip(EXPERT_ORDER, risks, strict=True)),
            "tie_break": "frozen_expert_order",
        }
    elif method == "source_logistic_gate":
        train_x: list[np.ndarray] = []
        train_y: list[np.ndarray] = []
        validation: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for group in source:
            for split in ("train", "validation"):
                keep = group.splits == split
                matrix = group.scores[keep].astype(np.float64)
                labels = (group.labels[keep] == 1).astype(np.float64)
                mask = group.availability[keep]
                errors = np.where(mask, np.abs(matrix - labels[:, None]), np.inf)
                features = np.column_stack((matrix, matrix.std(axis=1)))
                if split == "train":
                    train_x.append(features)
                    train_y.append(errors.argmin(axis=1))
                else:
                    validation.append((features, labels, mask))
        x_train = np.concatenate(train_x)
        y_train = np.concatenate(train_y)
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0] = 1.0
        x_scaled = (x_train - mean) / scale
        candidates = tuple(float(value) for value in config.payload["validation_selection"]["logistic_c"])
        best: tuple[float, float, LogisticRegression] | None = None
        for c_value in candidates:
            gate = LogisticRegression(
                C=c_value,
                max_iter=int(config.payload["optimization"]["logistic_max_iter"]),
                random_state=fit_seed,
            )
            gate.fit(x_scaled, y_train)
            risks = []
            for features, labels, mask in validation:
                probabilities = np.zeros((len(features), len(EXPERT_ORDER)), dtype=float)
                predicted = gate.predict_proba((features - mean) / scale)
                probabilities[:, gate.classes_.astype(int)] = predicted
                probabilities *= mask
                denominator = probabilities.sum(axis=1, keepdims=True)
                probabilities = np.divide(
                    probabilities,
                    denominator,
                    out=np.zeros_like(probabilities),
                    where=denominator > 0,
                )
                prediction = (probabilities * features[:, : len(EXPERT_ORDER)]).sum(axis=1)
                risks.append(float(np.mean((prediction - labels) ** 2)))
            record = (float(np.mean(risks)), c_value, gate)
            if best is None or record[:2] < best[:2]:
                best = record
        if best is None:
            raise RuntimeError("source logistic validation produced no candidate")
        validation_risk, selected_c, gate = best
        weight_chunks = []
        score_chunks = []
        selected_chunks = []
        for chunk in slices:
            chunk_scores = target.scores[chunk]
            target_features = np.column_stack(
                (chunk_scores, chunk_scores.std(axis=1))
            )
            probabilities = np.zeros_like(chunk_scores, dtype=np.float64)
            probabilities[:, gate.classes_.astype(int)] = gate.predict_proba(
                (target_features - mean) / scale
            )
            probabilities *= available[chunk]
            denominator = probabilities.sum(axis=1, keepdims=True)
            chunk_weights = np.divide(
                probabilities,
                denominator,
                out=np.zeros_like(probabilities),
                where=denominator > 0,
            ).astype(np.float32)
            weight_chunks.append(chunk_weights)
            score_chunks.append((chunk_weights * chunk_scores).sum(axis=1))
            selected_chunks.append(
                np.where(executable[chunk], np.argmax(chunk_weights, axis=1), -1)
            )
        weights = np.concatenate(weight_chunks)
        scores = np.concatenate(score_chunks)
        selected = np.concatenate(selected_chunks)
        state = {
            "classes": gate.classes_.astype(int).tolist(),
            "coef": gate.coef_.tolist(),
            "intercept": gate.intercept_.tolist(),
            "selected_c": selected_c,
            "converged": bool(np.all(gate.n_iter_ < gate.max_iter)),
            "feature_order": [*EXPERT_ORDER, "expert_score_std"],
        }
        preprocessing = {
            "fit_scope": "source_train_only",
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
        }
        fit_report["source_validation_brier"] = validation_risk
    elif method == "coregraph":
        source_groups = []
        for group in source:
            source_groups.append(
                SavedSourceGroup(
                    contract=_deployment_contract(
                        scenario, group.protocol, ContractRole.SOURCE, config
                    ),
                    scores={
                        expert: group.scores[:, index]
                        for index, expert in enumerate(EXPERT_ORDER)
                    },
                    labels=group.labels,
                    splits=group.splits,
                    availability={
                        expert: group.availability[:, index]
                        for index, expert in enumerate(EXPERT_ORDER)
                    },
                    expert_costs=dict(zip(EXPERT_ORDER, costs.tolist(), strict=True)),
                )
            )
        prediction = fit_saved_output_corerouter(
            source_groups,
            target_contract=_deployment_contract(
                scenario, scenario.definition.target_protocol, ContractRole.TARGET, config
            ),
            target_scores={
                expert: target.scores[:, index] for index, expert in enumerate(EXPERT_ORDER)
            },
            target_availability={
                expert: target.availability[:, index]
                for index, expert in enumerate(EXPERT_ORDER)
            },
            target_expert_costs=dict(zip(EXPERT_ORDER, costs.tolist(), strict=True)),
            expert_prediction_seed=scenario.definition.provider_seed,
            steps=int(config.payload["optimization"]["coregraph_steps"]),
            ablation=PilotAblation.FULL,
            abstention_cost=float(config.payload["abstention"]["cost"]),
            target_chunk_rows=chunk_rows,
        )
        weights = prediction.routing_weights.astype(np.float32)
        scores = prediction.scores.astype(np.float32)
        selected = prediction.selected_experts.astype(np.int16)
        executable = ~prediction.forced_abstention
        state = {
            "architecture": "factorised_contract_encoder_plus_resource_aware_corerouter",
            "source_fit_hash": prediction.source_fit_hash,
            "model_state": prediction.frozen_model_state,
            "early_stopping_source_only": prediction.early_stopping_source_only,
        }
        threshold_state = {
            "fit_scope": "source_validation_only",
            "abstention_threshold": prediction.abstention_threshold,
            "provenance": prediction.abstention_threshold_fitted_on,
        }
        fit_report.update(
            {
                "perturbation_flip_rate": prediction.perturbation_flip_rate,
                "source_train_rows": prediction.source_train_examples,
                "source_validation_rows": prediction.source_validation_examples,
            }
        )
        policy = FrozenPilotPolicy(
            method=method,
            scenario_fingerprint=scenario.scenario_fingerprint,
            provider_seed=scenario.definition.provider_seed,
            fit_seed=fit_seed,
            state=state,
            preprocessing_state=preprocessing,
            threshold_state=threshold_state,
            fit_report=fit_report,
        )
        route_summary = _route_summary(
            weights, selected, prediction.abstain, prediction.expected_compute, fit_report
        )
        return MethodInference(
            policy=policy,
            scores=scores,
            routing_weights=weights,
            abstain=np.asarray(prediction.abstain, dtype=bool),
            selected_experts=selected,
            expected_compute=prediction.expected_compute.astype(np.float32),
            route_summary=route_summary,
        )
    else:
        raise ValueError(f"unknown primary method {method!r}")
    abstain = ~executable
    expected_compute = (weights * costs[None, :]).sum(axis=1).astype(np.float32)
    policy = FrozenPilotPolicy(
        method=method,
        scenario_fingerprint=scenario.scenario_fingerprint,
        provider_seed=scenario.definition.provider_seed,
        fit_seed=fit_seed,
        state=state,
        preprocessing_state=preprocessing,
        threshold_state=threshold_state,
        fit_report=fit_report,
    )
    return MethodInference(
        policy=policy,
        scores=np.asarray(scores, dtype=np.float32),
        routing_weights=np.asarray(weights, dtype=np.float32),
        abstain=np.asarray(abstain, dtype=bool),
        selected_experts=np.asarray(selected, dtype=np.int16),
        expected_compute=expected_compute,
        route_summary=_route_summary(weights, selected, abstain, expected_compute, fit_report),
    )


def _route_summary(
    weights: np.ndarray,
    selected: np.ndarray,
    abstain: np.ndarray,
    expected_compute: np.ndarray,
    fit_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    counts = {
        expert: int(np.sum(selected == index)) for index, expert in enumerate(EXPERT_ORDER)
    }
    counts["unavailable_or_abstain"] = int(np.sum(selected < 0))
    return {
        "row_count": int(len(selected)),
        "selection_counts": counts,
        "mean_weights": dict(
            zip(EXPERT_ORDER, np.mean(weights, axis=0).astype(float).tolist(), strict=True)
        ),
        "abstained_count": int(np.sum(abstain)),
        "coverage": float(np.mean(~abstain)),
        "mean_relative_compute": float(np.mean(expected_compute)),
        "perturbation_flip_rate": fit_report.get("perturbation_flip_rate"),
    }


def _evaluate(
    inference: MethodInference,
    target: TargetUnlabeledBundle,
    evaluation: TargetEvaluationBundle,
    scenario: V5ScenarioMaterialization,
    config: V5PilotConfig,
) -> Mapping[str, float]:
    if evaluation.row_keys != target.row_keys:
        raise RuntimeError("offline evaluation rows do not match target-unlabelled rows")
    labels = (evaluation.labels == 1).astype(np.int8)
    accepted = ~inference.abstain
    if not accepted.any():
        auprc = float("nan")
        selective_risk = float("nan")
    else:
        auprc = float(average_precision_score(labels, inference.scores))
        predicted = inference.scores[accepted] >= 0.5
        selective_risk = float(np.mean(predicted != labels[accepted]))
    review_k = max(1, int(math.ceil(len(labels) * scenario.definition.review_fraction)))
    order = np.argsort(-inference.scores, kind="stable")[:review_k]
    positives = int(labels.sum())
    recall = float(labels[order].sum() / positives) if positives else float("nan")
    expert_risks = np.mean((target.scores - labels[:, None]) ** 2, axis=0)
    feasible = target.availability.all(axis=0)
    oracle_risk = float(np.min(np.where(feasible, expert_risks, np.inf)))
    abstention_cost = float(config.payload["abstention"]["cost"])
    method_loss = np.where(
        inference.abstain,
        abstention_cost,
        (inference.scores - labels) ** 2,
    )
    return {
        "auprc": auprc,
        "recall_at_frozen_review_fraction": recall,
        "selective_risk": selective_risk,
        "coverage": float(np.mean(accepted)),
        "contract_brier_risk": float(np.mean(method_loss)),
        "contract_regret": float(np.mean(method_loss) - oracle_risk),
        "feasible_best_expert_oracle_brier": oracle_risk,
        "mean_relative_compute": float(np.mean(inference.expected_compute)),
        "resource_feasible": float(np.all(target.availability.any(axis=1))),
    }


def execute_coordinate(
    *,
    coordinate: PilotCoordinate,
    scenario: V5ScenarioMaterialization,
    source: Sequence[SourceEnvironmentBundle],
    target: TargetUnlabeledBundle,
    store: ArchiveStore,
    config: V5PilotConfig,
    output_root: Path,
    code_sha: str,
    dependency_lock_sha256: str,
    resume: bool,
) -> Mapping[str, Any]:
    method_root = output_root / "scenarios" / coordinate.scenario_id / "methods" / coordinate.method
    method_root.mkdir(parents=True, exist_ok=True)
    identity_hash = coordinate_identity_hash(
        coordinate,
        code_sha=code_sha,
        config_sha256=config.config_sha256,
        preregistration_sha256=config.preregistration_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    existing = reusable_complete(
        method_root, coordinate=coordinate, identity_hash=identity_hash
    )
    if resume and existing[0]:
        return json.loads((method_root / "evaluation.json").read_text(encoding="utf-8"))
    checkpoint = PilotCheckpoint(
        coordinate_key=coordinate.key,
        identity_hash=identity_hash,
        stage=PilotStage.SOURCE_ASSEMBLED,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        retry_count=(0 if existing[0] else 1 if (method_root / "checkpoint.json").exists() else 0),
    )
    write_checkpoint(method_root, checkpoint)
    current_stage = PilotStage.SOURCE_ASSEMBLED
    try:
        inference = _method_inference(
            coordinate.method, source, target, scenario, config
        )
        current_stage = PilotStage.POLICY_FITTED
        write_checkpoint(
            method_root,
            PilotCheckpoint(
                coordinate.key,
                identity_hash,
                current_stage,
                OUTPUT_SCHEMA_VERSION,
                retry_count=checkpoint.retry_count,
            ),
        )
        fit_path = method_root / "fit_report.json"
        atomic_write_json(
            fit_path,
            {
                "schema": "coregraph_v5_fit_report_v1",
                "method": coordinate.method,
                "policy_state_hash": inference.policy.policy_state_hash,
                **dict(inference.policy.fit_report),
            },
        )
        current_stage = PilotStage.POLICY_FROZEN
        scores_path = method_root / "target_scores.npz"
        atomic_write_npz(
            scores_path,
            row_keys=np.asarray(target.row_keys),
            scores=inference.scores.astype(np.float32),
            routing_weights=inference.routing_weights.astype(np.float32),
            abstain=inference.abstain.astype(np.bool_),
            selected_experts=inference.selected_experts.astype(np.int16),
            expected_compute=inference.expected_compute.astype(np.float32),
        )
        route_path = method_root / "route_summary.json"
        atomic_write_json(route_path, dict(inference.route_summary))
        freeze_path = method_root / "POLICY_FREEZE_MANIFEST.json"
        source_artifacts = [
            scenario.artifacts_by_coordinate[item.base_coordinate_id].identity_dict()
            for item in scenario.source_bindings
        ]
        target_artifacts = [
            scenario.artifacts_by_coordinate[item.base_coordinate_id].identity_dict()
            for item in scenario.target_bindings
        ]
        freeze_payload = {
            "schema": "coregraph_v5_policy_freeze_manifest_v1",
            "code_sha": code_sha,
            "config_sha256": config.config_sha256,
            "preregistration_sha256": config.preregistration_sha256,
            "scenario_fingerprint": scenario.scenario_fingerprint,
            "source_artifacts": source_artifacts,
            "target_artifacts": target_artifacts,
            "preprocessing_state": inference.policy.preprocessing_state,
            "preprocessing_sha256": canonical_hash(inference.policy.preprocessing_state),
            "policy_state": inference.policy.state,
            "policy_state_sha256": inference.policy.policy_state_hash,
            "threshold_state": inference.policy.threshold_state,
            "threshold_sha256": canonical_hash(inference.policy.threshold_state),
            "target_score_sha256": sha256_path(scores_path),
            "target_row_key_sha256": canonical_hash(target.row_keys),
            "timestamp_utc": _utc_now(),
            "provider_prediction_seed": scenario.definition.provider_seed,
            "router_model_seed": inference.policy.fit_seed,
            "data_order_seed": inference.policy.fit_seed,
            "bootstrap_seed": int(config.payload["determinism"]["bootstrap_seed"]),
            "target_labels_loaded": False,
            "target_labels_available_to_fit_or_inference": False,
        }
        atomic_write_json(freeze_path, freeze_payload)
        current_stage = PilotStage.TARGET_SCORED
        write_checkpoint(
            method_root,
            PilotCheckpoint(
                coordinate.key,
                identity_hash,
                current_stage,
                OUTPUT_SCHEMA_VERSION,
                checksums={
                    "target_scores.npz": sha256_path(scores_path),
                    "POLICY_FREEZE_MANIFEST.json": sha256_path(freeze_path),
                },
                retry_count=checkpoint.retry_count,
            ),
        )
        evaluation_bundle = TargetLabelVault(store, scenario).open_after_freeze(
            freeze_manifest_path=freeze_path,
            target_scores_path=scores_path,
            expected_row_keys=target.row_keys,
        )
        metrics = _evaluate(inference, target, evaluation_bundle, scenario, config)
        current_stage = PilotStage.EVALUATED
        evaluation_path = method_root / "evaluation.json"
        evaluation_payload = {
            "schema": "coregraph_v5_pilot_method_result_v1",
            "coordinate": asdict(coordinate),
            "coordinate_key": coordinate.key,
            "execution_status": "COMPLETE",
            "metrics": metrics,
            "route_summary": inference.route_summary,
            "policy_freeze_sha256": sha256_path(freeze_path),
            "target_score_sha256": sha256_path(scores_path),
            "target_labels_loaded_only_by_offline_evaluator": True,
            "target_oracle_is_offline_diagnostic_only": True,
        }
        atomic_write_json(evaluation_path, evaluation_payload)
        mark_complete(
            method_root,
            coordinate=coordinate,
            identity_hash=identity_hash,
            outputs=(
                "fit_report.json",
                "POLICY_FREEZE_MANIFEST.json",
                "target_scores.npz",
                "route_summary.json",
                "evaluation.json",
            ),
            retry_count=checkpoint.retry_count,
        )
        return evaluation_payload
    except BaseException as exc:
        write_failure(
            method_root,
            coordinate=coordinate,
            identity_hash=identity_hash,
            stage=current_stage,
            exception=exc,
            traceback_text=traceback.format_exc(),
            retry_count=checkpoint.retry_count,
        )
        raise


def compute_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    coordinates: Sequence[PilotCoordinate],
    config: V5PilotConfig,
) -> Mapping[str, Any]:
    expected = {item.key for item in coordinates}
    observed = {str(item.get("coordinate_key", "")) for item in results}
    reasons: list[str] = []
    if len(results) != len(expected) or observed != expected:
        reasons.append("required_coordinate_set_incomplete_or_duplicated")
    by_cell: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for item in results:
        coordinate = item.get("coordinate", {})
        cell = (
            str(coordinate.get("dataset")),
            str(coordinate.get("target_protocol")),
            int(coordinate.get("provider_seed", -1)),
        )
        method = str(coordinate.get("method"))
        if method in by_cell.setdefault(cell, {}):
            reasons.append("duplicated_method_within_paired_cell")
        by_cell[cell][method] = item
    if any(set(methods) != set(config.methods) for methods in by_cell.values()):
        reasons.append("paired_cell_method_family_malformed")
    if reasons:
        record = PilotGateRecord(
            "INCONCLUSIVE", tuple(sorted(set(reasons))), len(results), config.preregistration_sha256
        )
        return {"schema": "coregraph_v5_pilot_gate_v1", **asdict(record)}
    minimum_regret = float(config.payload["gate"]["minimum_contract_regret_improvement"])
    harm_floor = float(config.payload["gate"]["average_auprc_harm_floor"])
    comparisons: dict[str, Any] = {}
    no_go = False
    for baseline in config.methods[1:]:
        regret_improvements = []
        auprc_deltas = []
        for methods in by_cell.values():
            core = methods["coregraph"]["metrics"]
            other = methods[baseline]["metrics"]
            regret_improvements.append(float(other["contract_regret"]) - float(core["contract_regret"]))
            auprc_deltas.append(float(core["auprc"]) - float(other["auprc"]))
        comparisons[baseline] = {
            "mean_contract_regret_improvement": float(np.mean(regret_improvements)),
            "worst_contract_regret_improvement": float(np.min(regret_improvements)),
            "mean_auprc_delta": float(np.mean(auprc_deltas)),
        }
        if np.min(regret_improvements) < minimum_regret or np.mean(auprc_deltas) < harm_floor:
            no_go = True
    outcome = "NO_GO" if no_go else "GO"
    record = PilotGateRecord(
        outcome,
        ("frozen_paired_gate_passed" if outcome == "GO" else "frozen_effect_gate_failed",),
        len(results),
        config.preregistration_sha256,
    )
    return {"schema": "coregraph_v5_pilot_gate_v1", **asdict(record), "comparisons": comparisons}
