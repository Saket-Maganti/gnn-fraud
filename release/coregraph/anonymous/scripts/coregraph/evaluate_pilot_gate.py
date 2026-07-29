#!/usr/bin/env python3
"""Evaluate frozen pilot gates and the V4 no-training completeness surface."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evaluation.corrections import holm  # noqa: E402
from coregraph.evaluation.statistics import (  # noqa: E402
    bootstrap_seed_blocks,
    exact_wilcoxon,
    hierarchical_dataset_bootstrap,
    paired_permutation,
)
from coregraph.experiments.pilot import (  # noqa: E402
    MethodExecutionStatus,
    PILOT_RESULT_ROW_FIELDS,
    derive_router_seed,
)

REQUIRED_BASELINES = {
    "average_all_feasible",
    "best_source_validation",
    "source_validation_convex_mixture",
    "graphsafe_confidence_abstention_component",
    "current_graph_feature_gate_adapter",
    "learned_no_contract_router",
    "learned_atomic_contract_router",
    "MOWST_INSPIRED_REIMPLEMENTATION",
}
REQUIRED_ABLATIONS = {
    "ablation:no_contract",
    "ablation:atomic_contract",
    "ablation:no_regret",
    "ablation:no_budget",
    "ablation:no_resource_mask",
    "ablation:no_stability",
    "ablation:no_abstention",
    "ablation:no_diagnostics",
}
REQUIRED_DATASETS = {"elliptic", "dgraphfin"}
REQUIRED_TARGET_CONTRACTS = {
    "strict_inductive",
    "isolated_inductive",
    "transductive_structure",
}
REQUIRED_EXPERTS = {"feature_mlp", "gcn", "graphsage"}
REQUIRED_CONTRACT_METRICS = {
    "auprc",
    "recall_at_0.5pct",
    "recall_at_1pct",
    "recall_at_2pct",
    "budget_curve_area",
    "contract_regret",
    "selective_risk",
    "coverage",
    "aurc",
    "abstention_cost",
    "compute",
}
V4_STRONG_BASELINES = {
    "average_all_feasible",
    "best_source_validation",
    "source_validation_convex_mixture",
    "current_graph_feature_gate_adapter",
    "learned_no_contract_router",
    "learned_atomic_contract_router",
    "MOWST_INSPIRED_REIMPLEMENTATION",
}
V4_COMPATIBILITY_COMPONENTS = {
    "graphsafe_confidence_abstention_component",
}
V4_OFFLINE_ORACLES = {
    "contract_feasible_oracle",
    "instance_clairvoyant_oracle_ceiling",
}
V4_REQUIRED_CONTRACT_METRICS = {
    "auprc",
    "recall_at_0.5pct",
    "recall_at_1pct",
    "recall_at_2pct",
    "budget_curve_area",
    "brier_contract_regret",
    "selective_zero_one_risk",
    "coverage",
    "aurc",
    "abstention_cost",
    "compute",
}
HIGHER_IS_BETTER = {
    "auprc",
    "recall_at_0.5pct",
    "recall_at_1pct",
    "recall_at_2pct",
    "budget_curve_area",
}
LOWER_IS_BETTER = {
    "contract_regret",
    "mean_regret",
    "maximum_regret",
    "cvar_regret",
    "selective_risk",
    "abstention_cost",
    "compute",
}
ROBUST_OUTCOMES = {"mean_regret", "maximum_regret", "cvar_regret"}
V4_ROBUST_OUTCOMES = {
    "mean_brier_contract_regret",
    "maximum_brier_contract_regret",
    "cvar_brier_contract_regret",
}


def _expert_seed(row: Mapping[str, Any]) -> int:
    if "expert_prediction_seed" in row:
        return int(row["expert_prediction_seed"])
    return int(row["seed"])


def _target_protocol(row: Mapping[str, Any]) -> str:
    if "target_protocol_id" in row:
        return str(row["target_protocol_id"])
    return str(row["target_contract"])


def _schema_target_protocols(schema: Mapping[str, Any]) -> tuple[str, ...]:
    values = schema.get(
        "required_target_protocols",
        schema.get("required_target_contracts", ()),
    )
    return tuple(str(value) for value in values)


def _schema_strong_baselines(schema: Mapping[str, Any]) -> tuple[str, ...]:
    values = schema.get(
        "strong_baselines",
        schema.get("required_strong_baselines", ()),
    )
    return tuple(str(value) for value in values)


def _holm_family(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "ranking_and_budget"
    if metric in ROBUST_OUTCOMES | {"contract_regret"}:
        return "robust_risk"
    if metric in {"selective_risk", "abstention_cost", "compute"}:
        return "deployment"
    raise ValueError(f"metric {metric!r} has no declared Holm family")


def _holm_family_v4(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "ranking_and_budget"
    if metric in V4_ROBUST_OUTCOMES | {"brier_contract_regret"}:
        return "headline_brier_risk"
    if metric in {
        "selective_zero_one_risk",
        "abstention_cost",
        "compute",
    }:
        return "deployment"
    raise ValueError(f"V4 metric {metric!r} has no declared Holm family")


def _validate_gate_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") == "coregraph_pilot_gate_v4":
        _validate_gate_schema_v4(schema)
        return
    if schema.get("schema_version") != "coregraph_pilot_gate_v3":
        raise ValueError("pilot gate must use the frozen V3 schema")
    if set(schema.get("required_datasets", ())) != REQUIRED_DATASETS:
        raise ValueError("pilot gate must require Elliptic and DGraphFin")
    if (
        set(schema.get("required_target_contracts", ()))
        != REQUIRED_TARGET_CONTRACTS
    ):
        raise ValueError("pilot gate target-contract registry does not match code")
    if tuple(schema.get("required_expert_prediction_seeds", ())) != tuple(
        range(1, 11)
    ):
        raise ValueError("pilot gate must require expert-prediction seeds 1-10")
    if set(schema.get("required_folds", ())) != {"fold0"}:
        raise ValueError("pilot gate must require the frozen fold")
    if set(schema.get("required_experts", ())) != REQUIRED_EXPERTS:
        raise ValueError("pilot gate expert registry does not match code")
    if set(schema.get("required_strong_baselines", ())) != REQUIRED_BASELINES:
        raise ValueError("pilot gate schema baseline registry does not match code")
    declared_ablations = {
        f"ablation:{name}"
        for name in schema.get("required_ablations", ())
    }
    if declared_ablations != REQUIRED_ABLATIONS:
        raise ValueError("pilot gate schema ablations do not match code")
    if (
        set(schema.get("required_contract_metrics", ()))
        != REQUIRED_CONTRACT_METRICS
    ):
        raise ValueError("pilot gate contract metrics do not match code")
    declared_families = schema.get("holm_families", {})
    if not isinstance(declared_families, dict):
        raise ValueError("pilot gate schema Holm families must be a mapping")
    if set(declared_families) != {
        "ranking_and_budget",
        "robust_risk",
        "deployment",
    }:
        raise ValueError("pilot gate schema must declare all three Holm families")
    for family, metrics in declared_families.items():
        if any(_holm_family(str(metric)) != family for metric in metrics):
            raise ValueError("pilot gate schema Holm family mapping is inconsistent")
    ablation_tests = schema.get("ablation_tests", {})
    if {
        f"ablation:{name}" for name in ablation_tests
    } != REQUIRED_ABLATIONS:
        raise ValueError("pilot gate must freeze one effect test per ablation")
    expected_meaningful = {
        "no_contract",
        "atomic_contract",
        "no_regret",
        "no_budget",
        "no_resource_mask",
    }
    if set(schema.get("required_meaningful_ablations", ())) != expected_meaningful:
        raise ValueError("pilot gate meaningful-ablation registry does not match code")
    thresholds = schema.get("effect_thresholds", {})
    if (
        float(thresholds.get("worst_contract_regret_improvement", 0.0)) <= 0
        or float(thresholds.get("robust_primary_improvement", 0.0)) <= 0
    ):
        raise ValueError("pilot gate robust effect thresholds must be positive")


def _validate_gate_schema_v4(schema: Mapping[str, Any]) -> None:
    if set(schema.get("required_datasets", ())) != REQUIRED_DATASETS:
        raise ValueError("V4 pilot gate must require Elliptic and DGraphFin")
    if set(_schema_target_protocols(schema)) != REQUIRED_TARGET_CONTRACTS:
        raise ValueError("V4 pilot gate protocol registry does not match code")
    if tuple(schema.get("required_expert_prediction_seeds", ())) != tuple(
        range(1, 11)
    ):
        raise ValueError("V4 pilot gate must require expert-prediction seeds 1-10")
    if set(schema.get("required_folds", ())) != {"fold0"}:
        raise ValueError("V4 pilot gate must require the frozen fold")
    if set(schema.get("required_experts", ())) != REQUIRED_EXPERTS:
        raise ValueError("V4 pilot gate expert registry does not match code")
    if set(_schema_strong_baselines(schema)) != V4_STRONG_BASELINES:
        raise ValueError("V4 strong-baseline registry does not match code")
    if (
        set(schema.get("compatibility_components", ()))
        != V4_COMPATIBILITY_COMPONENTS
    ):
        raise ValueError("V4 compatibility-component registry does not match code")
    if set(schema.get("offline_oracles", ())) != V4_OFFLINE_ORACLES:
        raise ValueError("V4 offline-oracle registry does not match code")
    if set(schema.get("diagnostic_comparators", ())) & (
        V4_STRONG_BASELINES | V4_COMPATIBILITY_COMPONENTS | V4_OFFLINE_ORACLES
    ):
        raise ValueError("V4 comparator taxonomy categories must be disjoint")
    declared_ablations = {
        f"ablation:{name}"
        for name in schema.get("required_ablations", ())
    }
    if declared_ablations != REQUIRED_ABLATIONS:
        raise ValueError("V4 pilot gate ablations do not match code")
    if (
        set(schema.get("required_contract_metrics", ()))
        != V4_REQUIRED_CONTRACT_METRICS
    ):
        raise ValueError("V4 pilot gate contract metrics do not match code")
    if schema.get("headline_risk") != "brier_contract_regret":
        raise ValueError("V4 pilot gate headline risk must be Brier contract regret")
    declared_families = schema.get("holm_families", {})
    if not isinstance(declared_families, Mapping):
        raise ValueError("V4 pilot gate Holm families must be a mapping")
    if set(declared_families) != {
        "ranking_and_budget",
        "headline_brier_risk",
        "deployment",
    }:
        raise ValueError("V4 pilot gate must declare all three Holm families")
    for family, metrics in declared_families.items():
        if any(_holm_family_v4(str(metric)) != family for metric in metrics):
            raise ValueError("V4 pilot gate Holm family mapping is inconsistent")
    ablation_tests = schema.get("ablation_tests", {})
    if {
        f"ablation:{name}" for name in ablation_tests
    } != REQUIRED_ABLATIONS:
        raise ValueError("V4 pilot gate must freeze one test per ablation")
    expected_meaningful = {
        "no_contract",
        "atomic_contract",
        "no_regret",
        "no_budget",
        "no_resource_mask",
    }
    if set(schema.get("required_meaningful_ablations", ())) != expected_meaningful:
        raise ValueError("V4 meaningful-ablation registry does not match code")
    inference = set(schema.get("inference", ()))
    required_inference = {
        "dataset_stratified_exact_wilcoxon",
        "dataset_stratified_paired_permutation",
        "dataset_stratified_seed_block_bootstrap",
        "hierarchical_dataset_seed_bootstrap_secondary",
    }
    if inference != required_inference:
        raise ValueError("V4 dataset-stratified inference stack is not frozen")
    if schema.get("inferential_block") != (
        "dataset_stratified_expert_prediction_seed"
    ):
        raise ValueError("V4 inferential block must preserve dataset strata")
    thresholds = schema.get("effect_thresholds", {})
    if (
        float(thresholds.get("worst_brier_contract_regret_improvement", 0.0))
        <= 0
        or float(thresholds.get("robust_primary_improvement", 0.0)) <= 0
    ):
        raise ValueError("V4 robust effect thresholds must be positive")


def _derive_regret_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    v4 = any(
        str(row.get("metric", "")) == "brier_contract_regret"
        for row in rows
    )
    contract_metric = (
        "brier_contract_regret" if v4 else "contract_regret"
    )
    contract_rows = [
        row for row in rows if str(row["metric"]) == contract_metric
    ]
    grouped: dict[tuple[str, int, str, str, int], list[float]] = defaultdict(
        list
    )
    for row in contract_rows:
        value = float(row["value"])
        if not np.isfinite(value):
            continue
        grouped[
            (
                str(row["dataset"]),
                _expert_seed(row),
                str(row.get("fold", "")),
                str(row["method"]),
                int(row.get("router_training_seed", -1)),
            )
        ].append(value)
    derived: list[dict[str, Any]] = []
    for (dataset, seed, fold, method, router_seed), values in sorted(
        grouped.items()
    ):
        ordered = sorted(values, reverse=True)
        tail_count = max(1, int(np.ceil(0.2 * len(ordered))))
        prefix = "brier_contract_" if v4 else ""
        metrics = {
            f"mean_{prefix}regret": float(np.mean(values)),
            f"maximum_{prefix}regret": float(np.max(values)),
            f"cvar_{prefix}regret": float(np.mean(ordered[:tail_count])),
        }
        for metric, value in metrics.items():
            derived.append(
                {
                    "dataset": dataset,
                    (
                        "target_protocol_id"
                        if v4
                        else "target_contract"
                    ): "__seed_aggregate__",
                    "seed": seed,
                    "expert_prediction_seed": seed,
                    "router_training_seed": router_seed,
                    "fold": fold,
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "execution_status": MethodExecutionStatus.EXECUTABLE.value,
                }
            )
    return derived


def _matched_worst_contract_seed_deltas(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    *,
    metric: str,
    direction: str,
) -> dict[tuple[str, int], float]:
    if direction not in {"higher", "lower"}:
        raise ValueError("matched worst-case direction must be higher or lower")
    values: dict[tuple[str, str, int, str, str], float] = {}
    for row in rows:
        if (
            str(row["metric"]) != metric
            or str(row["method"]) not in {"full_corerouter", baseline}
        ):
            continue
        value = float(row["value"])
        if not np.isfinite(value):
            continue
        key = (
            str(row["dataset"]),
            _target_protocol(row),
            _expert_seed(row),
            str(row.get("fold", "")),
            str(row["method"]),
        )
        if key in values:
            raise ValueError(f"duplicate matched target-contract row {key}")
        values[key] = value
    contexts = {
        key[:4] for key in values if key[4] == "full_corerouter"
    }
    if not contexts:
        raise ValueError("matched worst-case comparison has no method rows")
    by_block: dict[tuple[str, int], list[float]] = defaultdict(list)
    for context in sorted(contexts):
        method_key = (*context, "full_corerouter")
        baseline_key = (*context, baseline)
        if baseline_key not in values:
            raise ValueError(
                f"missing paired target-contract outcome {baseline_key}"
            )
        method_value = values[method_key]
        baseline_value = values[baseline_key]
        improvement = (
            method_value - baseline_value
            if direction == "higher"
            else baseline_value - method_value
        )
        by_block[(context[0], context[2])].append(improvement)
    return {
        block: float(min(differences))
        for block, differences in sorted(by_block.items())
    }


def _worst_contract_seed_deltas(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
) -> tuple[float, ...]:
    """V2 compatibility view over the matched-contract implementation."""

    matched = _matched_worst_contract_seed_deltas(
        rows,
        baseline,
        metric="auprc",
        direction="higher",
    )
    return tuple(matched[key] for key in sorted(matched))


def _required_methods(schema: Mapping[str, Any]) -> set[str]:
    return {
        "full_corerouter",
        *(
            f"expert:{name}"
            for name in schema.get("required_experts", ())
        ),
        *_schema_strong_baselines(schema),
        *(
            f"ablation:{name}"
            for name in schema["required_ablations"]
        ),
    }


def _validate_complete_coverage(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    required_datasets = {str(value) for value in schema["required_datasets"]}
    required_protocols = set(_schema_target_protocols(schema))
    required_seeds = {
        int(value) for value in schema["required_expert_prediction_seeds"]
    }
    required_folds = {str(value) for value in schema["required_folds"]}
    required_methods = _required_methods(schema)
    metric_values = schema.get("required_contract_metrics")
    if metric_values is None:
        metric_values = schema["primary_outcomes"]
    required_metrics = {str(value) for value in metric_values}
    ranking_outcomes = {
        str(value) for value in schema.get("ranking_outcomes", ())
    }
    actual_datasets: set[str] = set()
    actual_protocols: set[str] = set()
    actual_seeds: set[int] = set()
    actual_folds: set[str] = set()
    actual_methods: set[str] = set()
    actual_keys: set[tuple[str, str, int, str, str, str]] = set()
    duplicates: list[tuple[str, str, int, str, str, str]] = []
    semantic_errors: list[str] = []
    for row in rows:
        metric = str(row["metric"])
        if metric not in required_metrics:
            continue
        dataset = str(row["dataset"])
        protocol = _target_protocol(row)
        seed = _expert_seed(row)
        fold = str(row.get("fold", ""))
        method = str(row["method"])
        key = (dataset, protocol, seed, fold, method, metric)
        if key in actual_keys:
            duplicates.append(key)
        actual_keys.add(key)
        actual_datasets.add(dataset)
        actual_protocols.add(protocol)
        actual_seeds.add(seed)
        actual_folds.add(fold)
        actual_methods.add(method)
        if int(row.get("seed", seed)) != seed:
            semantic_errors.append(f"seed alias mismatch for {key}")
        expected_router_seed = derive_router_seed(seed, method)
        if int(row.get("router_training_seed", -1)) != expected_router_seed:
            semantic_errors.append(f"router seed mismatch for {key}")
        try:
            status = MethodExecutionStatus(str(row["execution_status"]))
        except (KeyError, ValueError):
            semantic_errors.append(f"invalid execution status for {key}")
            continue
        value = float(row["value"])
        if (
            status
            in {
                MethodExecutionStatus.EXECUTABLE,
                MethodExecutionStatus.EXECUTABLE_WITH_FALLBACK,
            }
            and not np.isfinite(value)
        ):
            semantic_errors.append(f"executable result is non-finite for {key}")
        if (
            status
            in {
                MethodExecutionStatus.ABSTAIN_ONLY,
                MethodExecutionStatus.RESOURCE_BLOCKED,
                MethodExecutionStatus.NOT_APPLICABLE,
            }
            and metric in ranking_outcomes
            and np.isfinite(value)
        ):
            semantic_errors.append(
                f"blocked prediction entered ranking metric for {key}"
            )
    expected_keys = {
        (dataset, protocol, seed, fold, method, metric)
        for dataset in required_datasets
        for protocol in required_protocols
        for seed in required_seeds
        for fold in required_folds
        for method in required_methods
        for metric in required_metrics
    }
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    complete = not any(
        (
            duplicates,
            semantic_errors,
            missing,
            unexpected,
            actual_datasets != required_datasets,
            actual_protocols != required_protocols,
            actual_seeds != required_seeds,
            actual_folds != required_folds,
            actual_methods != required_methods,
        )
    )
    return {
        "complete": complete,
        "missing_cell_count": len(missing),
        "unexpected_cell_count": len(unexpected),
        "duplicate_cell_count": len(duplicates),
        "semantic_errors": semantic_errors,
        "datasets": sorted(actual_datasets),
        "target_protocols": sorted(actual_protocols),
        "target_contracts": sorted(actual_protocols),
        "expert_prediction_seeds": sorted(actual_seeds),
        "folds": sorted(actual_folds),
        "methods": sorted(actual_methods),
    }


def validate_no_training_completeness(
    materialized: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact planned key surface without accepting measurements."""

    errors: list[str] = []
    if materialized.get("status") != "VALIDATED_NO_TRAINING":
        errors.append("materialization status is not VALIDATED_NO_TRAINING")
    for flag in (
        "training_performed",
        "metric_computation_performed",
        "target_oracle_measurement_performed",
    ):
        if materialized.get(flag) is not False:
            errors.append(f"{flag} must be false")
    rows = list(materialized.get("planned_rows", ()))
    required_fields = set(PILOT_RESULT_ROW_FIELDS)
    identities: dict[
        tuple[str, str, int, str],
        set[tuple[str, str]],
    ] = defaultdict(set)
    normalized_rows: list[dict[str, Any]] = []
    for index, row_value in enumerate(rows):
        if not isinstance(row_value, Mapping):
            errors.append(f"planned row {index} is not a mapping")
            continue
        row = dict(row_value)
        if set(row) != required_fields:
            errors.append(f"planned row {index} has result-schema drift")
            continue
        if row.get("value") is not None:
            errors.append(f"planned row {index} contains a measured value")
        if row.get("measurement_status") != "NOT_EXECUTED_NO_TRAINING":
            errors.append(f"planned row {index} has an invalid measurement status")
        protocol_id = str(row.get("target_protocol_id", ""))
        coordinate_hash = str(
            row.get("target_contract_coordinate_hash", "")
        )
        contract_id = str(row.get("target_contract_id", ""))
        if (
            not protocol_id
            or len(coordinate_hash) != 64
            or ":" not in contract_id
        ):
            errors.append(f"planned row {index} has invalid target identities")
        key = (
            str(row.get("dataset", "")),
            protocol_id,
            _expert_seed(row),
            str(row.get("fold", "")),
        )
        identities[key].add((coordinate_hash, contract_id))
        normalized = dict(row)
        normalized["value"] = float("nan")
        normalized_rows.append(normalized)
    for key, values in identities.items():
        if len(values) != 1:
            errors.append(
                f"target protocol binding {key} resolves to {len(values)} identities"
            )
    bindings = list(materialized.get("protocol_bindings", ()))
    binding_keys = {
        (
            str(binding["dataset"]),
            str(binding["target_protocol_id"]),
            int(binding["expert_prediction_seed"]),
            str(binding["fold"]),
        ): (
            str(binding["target_contract_coordinate_hash"]),
            str(binding["target_contract_id"]),
        )
        for binding in bindings
        if isinstance(binding, Mapping)
    }
    for key, values in identities.items():
        if len(values) == 1 and binding_keys.get(key) != next(iter(values)):
            errors.append(f"planned identity does not match registry binding for {key}")
    coverage = _validate_complete_coverage(normalized_rows, schema)
    complete = coverage["complete"] and not errors
    return {
        "status": (
            "NO_TRAINING_COMPLETENESS_VALIDATED"
            if complete
            else "BLOCKED_NO_TRAINING_COMPLETENESS"
        ),
        "complete": complete,
        "training_performed": False,
        "metric_computation_performed": False,
        "target_oracle_measurement_performed": False,
        "errors": errors,
        "coverage": coverage,
        "identity_binding_count": len(identities),
    }


def _paired_improvements(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str,
    direction: str,
    dataset: str | None = None,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    values: dict[tuple[str, str, int, str, str], float] = {}
    for row in rows:
        if (
            str(row["metric"]) != metric
            or str(row["method"]) not in {method, baseline}
            or (dataset is not None and str(row["dataset"]) != dataset)
        ):
            continue
        value = float(row["value"])
        if not np.isfinite(value):
            continue
        key = (
            str(row["dataset"]),
            str(row["target_contract"]),
            _expert_seed(row),
            str(row.get("fold", "")),
            str(row["method"]),
        )
        if key in values:
            raise ValueError(f"duplicate paired effect row {key}")
        values[key] = value
    contexts = {
        key[:4] for key in values if key[4] == method
    }
    if not contexts:
        raise ValueError(f"paired effect has no rows for {method}/{metric}")
    missing = [
        context
        for context in sorted(contexts)
        if (*context, baseline) not in values
    ]
    if missing:
        raise ValueError(f"paired effect has missing baseline contexts: {missing}")
    by_seed: dict[int, list[float]] = defaultdict(list)
    for context in sorted(contexts):
        method_value = values[(*context, method)]
        baseline_value = values[(*context, baseline)]
        improvement = (
            method_value - baseline_value
            if direction == "higher"
            else baseline_value - method_value
        )
        by_seed[context[2]].append(improvement)
    context_counts = {len(value) for value in by_seed.values()}
    if len(context_counts) != 1:
        raise ValueError("paired seeds have unequal target-context coverage")
    seeds = tuple(sorted(by_seed))
    return (
        np.asarray(
            [float(np.mean(by_seed[seed])) for seed in seeds],
            dtype=float,
        ),
        seeds,
        tuple(len(by_seed[seed]) for seed in seeds),
    )


def _effect_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str,
    direction: str,
    minimum_effect: float,
) -> dict[str, Any]:
    improvement, seeds, context_counts = _paired_improvements(
        rows,
        method=method,
        baseline=baseline,
        metric=metric,
        direction=direction,
    )
    zeros = np.zeros_like(improvement)
    wilcoxon = exact_wilcoxon(improvement, zeros, alternative="greater")
    permutation = paired_permutation(
        improvement,
        zeros,
        alternative="greater",
    )
    low, high = bootstrap_seed_blocks(improvement)
    return {
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "direction": direction,
        "seeds": list(seeds),
        "contexts_per_seed": list(context_counts),
        "mean_improvement": float(improvement.mean()),
        "minimum_effect": float(minimum_effect),
        "exact_wilcoxon_p": wilcoxon.p_value,
        "raw_p": permutation.p_value,
        "paired_permutation_p": permutation.p_value,
        "seed_block_bootstrap_95": [low, high],
    }


def _apply_holm_to_effect_records(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    if not records:
        return []
    correction = holm(
        [float(record["raw_p"]) for record in records],
        alpha=alpha,
    )
    output: list[dict[str, Any]] = []
    for record, adjusted, reject in zip(
        records,
        correction.adjusted,
        correction.reject,
        strict=True,
    ):
        updated = dict(record)
        updated["holm_adjusted_p"] = adjusted
        updated["holm_reject"] = reject
        interval = updated.get("seed_block_bootstrap_95")
        ci_support = (
            True
            if interval is None
            else float(interval[0]) >= 0.0
        )
        updated["meaningful"] = bool(
            reject
            and float(updated["mean_improvement"])
            >= float(updated["minimum_effect"])
            and ci_support
        )
        output.append(updated)
    return output


def _evaluate_ablation_effects(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for name, test in schema["ablation_tests"].items():
        record = _effect_record(
            rows,
            method="full_corerouter",
            baseline=f"ablation:{name}",
            metric=str(test["metric"]),
            direction=str(test["direction"]),
            minimum_effect=float(test["minimum_effect"]),
        )
        record["ablation"] = name
        record["required_contribution"] = bool(
            test["required_contribution"]
        )
        records.append(record)
    return _apply_holm_to_effect_records(
        records,
        alpha=float(schema.get("alpha", 0.05)),
    )


def _baseline_comparisons(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    positions: dict[str, list[int]] = defaultdict(list)
    p_values: dict[str, list[float]] = defaultdict(list)
    thresholds = schema.get("minimum_effects", {})
    for metric in schema["primary_outcomes"]:
        direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
        for baseline in _schema_strong_baselines(schema):
            record = _effect_record(
                rows,
                method="full_corerouter",
                baseline=str(baseline),
                metric=str(metric),
                direction=direction,
                minimum_effect=float(thresholds.get(metric, 0.0)),
            )
            records.append(record)
            family = _holm_family(str(metric))
            positions[family].append(len(records) - 1)
            p_values[family].append(float(record["raw_p"]))
    for family, values in p_values.items():
        correction = holm(values, alpha=float(schema.get("alpha", 0.05)))
        for position, adjusted, reject in zip(
            positions[family],
            correction.adjusted,
            correction.reject,
            strict=True,
        ):
            records[position]["holm_adjusted_p"] = adjusted
            records[position]["holm_reject"] = reject
    return records


def _paired_improvements_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str,
    direction: str,
    dataset: str,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    if direction not in {"higher", "lower"}:
        raise ValueError("paired effect direction must be higher or lower")
    values: dict[tuple[str, int, str, str], float] = {}
    for row in rows:
        if (
            str(row.get("dataset")) != dataset
            or str(row.get("metric")) != metric
            or str(row.get("method")) not in {method, baseline}
        ):
            continue
        value = float(row["value"])
        if not np.isfinite(value):
            continue
        key = (
            _target_protocol(row),
            _expert_seed(row),
            str(row.get("fold", "")),
            str(row["method"]),
        )
        if key in values:
            raise ValueError(f"duplicate dataset-stratified paired row {key}")
        values[key] = value
    contexts = {key[:3] for key in values if key[3] == method}
    if not contexts:
        raise ValueError(
            f"paired effect has no rows for {dataset}/{method}/{metric}"
        )
    missing = [
        context
        for context in sorted(contexts)
        if (*context, baseline) not in values
    ]
    if missing:
        raise ValueError(
            f"paired effect has missing baseline contexts for {dataset}: {missing}"
        )
    by_seed: dict[int, list[float]] = defaultdict(list)
    for context in sorted(contexts):
        method_value = values[(*context, method)]
        baseline_value = values[(*context, baseline)]
        improvement = (
            method_value - baseline_value
            if direction == "higher"
            else baseline_value - method_value
        )
        by_seed[context[1]].append(improvement)
    context_counts = {len(value) for value in by_seed.values()}
    if len(context_counts) != 1:
        raise ValueError(
            f"dataset {dataset} seed blocks have unequal target-context coverage"
        )
    seeds = tuple(sorted(by_seed))
    return (
        np.asarray(
            [float(np.mean(by_seed[seed])) for seed in seeds],
            dtype=float,
        ),
        seeds,
        tuple(len(by_seed[seed]) for seed in seeds),
    )


def _dataset_effect_record_v4(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str,
    direction: str,
    minimum_effect: float,
    dataset: str,
) -> tuple[dict[str, Any], np.ndarray]:
    improvement, seeds, context_counts = _paired_improvements_v4(
        rows,
        method=method,
        baseline=baseline,
        metric=metric,
        direction=direction,
        dataset=dataset,
    )
    zeros = np.zeros_like(improvement)
    wilcoxon = exact_wilcoxon(improvement, zeros, alternative="greater")
    permutation = paired_permutation(
        improvement,
        zeros,
        alternative="greater",
    )
    low, high = bootstrap_seed_blocks(improvement)
    return (
        {
            "dataset": dataset,
            "method": method,
            "baseline": baseline,
            "metric": metric,
            "direction": direction,
            "seeds": list(seeds),
            "contexts_per_seed": list(context_counts),
            "mean_improvement": float(improvement.mean()),
            "minimum_effect": float(minimum_effect),
            "exact_wilcoxon_p": wilcoxon.p_value,
            "raw_p": permutation.p_value,
            "paired_permutation_p": permutation.p_value,
            "dataset_seed_block_bootstrap_95": [low, high],
        },
        improvement,
    )


def _holm_dataset_records_v4(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    if not records:
        return []
    correction = holm(
        [float(record["raw_p"]) for record in records],
        alpha=alpha,
    )
    output: list[dict[str, Any]] = []
    for record, adjusted, reject in zip(
        records,
        correction.adjusted,
        correction.reject,
        strict=True,
    ):
        updated = dict(record)
        updated["holm_adjusted_p"] = adjusted
        updated["holm_reject"] = reject
        interval = updated["dataset_seed_block_bootstrap_95"]
        updated["meaningful"] = bool(
            reject
            and float(updated["mean_improvement"])
            >= float(updated["minimum_effect"])
            and float(interval[0]) >= 0.0
        )
        output.append(updated)
    return output


def _v4_baseline_comparisons(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    differences: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ] = defaultdict(dict)
    thresholds = schema.get("minimum_effects", {})
    for metric_value in schema["primary_outcomes"]:
        metric = str(metric_value)
        direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
        family = _holm_family_v4(metric)
        for baseline in _schema_strong_baselines(schema):
            for dataset_value in schema["required_datasets"]:
                dataset = str(dataset_value)
                record, improvement = _dataset_effect_record_v4(
                    rows,
                    method="full_corerouter",
                    baseline=baseline,
                    metric=metric,
                    direction=direction,
                    minimum_effect=float(thresholds.get(metric, 0.0)),
                    dataset=dataset,
                )
                raw_by_family[family].append(record)
                differences[(baseline, metric)][dataset] = improvement
    corrected: list[dict[str, Any]] = []
    for family in sorted(raw_by_family):
        family_records = _holm_dataset_records_v4(
            raw_by_family[family],
            alpha=float(schema.get("alpha", 0.05)),
        )
        for record in family_records:
            record["holm_family"] = family
        corrected.extend(family_records)
    combined = []
    for (baseline, metric), by_dataset in sorted(differences.items()):
        low, high = hierarchical_dataset_bootstrap(by_dataset)
        combined.append(
            {
                "baseline": baseline,
                "metric": metric,
                "datasets": sorted(by_dataset),
                "hierarchical_dataset_seed_bootstrap_95": [low, high],
                "mean_of_dataset_means": float(
                    np.mean(
                        [
                            values.mean()
                            for values in by_dataset.values()
                        ]
                    )
                ),
                "secondary_evidence_only": True,
            }
        )
    return corrected, combined


def _v4_ablation_effects(
    rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for name, test in schema["ablation_tests"].items():
        for dataset_value in schema["required_datasets"]:
            dataset = str(dataset_value)
            record, _ = _dataset_effect_record_v4(
                rows,
                method="full_corerouter",
                baseline=f"ablation:{name}",
                metric=str(test["metric"]),
                direction=str(test["direction"]),
                minimum_effect=float(test["minimum_effect"]),
                dataset=dataset,
            )
            record["ablation"] = name
            record["required_contribution"] = bool(
                test["required_contribution"]
            )
            raw.append(record)
    return _holm_dataset_records_v4(
        raw,
        alpha=float(schema.get("alpha", 0.05)),
    )


def _evaluate_pilot_gate_v4(
    result: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(result.get("rows", ()))
    coverage = _validate_complete_coverage(rows, schema)
    if not coverage["complete"]:
        return {
            "status": "BLOCKED_INCOMPLETE_PILOT_COVERAGE",
            "passed": False,
            "coverage": coverage,
        }
    analysis_rows = [*rows, *_derive_regret_rows(rows)]
    try:
        comparisons, combined = _v4_baseline_comparisons(
            analysis_rows,
            schema,
        )
        ablations = _v4_ablation_effects(analysis_rows, schema)
    except ValueError as error:
        return {
            "status": "BLOCKED_INCOMPLETE_COMPARABLE_EVIDENCE",
            "passed": False,
            "coverage": coverage,
            "reason": str(error),
        }
    headline_baselines = {
        "average_all_feasible",
        "best_source_validation",
    }
    robust_threshold = float(
        schema.get("effect_thresholds", {}).get(
            "robust_primary_improvement",
            0.0,
        )
    )
    robust_candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for record in comparisons:
        if (
            record["baseline"] in headline_baselines
            and record["metric"] in V4_ROBUST_OUTCOMES
        ):
            robust_candidates[
                (str(record["baseline"]), str(record["metric"]))
            ].append(record)
    robust_support = [
        {
            "baseline": baseline,
            "metric": metric,
            "datasets": [str(record["dataset"]) for record in records],
            "corrected_support_on_at_least_one_dataset": any(
                bool(record["holm_reject"])
                and float(record["mean_improvement"]) >= robust_threshold
                and float(record["dataset_seed_block_bootstrap_95"][0]) >= 0
                for record in records
            ),
            "no_contradictory_dataset_effect": all(
                float(record["mean_improvement"]) > 0
                for record in records
            ),
        }
        for (baseline, metric), records in sorted(robust_candidates.items())
    ]
    robust_rule = any(
        record["corrected_support_on_at_least_one_dataset"]
        and record["no_contradictory_dataset_effect"]
        for record in robust_support
    )
    cvar_records = [
        record
        for record in comparisons
        if record["metric"] == "cvar_brier_contract_regret"
        and record["baseline"] in headline_baselines
    ]
    positive_both_datasets = bool(cvar_records) and all(
        float(record["mean_improvement"]) > robust_threshold
        for record in cvar_records
    )
    auprc_floor = float(
        schema.get("effect_thresholds", {}).get(
            "average_auprc_harm_floor",
            -0.002,
        )
    )
    auprc_records = [
        record
        for record in comparisons
        if record["metric"] == "auprc"
        and record["baseline"] in headline_baselines
    ]
    required_ablations = set(
        schema.get("required_meaningful_ablations", ())
    )
    meaningful_ablations = all(
        any(
            record["ablation"] == name and bool(record["meaningful"])
            for record in ablations
        )
        and all(
            float(record["mean_improvement"]) >= 0
            for record in ablations
            if record["ablation"] == name
        )
        for name in required_ablations
    )
    routing = list(result.get("routing", ()))
    expected_routing = (
        len(schema["required_datasets"])
        * len(_schema_target_protocols(schema))
        * len(schema["required_expert_prediction_seeds"])
        * len(schema["required_folds"])
    )
    full_coverage_rows = [
        row
        for row in rows
        if row["method"] == "full_corerouter"
        and row["metric"] == "coverage"
    ]
    criteria = {
        "complete_two_dataset_ten_seed_protocol_coverage": coverage["complete"],
        "dataset_stratified_corrected_robust_support": robust_rule,
        "positive_effects_on_both_datasets": positive_both_datasets,
        "no_material_dataset_level_auprc_harm": all(
            float(record["mean_improvement"]) >= auprc_floor
            and float(record["dataset_seed_block_bootstrap_95"][0])
            >= auprc_floor
            for record in auprc_records
        ),
        "meaningful_required_ablations": meaningful_ablations,
        "zero_coverage_full_method_blocked": bool(
            full_coverage_rows
            and all(float(row["value"]) > 0 for row in full_coverage_rows)
        ),
        "routing_diversity": bool(
            len(routing) == expected_routing
            and min(int(record["distinct_experts"]) for record in routing) >= 2
        ),
        "routing_stability": bool(
            len(routing) == expected_routing
            and max(
                float(record["perturbation_flip_rate"])
                for record in routing
            )
            <= float(
                schema.get("effect_thresholds", {}).get(
                    "maximum_routing_flip_rate",
                    0.1,
                )
            )
        ),
        "no_target_label_selection": (
            result.get("target_label_selection") is False
            and result.get("oracle_target_selection") is False
        ),
        "contract_feasible_headline_oracle": (
            result.get("headline_oracle") == "contract_feasible_oracle"
            and result.get("diagnostic_oracle")
            == "instance_clairvoyant_oracle_ceiling"
        ),
        "compatibility_component_excluded_from_headline": (
            not (
                set(_schema_strong_baselines(schema))
                & set(schema.get("compatibility_components", ()))
            )
            and "graphsafe_confidence_abstention_component"
            not in _schema_strong_baselines(schema)
        ),
    }
    return {
        "status": "GATE_EVALUATED_V4",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "coverage": coverage,
        "dataset_stratified_comparisons": comparisons,
        "dataset_stratified_ablation_effects": ablations,
        "hierarchical_bootstrap_secondary_evidence": combined,
        "robust_supporting_comparisons": robust_support,
        "pairing_unit": (
            "exact target protocols within dataset and expert-prediction seed"
        ),
        "inferential_block": "dataset_stratified_expert_prediction_seed",
    }


def evaluate_pilot_gate(
    result: Mapping[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    _validate_gate_schema(schema)
    if schema.get("schema_version") == "coregraph_pilot_gate_v4":
        return _evaluate_pilot_gate_v4(result, schema)
    rows = list(result.get("rows", []))
    coverage = _validate_complete_coverage(rows, schema)
    if not coverage["complete"]:
        return {
            "status": "BLOCKED_INCOMPLETE_PILOT_COVERAGE",
            "passed": False,
            "coverage": coverage,
        }
    analysis_rows = [*rows, *_derive_regret_rows(rows)]
    try:
        comparisons = _baseline_comparisons(analysis_rows, schema)
        ablations = _evaluate_ablation_effects(analysis_rows, schema)
    except ValueError as error:
        return {
            "status": "BLOCKED_INCOMPLETE_COMPARABLE_EVIDENCE",
            "passed": False,
            "coverage": coverage,
            "reason": str(error),
        }
    headline_baselines = (
        "average_all_feasible",
        "best_source_validation",
    )
    worst_regret = {
        baseline: _matched_worst_contract_seed_deltas(
            rows,
            baseline,
            metric="contract_regret",
            direction="lower",
        )
        for baseline in headline_baselines
    }
    worst_threshold = float(
        schema.get("effect_thresholds", {}).get(
            "worst_contract_regret_improvement",
            0.0,
        )
    )
    robust_threshold = float(
        schema.get("effect_thresholds", {}).get(
            "robust_primary_improvement",
            0.0,
        )
    )
    robust_support = [
        record
        for record in comparisons
        if record["metric"] in ROBUST_OUTCOMES
        and record["baseline"] in headline_baselines
        and record["holm_reject"]
        and record["mean_improvement"] >= robust_threshold
        and record["seed_block_bootstrap_95"][0] > 0
    ]
    required_datasets = [str(value) for value in schema["required_datasets"]]
    positive_by_dataset = True
    dataset_effects: dict[str, dict[str, float]] = {}
    for dataset in required_datasets:
        dataset_effects[dataset] = {}
        for baseline in headline_baselines:
            improvement, _, _ = _paired_improvements(
                analysis_rows,
                method="full_corerouter",
                baseline=baseline,
                metric="cvar_regret",
                direction="lower",
                dataset=dataset,
            )
            value = float(improvement.mean())
            dataset_effects[dataset][baseline] = value
            positive_by_dataset &= value > robust_threshold
    auprc_harm_floor = float(
        schema.get("effect_thresholds", {}).get(
            "average_auprc_harm_floor",
            -0.002,
        )
    )
    auprc_guardrails = [
        record
        for record in comparisons
        if record["metric"] == "auprc"
        and record["baseline"] in headline_baselines
    ]
    required_ablation_names = set(
        schema.get("required_meaningful_ablations", ())
    )
    meaningful_by_name = {
        str(record["ablation"]): bool(record["meaningful"])
        for record in ablations
    }
    routing = list(result.get("routing", []))
    expected_routing = (
        len(schema["required_datasets"])
        * len(schema["required_target_contracts"])
        * len(schema["required_expert_prediction_seeds"])
        * len(schema["required_folds"])
    )
    full_coverage_rows = [
        row
        for row in rows
        if row["method"] == "full_corerouter"
        and row["metric"] == "coverage"
    ]
    criteria = {
        "complete_two_dataset_ten_seed_coverage": coverage["complete"],
        "positive_worst_contract_regret_vs_average": all(
            value > worst_threshold
            for value in worst_regret["average_all_feasible"].values()
        ),
        "positive_worst_contract_regret_vs_source_best": all(
            value > worst_threshold
            for value in worst_regret["best_source_validation"].values()
        ),
        "corrected_robust_primary_support": bool(robust_support),
        "positive_effects_on_both_datasets": positive_by_dataset,
        "no_material_average_auprc_harm": all(
            record["mean_improvement"] >= auprc_harm_floor
            and record["seed_block_bootstrap_95"][0] >= auprc_harm_floor
            for record in auprc_guardrails
        ),
        "no_one_contract_only_effect": all(
            value > worst_threshold
            for baseline_values in worst_regret.values()
            for value in baseline_values.values()
        ),
        "all_ablation_effects_evaluated": len(ablations)
        == len(REQUIRED_ABLATIONS),
        "meaningful_required_ablations": all(
            meaningful_by_name.get(name, False)
            for name in required_ablation_names
        ),
        "zero_coverage_full_method_blocked": bool(
            full_coverage_rows
            and all(float(row["value"]) > 0 for row in full_coverage_rows)
        ),
        "routing_diversity": bool(
            len(routing) == expected_routing
            and min(int(record["distinct_experts"]) for record in routing) >= 2
        ),
        "routing_stability": bool(
            len(routing) == expected_routing
            and max(
                float(record["perturbation_flip_rate"])
                for record in routing
            )
            <= float(
                schema.get("effect_thresholds", {}).get(
                    "maximum_routing_flip_rate",
                    0.1,
                )
            )
        ),
        "no_target_label_selection": (
            result.get("target_label_selection") is False
            and result.get("oracle_target_selection") is False
        ),
        "contract_feasible_headline_oracle": (
            result.get("headline_oracle") == "contract_feasible_oracle"
            and result.get("diagnostic_oracle")
            == "instance_clairvoyant_oracle_ceiling"
        ),
        "instance_oracle_excluded_from_methods": not any(
            row["method"] == "instance_clairvoyant_oracle_ceiling"
            for row in rows
        ),
    }
    return {
        "status": "GATE_EVALUATED",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "coverage": coverage,
        "comparisons": comparisons,
        "ablation_effects": ablations,
        "matched_worst_contract_regret_improvements": {
            baseline: {
                f"{dataset}:seed{seed}": value
                for (dataset, seed), value in values.items()
            }
            for baseline, values in worst_regret.items()
        },
        "robust_supporting_comparisons": robust_support,
        "dataset_effects": dataset_effects,
        "pairing_unit": (
            "matched target contracts within expert-prediction seed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-result",
        default="results/coregraph_pilot/saved_output_pilot.json",
    )
    parser.add_argument(
        "--schema",
        default="results/coregraph_build/PILOT_GATE_FROZEN_SPEC.json",
    )
    parser.add_argument(
        "--output",
        default="results/coregraph_pilot/gate_report.json",
    )
    args = parser.parse_args()
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    result_path = Path(args.pilot_result)
    if not result_path.exists():
        report = {
            "status": "BLOCKED_MISSING_PILOT_RESULT",
            "passed": False,
            "required_datasets": schema["required_datasets"],
            "required_expert_prediction_seeds": (
                schema["required_expert_prediction_seeds"]
            ),
        }
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = evaluate_pilot_gate(result, schema)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"].startswith("BLOCKED"):
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
