#!/usr/bin/env python3
"""Plan/validate V5 readiness or operate the separately gated legacy V4 pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coregraph.experiments.pilot import (  # noqa: E402
    BaselinePrediction,
    MethodExecutionStatus,
    PILOT_RESULT_ROW_FIELDS,
    PilotAblation,
    SavedSourceGroup,
    align_artifact_group,
    baseline_scores,
    contract_feasible_oracle,
    derive_router_seed,
    discover_prediction_manifests,
    evaluate_saved_output_pilot,
    fit_saved_output_corerouter,
    instance_clairvoyant_oracle_ceiling,
    load_prediction_artifacts,
    validate_artifact_groups,
)
from coregraph.experiments.scenario_manifests import (  # noqa: E402
    load_base_prediction_artifacts,
    load_evaluation_scenarios,
    validate_no_training_scenarios,
)
from coregraph.data.leakage import (  # noqa: E402
    audit_cross_role_prediction_scopes,
)
from coregraph.experiments.protocol_registry import (  # noqa: E402
    load_protocol_registry,
    validate_protocol_bindings,
)


def _availability_and_cost(group, rows: int):
    availability = {
        artifact.canonical_expert_id: np.full(
            rows,
            artifact.expert_available,
            dtype=bool,
        )
        for artifact in group
    }
    costs = {
        artifact.canonical_expert_id: artifact.compute_cost
        for artifact in group
    }
    return availability, costs


def _router_execution_status(
    prediction,
    target_availability,
) -> MethodExecutionStatus:
    if prediction.forced_abstention.any():
        return MethodExecutionStatus.RESOURCE_BLOCKED
    if prediction.abstain.all():
        return MethodExecutionStatus.ABSTAIN_ONLY
    if any(
        not np.asarray(target_availability[expert]).all()
        for expert in target_availability
    ):
        return MethodExecutionStatus.EXECUTABLE_WITH_FALLBACK
    return MethodExecutionStatus.EXECUTABLE


def _group_roles(groups):
    roles = {
        key: {artifact.contract_role for artifact in group}
        for key, group in groups.items()
    }
    if any(len(value) != 1 for value in roles.values()):
        raise RuntimeError(
            "each prediction group must declare exactly one contract role"
        )
    return roles


def build_no_training_materialization(
    artifacts,
    groups,
    *,
    gate_schema,
    registry,
):
    """Materialise the full V4 result-key surface without fitting or scoring."""

    bindings = validate_protocol_bindings(artifacts, registry)
    roles = _group_roles(groups)
    source_keys = sorted(
        key for key, value in roles.items() if value == {"source"}
    )
    target_keys = sorted(
        key for key, value in roles.items() if value == {"target"}
    )
    if not target_keys:
        raise RuntimeError("pilot validation requires declared target groups")
    aligned = {
        key: align_artifact_group(groups[key])
        for key in [*source_keys, *target_keys]
    }
    scopes = tuple(
        aligned[key].as_leakage_scope(artifact)
        for key in [*source_keys, *target_keys]
        for artifact in groups[key]
    )
    leakage_reports = audit_cross_role_prediction_scopes(scopes)
    atomic = [
        report
        for report in leakage_reports
        if not report.passed
    ]
    if atomic:
        codes = sorted(
            {
                finding.code
                for report in atomic
                for finding in report.findings
                if finding.severity == "ATOMIC"
            }
        )
        raise RuntimeError(f"atomic cross-role leakage blocks validation: {codes}")
    required_methods = {
        "full_corerouter",
        *(
            f"expert:{name}"
            for name in gate_schema.get("required_experts", ())
        ),
        *(str(name) for name in gate_schema.get("strong_baselines", ())),
        *(
            f"ablation:{name}"
            for name in gate_schema.get("required_ablations", ())
        ),
    }
    metrics = {
        str(value)
        for value in gate_schema.get("required_contract_metrics", ())
    }
    planned_rows: list[dict[str, object]] = []
    planned_routing: list[dict[str, object]] = []
    source_groups: dict[str, list[object]] = {}
    for target_key in target_keys:
        dataset, task, _, _, expert_seed, fold = target_key
        compatible_sources = [
            key
            for key in source_keys
            if key[0] == dataset
            and key[1] == task
            and key[4] == expert_seed
            and key[5] == fold
        ]
        if len(compatible_sources) < 2:
            raise RuntimeError(
                f"target {target_key} requires two same-seed source contracts"
            )
        target_artifact = groups[target_key][0]
        for method in sorted(required_methods):
            for metric in sorted(metrics):
                row = {
                    "dataset": dataset,
                    "target_protocol_id": target_artifact.protocol_id,
                    "target_contract_coordinate_hash": (
                        target_artifact.contract_coordinate_hash
                    ),
                    "target_contract_id": target_artifact.contract_id,
                    "seed": int(expert_seed),
                    "expert_prediction_seed": int(expert_seed),
                    "router_training_seed": derive_router_seed(
                        int(expert_seed),
                        method,
                    ),
                    "fold": fold,
                    "method": method,
                    "metric": metric,
                    "value": None,
                    "measurement_status": "NOT_EXECUTED_NO_TRAINING",
                    "risk_definition": {
                        "brier_contract_regret": (
                            "contract_mean_brier_risk_minus_"
                            "contract_feasible_oracle_brier_risk"
                        ),
                        "selective_zero_one_risk": (
                            "zero_one_error_on_non_abstained_rows"
                        ),
                    }.get(metric, "not_a_risk_quantity"),
                    "execution_status": (
                        MethodExecutionStatus.NOT_APPLICABLE.value
                    ),
                    "abstention_threshold": None,
                    "abstention_threshold_provenance": "not_fitted",
                    "routing_threshold": None,
                    "routing_threshold_provenance": "not_fitted",
                    "abstention_decision_sha256": None,
                    "accepted_count": None,
                    "abstained_count": None,
                    "forced_abstention": None,
                    "forced_abstention_count": None,
                    "abstention_capacity": None,
                    "abstention_cost_per_decision": None,
                    "offline_oracle": False,
                }
                if set(row) != set(PILOT_RESULT_ROW_FIELDS):
                    raise RuntimeError("no-training row schema drift")
                planned_rows.append(row)
        planned_routing.append(
            {
                "dataset": dataset,
                "target_protocol_id": target_artifact.protocol_id,
                "target_contract_coordinate_hash": (
                    target_artifact.contract_coordinate_hash
                ),
                "target_contract_id": target_artifact.contract_id,
                "expert_prediction_seed": int(expert_seed),
                "fold": fold,
                "measurement_status": "NOT_EXECUTED_NO_TRAINING",
            }
        )
        source_groups[str(target_key)] = [
            list(key) for key in compatible_sources
        ]
    return {
        "schema": "coregraph_saved_output_pilot_validation_v4",
        "status": "VALIDATED_NO_TRAINING",
        "training_performed": False,
        "metric_computation_performed": False,
        "target_oracle_measurement_performed": False,
        "target_label_selection": False,
        "oracle_target_selection": False,
        "protocol_bindings": bindings,
        "row_scope_reports": [
            {
                **dict(aligned[key].audit),
                "expert_id": artifact.canonical_expert_id,
                "artifact_path": str(artifact.path),
                "artifact_checksum": artifact.checksum,
            }
            for key in [*source_keys, *target_keys]
            for artifact in groups[key]
        ],
        "leakage_reports": [
            report.to_dict() for report in leakage_reports
        ],
        "planned_rows": planned_rows,
        "planned_routing": planned_routing,
        "source_groups": source_groups,
        "target_groups": [list(key) for key in target_keys],
        "result_row_fields": list(PILOT_RESULT_ROW_FIELDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/coregraph/pilot/saved_output.yaml",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--protocol-registry",
        default=(
            "results/coregraph_build/"
            "CONTRACT_PROTOCOL_REGISTRY_V4.json"
        ),
    )
    parser.add_argument(
        "--gate-schema",
        default="results/coregraph_build/PILOT_GATE_FROZEN_SPEC_V4.json",
    )
    args = parser.parse_args()
    if args.execute and args.validate_only:
        parser.error("--execute and --validate-only are mutually exclusive")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("manifest_schema_version") == "v5":
        if args.execute:
            raise RuntimeError(
                "V5 manifests are readiness-only in this review; pilot execution "
                "requires separate independent authorization"
            )
        base_roots = [
            Path(os.path.expandvars(value)).expanduser().resolve()
            for value in config.get("base_prediction_manifest_roots", ())
        ]
        base_manifests: list[Path] = []
        for root in base_roots:
            if root.is_file():
                base_manifests.append(root)
            elif root.is_dir():
                base_manifests.extend(
                    root.rglob("*base_prediction_manifest_v5.json")
                )
        base_manifests = sorted(set(base_manifests))
        scenario_index_path = Path(
            os.path.expandvars(config["scenario_binding_index"])
        ).expanduser().resolve()
        v5_plan = {
            "schema": "coregraph_saved_output_pilot_plan_v5",
            "execute_requested": False,
            "validate_only_requested": args.validate_only,
            "base_manifest_roots": [str(value) for value in base_roots],
            "discovered_base_manifests": [
                str(path) for path in base_manifests
            ],
            "scenario_binding_index": str(scenario_index_path),
            "status": "PLANNED_READINESS_ONLY",
            "training_performed": False,
            "fitting_path_reachable": False,
            "metric_computation_performed": False,
            "oracle_computation_performed": False,
        }
        output = Path(config["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if not args.validate_only:
            output.write_text(
                json.dumps(v5_plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(v5_plan, indent=2, sort_keys=True))
            return 0
        artifacts_v5 = load_base_prediction_artifacts(base_manifests)
        scenarios_v5 = load_evaluation_scenarios(scenario_index_path)
        registry_v5 = load_protocol_registry(args.protocol_registry)
        validation_v5 = validate_no_training_scenarios(
            artifacts_v5,
            scenarios_v5,
            registry=registry_v5,
            expected_datasets=tuple(config["required_datasets"]),
            expected_protocols=tuple(config["required_target_protocols"]),
            expected_experts=tuple(config["required_experts"]),
            expected_seeds=tuple(
                int(seed) for seed in config["required_seeds"]
            ),
            expected_folds=tuple(config.get("required_folds", ("fold0",))),
        )
        validation_output = Path(
            config.get(
                "validation_output",
                "results/coregraph_pilot/saved_output_validation_v5.json",
            )
        )
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(
            json.dumps(validation_v5, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(validation_v5, indent=2, sort_keys=True))
        return 0
    roots = [
        os.path.expandvars(value)
        for value in config.get("prediction_manifest_roots", [])
    ]
    manifests = discover_prediction_manifests(roots)
    plan = {
        "schema": "coregraph_saved_output_pilot_plan_v4",
        "execute_requested": args.execute,
        "manifest_roots": roots,
        "discovered_manifests": [str(path) for path in manifests],
        "status": "PLANNED",
        "forbidden_target_selection": True,
        "required_datasets": config["required_datasets"],
        "required_target_protocols": config["required_target_protocols"],
        "required_expert_prediction_seeds": config["required_seeds"],
        "required_ablations": [
            ablation.value
            for ablation in PilotAblation
            if ablation is not PilotAblation.FULL
        ],
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute and not args.validate_only:
        output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    artifacts = load_prediction_artifacts(manifests)
    groups = validate_artifact_groups(
        artifacts,
        expected_experts=tuple(config["required_experts"]),
        expected_seeds=tuple(int(seed) for seed in config["required_seeds"]),
        expected_datasets=tuple(config["required_datasets"]),
        expected_target_protocols=tuple(
            config["required_target_protocols"]
        ),
    )
    registry = load_protocol_registry(args.protocol_registry)
    gate_schema = json.loads(
        Path(args.gate_schema).read_text(encoding="utf-8")
    )
    validation = build_no_training_materialization(
        artifacts,
        groups,
        gate_schema=gate_schema,
        registry=registry,
    )
    if args.validate_only:
        validation_output = Path(
            config.get(
                "validation_output",
                "results/coregraph_pilot/saved_output_validation_v4.json",
            )
        )
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    roles = _group_roles(groups)
    source_keys = sorted(
        key for key, value in roles.items() if value == {"source"}
    )
    target_keys = sorted(
        key for key, value in roles.items() if value == {"target"}
    )
    if not target_keys:
        raise RuntimeError("pilot requires declared target groups")

    all_rows: list[dict[str, object]] = []
    all_headline_references: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    routing_records: list[dict[str, object]] = []
    used_sources: dict[
        str,
        list[tuple[str, str, str, str, int, str]],
    ] = {}
    for target_key in target_keys:
        dataset, task, _, _, expert_seed, fold = target_key
        compatible_sources = [
            key
            for key in source_keys
            if key[0] == dataset
            and key[1] == task
            and key[4] == expert_seed
            and key[5] == fold
        ]
        if len(compatible_sources) < 2:
            raise RuntimeError(
                f"target {target_key} requires two same-seed source contracts"
            )
        source_groups: list[SavedSourceGroup] = []
        for key in compatible_sources:
            aligned_source = align_artifact_group(groups[key])
            scores = aligned_source.scores
            labels = aligned_source.labels
            splits = aligned_source.splits
            availability, costs = _availability_and_cost(
                groups[key],
                len(labels),
            )
            source_groups.append(
                SavedSourceGroup(
                    contract=groups[key][0].deployment_contract,
                    scores={
                        expert: values
                        for expert, values in scores.items()
                    },
                    labels=labels,
                    splits=splits,
                    availability=availability,
                    expert_costs=costs,
                )
            )
        aligned_target = align_artifact_group(groups[target_key])
        target_scores = aligned_target.scores
        target_labels = aligned_target.labels
        target_availability, target_costs = _availability_and_cost(
            groups[target_key],
            len(target_labels),
        )
        target_contract = groups[target_key][0].deployment_contract

        # Every learned parameter and label-free target decision freezes before
        # target labels enter the two explicit offline oracle constructors.
        router_predictions = {
            ablation: fit_saved_output_corerouter(
                source_groups,
                target_contract=target_contract,
                target_scores=target_scores,
                target_availability=target_availability,
                target_expert_costs=target_costs,
                expert_prediction_seed=expert_seed,
                ablation=ablation,
            )
            for ablation in PilotAblation
        }
        candidates = baseline_scores(
            source_groups,
            target_contract=target_contract,
            target_scores=target_scores,
            target_availability=target_availability,
            target_expert_costs=target_costs,
            expert_prediction_seed=expert_seed,
        )
        for ablation, prediction in router_predictions.items():
            name = (
                "full_corerouter"
                if ablation is PilotAblation.FULL
                else f"ablation:{ablation.value}"
            )
            candidates[name] = BaselinePrediction(
                scores=prediction.scores,
                abstention_probability=prediction.abstention_probability,
                abstain=prediction.abstain,
                forced_abstention=prediction.forced_abstention,
                expected_compute=prediction.expected_compute,
                abstention_threshold=prediction.abstention_threshold,
                abstention_threshold_provenance=(
                    prediction.abstention_threshold_fitted_on
                ),
                abstention_capacity=prediction.target_abstention_capacity,
                abstention_cost=prediction.abstention_cost,
                execution_status=_router_execution_status(
                    prediction,
                    target_availability,
                ),
                learned=True,
                adapter=f"coregraph:{ablation.value}",
                details={
                    "source_only_early_stopping": True,
                    "source_abstention_capacities": dict(
                        prediction.source_abstention_capacities
                    ),
                    "router_training_seed": prediction.router_training_seed,
                    "source_fit_hash": prediction.source_fit_hash,
                },
            )
        candidates["contract_feasible_oracle"] = contract_feasible_oracle(
            target_scores=target_scores,
            target_availability=target_availability,
            target_expert_costs=target_costs,
            target_labels=target_labels,
        )
        candidates["instance_clairvoyant_oracle_ceiling"] = (
            instance_clairvoyant_oracle_ceiling(
                target_scores=target_scores,
                target_availability=target_availability,
                target_expert_costs=target_costs,
                target_labels=target_labels,
            )
        )
        router_training_seeds = {
            method: int(
                prediction.details.get(
                    "router_training_seed",
                    derive_router_seed(expert_seed, method),
                )
            )
            for method, prediction in candidates.items()
            if not prediction.diagnostic_only
            and method != "contract_feasible_oracle"
        }
        evaluation = evaluate_saved_output_pilot(
            target_labels,
            candidates,
            dataset=dataset,
            target_protocol_id=groups[target_key][0].protocol_id,
            target_contract_coordinate_hash=target_contract.coordinate_hash,
            target_contract_id=target_contract.contract_id,
            expert_prediction_seed=expert_seed,
            router_training_seeds=router_training_seeds,
            fold=fold,
        )
        all_rows.extend(evaluation["rows"])
        all_headline_references.append(
            evaluation["headline_oracle_reference"]
        )
        all_diagnostics.extend(evaluation["diagnostic_oracles"])
        full = router_predictions[PilotAblation.FULL]
        routing_records.append(
            {
                "dataset": dataset,
                "target_protocol_id": groups[target_key][0].protocol_id,
                "target_contract_coordinate_hash": (
                    target_contract.coordinate_hash
                ),
                "target_contract_id": target_contract.contract_id,
                "seed": expert_seed,
                "expert_prediction_seed": expert_seed,
                "router_training_seed": full.router_training_seed,
                "fold": fold,
                "distinct_experts": int(
                    len(set(full.selected_experts.tolist()) - {-1})
                ),
                "perturbation_flip_rate": full.perturbation_flip_rate,
                "coverage": float(1 - full.abstain.mean()),
            }
        )
        used_sources[str(target_key)] = compatible_sources
    report = {
        "schema": "coregraph_saved_output_pilot_v4",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": all_rows,
        "headline_oracle_references": all_headline_references,
        "diagnostic_oracles": all_diagnostics,
        "routing": routing_records,
        "source_groups": used_sources,
        "target_groups": target_keys,
        "target_information": (
            "metadata known; labels used only after all fits and predictions froze"
        ),
        "target_label_selection": False,
        "oracle_target_selection": False,
        "headline_oracle": "contract_feasible_oracle",
        "diagnostic_oracle": "instance_clairvoyant_oracle_ceiling",
        "inferential_block": "dataset_stratified_expert_prediction_seed",
        "paired_unit": (
            "dataset_task_target_protocol_contract_"
            "expert_prediction_seed_fold"
        ),
        "risk_taxonomy": {
            "training_surrogate": "bce_surrogate_contract_regret",
            "headline_evaluation": "brier_contract_regret",
            "selective_evaluation": "selective_zero_one_risk",
        },
        "pre_execution_validation": validation,
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
