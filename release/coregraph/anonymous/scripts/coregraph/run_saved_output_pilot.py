#!/usr/bin/env python3
"""Plan or execute the seed-bound saved-prediction pilot V3."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/coregraph/pilot/saved_output.yaml",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    roots = [
        os.path.expandvars(value)
        for value in config.get("prediction_manifest_roots", [])
    ]
    manifests = discover_prediction_manifests(roots)
    plan = {
        "schema": "coregraph_saved_output_pilot_plan_v3",
        "execute_requested": args.execute,
        "manifest_roots": roots,
        "discovered_manifests": [str(path) for path in manifests],
        "status": "PLANNED",
        "forbidden_target_selection": True,
        "required_datasets": config["required_datasets"],
        "required_target_contracts": config["required_target_contracts"],
        "required_expert_prediction_seeds": config["required_seeds"],
        "required_ablations": [
            ablation.value
            for ablation in PilotAblation
            if ablation is not PilotAblation.FULL
        ],
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
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
        expected_target_contracts=tuple(
            config["required_target_contracts"]
        ),
    )
    roles = {
        key: {artifact.contract_role for artifact in group}
        for key, group in groups.items()
    }
    if any(len(value) != 1 for value in roles.values()):
        raise RuntimeError(
            "each prediction group must declare exactly one contract role"
        )
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
            _, scores, labels, splits = align_artifact_group(groups[key])
            keep = np.isin(splits, ("train", "validation"))
            availability, costs = _availability_and_cost(
                groups[key],
                int(keep.sum()),
            )
            source_groups.append(
                SavedSourceGroup(
                    contract=groups[key][0].deployment_contract,
                    scores={
                        expert: values[keep]
                        for expert, values in scores.items()
                    },
                    labels=labels[keep],
                    splits=splits[keep],
                    availability=availability,
                    expert_costs=costs,
                )
            )
        _, target_scores, target_labels, _ = align_artifact_group(
            groups[target_key]
        )
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
            target_contract=target_contract.contract_id,
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
                "target_contract": target_contract.contract_id,
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
        "schema": "coregraph_saved_output_pilot_v3",
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
        "inferential_block": "expert_prediction_seed",
        "paired_unit": (
            "dataset_task_target_contract_expert_prediction_seed_fold"
        ),
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
