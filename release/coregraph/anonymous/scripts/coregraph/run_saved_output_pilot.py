#!/usr/bin/env python3
"""Plan or execute the seed-bound saved-prediction pilot."""

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
    PilotAblation,
    SavedSourceGroup,
    align_artifact_group,
    baseline_scores,
    discover_prediction_manifests,
    evaluate_saved_output_pilot,
    fit_saved_output_corerouter,
    load_prediction_artifacts,
    offline_feasible_oracle_ceiling,
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
        "schema": "coregraph_saved_output_pilot_plan_v2",
        "execute_requested": args.execute,
        "manifest_roots": roots,
        "discovered_manifests": [str(path) for path in manifests],
        "status": "PLANNED",
        "forbidden_target_selection": True,
        "required_ablations": [
            ablation.value for ablation in PilotAblation if ablation is not PilotAblation.FULL
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
    )
    roles = {
        key: {artifact.contract_role for artifact in group}
        for key, group in groups.items()
    }
    if any(len(value) != 1 for value in roles.values()):
        raise RuntimeError("each prediction group must declare exactly one contract role")
    source_keys = sorted(key for key, value in roles.items() if value == {"source"})
    target_keys = sorted(key for key, value in roles.items() if value == {"target"})
    if not target_keys:
        raise RuntimeError("pilot requires at least one declared target group")

    all_rows: list[dict[str, object]] = []
    routing_records: list[dict[str, object]] = []
    used_sources: dict[
        str,
        list[tuple[str, str, str, str, int, str]],
    ] = {}
    for target_key in target_keys:
        dataset, task, _, _, seed, fold = target_key
        compatible_sources = [
            key
            for key in source_keys
            if key[0] == dataset
            and key[1] == task
            and key[4] == seed
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
                        expert: values[keep] for expert, values in scores.items()
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

        # Every learned prediction and source-selected hyperparameter is frozen
        # before target_labels is passed to baseline/evaluation code.
        router_predictions = {
            ablation: fit_saved_output_corerouter(
                source_groups,
                target_contract=groups[target_key][0].deployment_contract,
                target_scores=target_scores,
                target_availability=target_availability,
                target_expert_costs=target_costs,
                seed=int(config.get("router_seed", 20260729)),
                ablation=ablation,
            )
            for ablation in PilotAblation
        }
        candidates = baseline_scores(
            source_groups,
            target_scores=target_scores,
            target_availability=target_availability,
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
                expected_compute=prediction.expected_compute,
                learned=True,
                adapter=f"coregraph:{ablation.value}",
                details={
                    "source_only_early_stopping": True,
                    "abstention_threshold_fitted_on": (
                        prediction.abstention_threshold_fitted_on
                    ),
                },
            )
        # This is the first point at which target labels enter method
        # construction, and only the explicitly offline ceiling consumes them.
        candidates["offline_feasible_oracle_ceiling"] = (
            offline_feasible_oracle_ceiling(
                target_scores=target_scores,
                target_availability=target_availability,
                target_expert_costs=target_costs,
                target_labels=target_labels,
            )
        )
        evaluation = evaluate_saved_output_pilot(
            target_labels,
            candidates,
            dataset=dataset,
            target_contract=(
                groups[target_key][0].deployment_contract.contract_id
            ),
            seed=seed,
            fold=fold,
        )
        all_rows.extend(evaluation["rows"])
        full = router_predictions[PilotAblation.FULL]
        routing_records.append(
            {
                "dataset": dataset,
                "target_contract": (
                    groups[target_key][0].deployment_contract.contract_id
                ),
                "seed": seed,
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
        "schema": "coregraph_saved_output_pilot_v2",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": all_rows,
        "routing": routing_records,
        "source_groups": used_sources,
        "target_groups": target_keys,
        "target_information": "labels_used_only_after_all fits and predictions froze",
        "oracle_target_selection": False,
        "paired_unit": "dataset_task_target_contract_seed_fold",
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
