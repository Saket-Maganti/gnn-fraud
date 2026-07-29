#!/usr/bin/env python3
"""Discover, validate, align, and evaluate saved prediction outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coregraph.experiments.pilot import (
    align_artifact_group,
    baseline_scores,
    discover_prediction_manifests,
    evaluate_saved_output_pilot,
    fit_saved_output_corerouter,
    load_prediction_artifacts,
)


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
        "schema": "coregraph_saved_output_pilot_plan_v1",
        "execute_requested": args.execute,
        "manifest_roots": roots,
        "discovered_manifests": [str(path) for path in manifests],
        "status": "PLANNED",
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    artifacts = load_prediction_artifacts(manifests)
    grouped = defaultdict(list)
    for artifact in artifacts:
        grouped[(artifact.dataset, artifact.contract_id, artifact.prediction_unit)].append(artifact)
    if len(grouped) < 3:
        raise RuntimeError(
            "pilot execution needs at least two source and one target contract groups"
        )
    roles = {
        key: {artifact.contract_role for artifact in group}
        for key, group in grouped.items()
    }
    if any(len(value) != 1 for value in roles.values()):
        raise RuntimeError("each prediction group must declare exactly one contract role")
    source_keys = sorted(key for key, value in roles.items() if value == {"source"})
    target_keys = sorted(key for key, value in roles.items() if value == {"target"})
    if not target_keys:
        raise RuntimeError("pilot requires at least one declared target group")
    all_rows = []
    routing_records = []
    used_sources = {}
    for target_key in target_keys:
        compatible_sources = [
            key
            for key in source_keys
            if key[0] == target_key[0] and key[2] == target_key[2]
        ]
        if len(compatible_sources) < 2:
            raise RuntimeError(
                f"target {target_key} requires two same-dataset/task source contracts"
            )
        _, source_scores, source_labels, source_split = align_artifact_group(
            grouped[compatible_sources[0]]
        )
        _, target_scores, target_labels, _ = align_artifact_group(grouped[target_key])
        candidates = baseline_scores(
            source_scores,
            source_labels,
            source_split,
            target_scores,
        )
        source_groups = []
        for key in compatible_sources:
            _, scores, labels, splits = align_artifact_group(grouped[key])
            source_groups.append(
                (grouped[key][0].deployment_contract, scores, labels, splits)
            )
        router_prediction = fit_saved_output_corerouter(
            source_groups,
            target_contract=grouped[target_key][0].deployment_contract,
            target_scores=target_scores,
            seed=int(config.get("router_seed", 20260729)),
        )
        candidates["full_corerouter"] = router_prediction.scores
        evaluation = evaluate_saved_output_pilot(
            target_labels,
            candidates,
            review_fraction=float(config.get("review_fraction", 0.01)),
            dataset=target_key[0],
            contract_id=target_key[1],
        )
        all_rows.extend(evaluation["rows"])
        routing_records.append(
            {
                "dataset": target_key[0],
                "contract_id": target_key[1],
                "distinct_experts": int(
                    len(set(router_prediction.selected_experts.tolist()) - {-1})
                ),
                "perturbation_flip_rate": router_prediction.perturbation_flip_rate,
            }
        )
        used_sources[str(target_key)] = compatible_sources
    report = {
        "schema": "coregraph_saved_output_pilot_v1",
        "status": "MEASURED_FROM_SAVED_PREDICTIONS",
        "rows": all_rows,
        "routing": routing_records,
        "source_groups": used_sources,
        "target_groups": target_keys,
        "target_information": "labels_used_for_final_offline_scoring_only",
        "oracle_target_selection": False,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
