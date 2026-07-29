#!/usr/bin/env python3
"""Evaluate predeclared pilot go/no-go gates without inventing missing results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-result",
        default="results/coregraph_pilot/saved_output_pilot.json",
    )
    parser.add_argument(
        "--schema",
        default="results/coregraph_build/PILOT_GO_NO_GO_SCHEMA.json",
    )
    parser.add_argument("--output", default="results/coregraph_pilot/gate_report.json")
    args = parser.parse_args()
    result_path = Path(args.pilot_result)
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    if not result_path.exists():
        report = {
            "status": "BLOCKED_MISSING_PILOT_RESULT",
            "passed": False,
            "criteria": schema["criteria"],
        }
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = result.get("rows", [])
        by_method: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            by_method.setdefault(str(row["method"]), []).append(row)
        baselines = {
            method: values
            for method, values in by_method.items()
            if method != "full_corerouter"
        }
        full_rows = by_method.get("full_corerouter", [])
        common_counts = {
            method: len(values)
            for method, values in baselines.items()
            if len(values) == len(full_rows)
        }
        strongest_method = (
            max(
                common_counts,
                key=lambda method: float(
                    np.mean([float(row["auprc"]) for row in baselines[method]])
                ),
            )
            if common_counts
            else None
        )
        if not full_rows or strongest_method is None:
            report = {
                "status": "BLOCKED_INCOMPLETE_METHOD_ROWS",
                "passed": False,
                "strongest_baseline": strongest_method,
                "criteria": schema["criteria"],
            }
        else:
            group_key = lambda row: (str(row["dataset"]), str(row["contract_id"]))
            full = {group_key(row): row for row in full_rows}
            baseline = {group_key(row): row for row in baselines[strongest_method]}
            groups = sorted(set(full) & set(baseline))
            deltas = np.asarray(
                [
                    float(full[group]["auprc"]) - float(baseline[group]["auprc"])
                    for group in groups
                ]
            )
            rng = np.random.default_rng(20260729)
            bootstrap = (
                deltas[
                    rng.integers(0, len(deltas), size=(10_000, len(deltas)))
                ].mean(axis=1)
                if len(deltas)
                else np.asarray([])
            )
            lower = float(np.quantile(bootstrap, 0.025)) if len(bootstrap) else float("nan")
            required_datasets = {"elliptic", "dgraphfin"}
            dataset_gain = {
                dataset: float(
                    np.mean(
                        [
                            deltas[index]
                            for index, group in enumerate(groups)
                            if group[0] == dataset
                        ]
                    )
                )
                for dataset in required_datasets
                if any(group[0] == dataset for group in groups)
            }
            no_contract = {
                group_key(row): row
                for row in by_method.get("no_contract_router", [])
            }
            ablation_deltas = [
                float(full[group]["auprc"]) - float(no_contract[group]["auprc"])
                for group in groups
                if group in no_contract
            ]
            routing = result.get("routing", [])
            criteria = {
                "strongest_baseline_identified": strongest_method is not None,
                "improvement_on_elliptic_and_dgraphfin": (
                    set(dataset_gain) == required_datasets
                    and all(value > 0 for value in dataset_gain.values())
                ),
                "worst_contract_gain": (
                    min(float(row["auprc"]) for row in full.values())
                    > min(float(row["auprc"]) for row in baseline.values())
                ),
                "average_utility_guardrail": bool(
                    len(deltas) and deltas.mean() >= -0.002
                ),
                "paired_ci_excludes_material_harm": bool(
                    len(deltas) >= 2 and lower >= -0.002
                ),
                "ablation_contribution": bool(
                    ablation_deltas and np.mean(ablation_deltas) > 0
                ),
                "no_oracle_target_information": (
                    result.get("oracle_target_selection") is False
                ),
                "routing_diversity": bool(
                    routing and min(record["distinct_experts"] for record in routing) >= 2
                ),
                "routing_stability": bool(
                    routing
                    and max(record["perturbation_flip_rate"] for record in routing) <= 0.1
                ),
            }
            report = {
                "status": "GATE_EVALUATED",
                "passed": all(criteria.values()),
                "strongest_baseline": strongest_method,
                "criteria": criteria,
                "dataset_auprc_deltas": dataset_gain,
                "average_auprc_delta": float(deltas.mean()),
                "paired_bootstrap_lower_95": lower,
                "groups": len(groups),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"].startswith("BLOCKED"):
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
