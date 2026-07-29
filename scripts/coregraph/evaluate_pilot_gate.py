#!/usr/bin/env python3
"""Evaluate predeclared seed-blocked pilot go/no-go gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evaluation.corrections import holm  # noqa: E402
from coregraph.evaluation.statistics import (  # noqa: E402
    bootstrap_seed_blocks,
    build_paired_seed_blocks,
    exact_wilcoxon,
    paired_permutation,
)

REQUIRED_BASELINES = {
    "average_all_feasible",
    "best_source_validation",
    "source_validation_convex_mixture",
    "graphsafe_v2_adapter",
    "current_graph_feature_gate_adapter",
    "learned_no_contract_router",
    "learned_atomic_contract_router",
    "MOWST_INSPIRED_REIMPLEMENTATION",
}
HIGHER_IS_BETTER = {
    "auprc",
    "recall_at_0.5pct",
    "recall_at_1pct",
    "recall_at_2pct",
    "budget_curve_area",
}
LOWER_IS_BETTER = {
    "mean_regret",
    "maximum_regret",
    "cvar_regret",
    "selective_risk",
    "compute",
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


def _holm_family(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "ranking_and_budget"
    if metric in {"mean_regret", "maximum_regret", "cvar_regret"}:
        return "robust_risk"
    if metric in {"selective_risk", "compute"}:
        return "deployment"
    raise ValueError(f"metric {metric!r} has no declared Holm family")


def _validate_gate_schema(schema: dict[str, Any]) -> None:
    if set(schema.get("required_strong_baselines", ())) != REQUIRED_BASELINES:
        raise ValueError("pilot gate schema baseline registry does not match code")
    declared_ablations = {
        f"ablation:{name}"
        for name in schema.get("required_ablations", ())
    }
    if declared_ablations != REQUIRED_ABLATIONS:
        raise ValueError("pilot gate schema ablations do not match code")
    declared_families = schema.get("holm_families", {})
    if not isinstance(declared_families, dict):
        raise ValueError("pilot gate schema Holm families must be a mapping")
    for family, metrics in declared_families.items():
        if any(_holm_family(str(metric)) != family for metric in metrics):
            raise ValueError("pilot gate schema Holm family mapping is inconsistent")
    if set(declared_families) != {
        "ranking_and_budget",
        "robust_risk",
        "deployment",
    }:
        raise ValueError("pilot gate schema must declare all three Holm families")


def _derive_regret_rows(rows):
    contract_rows = [
        row for row in rows if str(row["metric"]) == "contract_regret"
    ]
    grouped = defaultdict(list)
    for row in contract_rows:
        grouped[
            (
                str(row["dataset"]),
                int(row["seed"]),
                str(row.get("fold", "")),
                str(row["method"]),
            )
        ].append(float(row["value"]))
    derived = []
    for (dataset, seed, fold, method), values in grouped.items():
        ordered = sorted(values, reverse=True)
        tail_count = max(1, int(np.ceil(0.2 * len(ordered))))
        metrics = {
            "mean_regret": float(np.mean(values)),
            "maximum_regret": float(np.max(values)),
            "cvar_regret": float(np.mean(ordered[:tail_count])),
        }
        for metric, value in metrics.items():
            derived.append(
                {
                    "dataset": dataset,
                    "target_contract": "__seed_aggregate__",
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                    "metric": metric,
                    "value": value,
                }
            )
    return derived


def _worst_contract_seed_deltas(rows, baseline):
    values = {
        (
            str(row["dataset"]),
            str(row["target_contract"]),
            int(row["seed"]),
            str(row.get("fold", "")),
            str(row["method"]),
        ): float(row["value"])
        for row in rows
        if str(row["metric"]) == "auprc"
        and str(row["method"]) in {"full_corerouter", baseline}
    }
    contexts = {
        key[:4] for key in values if key[4] == "full_corerouter"
    }
    by_seed = defaultdict(list)
    for context in contexts:
        method_key = (*context, "full_corerouter")
        baseline_key = (*context, baseline)
        if baseline_key not in values:
            raise ValueError(f"missing paired target-contract outcome {baseline_key}")
        by_seed[context[2]].append(
            values[method_key] - values[baseline_key]
        )
    return tuple(
        min(by_seed[seed]) for seed in sorted(by_seed)
    )


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
    parser.add_argument(
        "--output",
        default="results/coregraph_pilot/gate_report.json",
    )
    args = parser.parse_args()
    result_path = Path(args.pilot_result)
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    _validate_gate_schema(schema)
    if not result_path.exists():
        report = {
            "status": "BLOCKED_MISSING_PILOT_RESULT",
            "passed": False,
            "criteria": schema["criteria"],
        }
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = list(result.get("rows", []))
        methods = {str(row["method"]) for row in rows}
        baselines = sorted(
            (REQUIRED_BASELINES | {name for name in methods if name.startswith("expert:")})
            & methods
        )
        missing_baselines = sorted(REQUIRED_BASELINES - methods)
        if "full_corerouter" not in methods or missing_baselines:
            report = {
                "status": "BLOCKED_INCOMPLETE_METHOD_ROWS",
                "passed": False,
                "missing_baselines": missing_baselines,
                "criteria": schema["criteria"],
            }
        else:
            analysis_rows = [*rows, *_derive_regret_rows(rows)]
            comparisons: list[dict[str, Any]] = []
            family_p_values = defaultdict(list)
            family_positions = defaultdict(list)
            for metric in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
                for baseline in baselines:
                    blocks = build_paired_seed_blocks(
                        analysis_rows,
                        method="full_corerouter",
                        baseline=baseline,
                        metric=metric,
                    )
                    method_values = np.asarray(blocks.method_values)
                    baseline_values = np.asarray(blocks.baseline_values)
                    improvement = (
                        method_values - baseline_values
                        if metric in HIGHER_IS_BETTER
                        else baseline_values - method_values
                    )
                    wilcoxon = exact_wilcoxon(
                        improvement,
                        np.zeros_like(improvement),
                        alternative="greater",
                    )
                    permutation = paired_permutation(
                        improvement,
                        np.zeros_like(improvement),
                        alternative="greater",
                    )
                    low, high = bootstrap_seed_blocks(improvement)
                    record = {
                        "metric": metric,
                        "baseline": baseline,
                        "seeds": list(blocks.seeds),
                        "contexts_per_seed": list(blocks.contexts_per_seed),
                        "mean_improvement": float(improvement.mean()),
                        "exact_wilcoxon_p": wilcoxon.p_value,
                        "paired_permutation_p": permutation.p_value,
                        "seed_block_bootstrap_95": [low, high],
                    }
                    comparisons.append(record)
                    family = _holm_family(metric)
                    family_p_values[family].append(permutation.p_value)
                    family_positions[family].append(len(comparisons) - 1)
            for family, p_values in family_p_values.items():
                corrected = holm(p_values, alpha=0.05)
                for position, adjusted, reject in zip(
                    family_positions[family],
                    corrected.adjusted,
                    corrected.reject,
                    strict=True,
                ):
                    comparisons[position]["holm_adjusted_p"] = adjusted
                    comparisons[position]["holm_reject"] = reject

            worst_contract = {
                baseline: _worst_contract_seed_deltas(rows, baseline)
                for baseline in baselines
            }
            worst_contract_gain = all(
                values and float(np.mean(values)) > 0
                for values in worst_contract.values()
            )
            auprc_comparisons: list[dict[str, Any]] = [
                row for row in comparisons if row["metric"] == "auprc"
            ]
            routing = list(result.get("routing", []))
            criteria = {
                "every_predeclared_strong_baseline_compared": (
                    not missing_baselines
                    and set(baselines) >= REQUIRED_BASELINES
                ),
                "paired_seed_blocks": all(
                    len(row["seeds"]) >= 2 for row in comparisons
                ),
                "worst_contract_gain_from_paired_outcomes": worst_contract_gain,
                "average_utility_guardrail": all(
                    row["mean_improvement"] >= -0.002
                    for row in auprc_comparisons
                ),
                "paired_ci_excludes_material_harm": all(
                    row["seed_block_bootstrap_95"][0] >= -0.002
                    for row in auprc_comparisons
                ),
                "declared_holm_families": all(
                    "holm_adjusted_p" in row for row in comparisons
                ),
                "ablation_contribution": (
                    REQUIRED_ABLATIONS.issubset(methods)
                ),
                "no_oracle_target_information": (
                    result.get("oracle_target_selection") is False
                ),
                "routing_diversity": bool(
                    routing
                    and min(
                        int(record["distinct_experts"])
                        for record in routing
                    )
                    >= 2
                ),
                "routing_stability": bool(
                    routing
                    and max(
                        float(record["perturbation_flip_rate"])
                        for record in routing
                    )
                    <= 0.1
                ),
            }
            report = {
                "status": "GATE_EVALUATED",
                "passed": all(criteria.values()),
                "criteria": criteria,
                "comparisons": comparisons,
                "worst_contract_seed_deltas": worst_contract,
                "holm_families": sorted(family_p_values),
                "pairing_unit": "seed after exact target-context pairing",
            }
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
