#!/usr/bin/env python3
"""Execute only the frozen statistical families on imported summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.evaluation.corrections import holm  # noqa: E402
from coregraph.evaluation.statistics import (  # noqa: E402
    bootstrap_seed_blocks,
    exact_wilcoxon,
    paired_permutation,
)


def _seed_block_values(
    rows: list[dict[str, str]],
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Pair exact contexts, then aggregate target contracts within seed."""

    by_context: dict[tuple[str, str, int, str], tuple[float, float]] = {}
    for row in rows:
        context = (
            row["dataset"],
            row["target_contract"],
            int(row["seed"]),
            row.get("fold", ""),
        )
        if context in by_context:
            raise ValueError(f"duplicate statistical context {context}")
        by_context[context] = (
            float(row["method_value"]),
            float(row["baseline_value"]),
        )
    by_seed: dict[int, list[tuple[float, float]]] = {}
    for context, pair in sorted(by_context.items()):
        by_seed.setdefault(context[2], []).append(pair)
    if not by_seed:
        raise ValueError("statistical comparison has no paired seed blocks")
    counts = {len(pairs) for pairs in by_seed.values()}
    if len(counts) != 1:
        raise ValueError("paired seeds have unequal target-context coverage")
    seeds = sorted(by_seed)
    method = [
        sum(pair[0] for pair in by_seed[seed]) / len(by_seed[seed])
        for seed in seeds
    ]
    baseline = [
        sum(pair[1] for pair in by_seed[seed]) / len(by_seed[seed])
        for seed in seeds
    ]
    return seeds, method, baseline, [len(by_seed[seed]) for seed in seeds]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-plan", required=True)
    parser.add_argument(
        "--input",
        default="results/coregraph_import/analysis/paired_blocks.csv",
    )
    parser.add_argument(
        "--output",
        default="results/coregraph_build/STATISTICAL_ANALYSIS.json",
    )
    args = parser.parse_args()
    freeze = json.loads(Path(args.frozen_plan).read_text(encoding="utf-8"))
    plan_path = ROOT / "configs/coregraph/analysis_families.yaml"
    expected = freeze["files"]["configs/coregraph/analysis_families.yaml"]
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("analysis plan hash does not match frozen plan")
    input_path = ROOT / args.input
    if not input_path.exists():
        report: dict[str, object] = {
            "schema": "coregraph_statistical_analysis_v1",
            "status": "BLOCKED_NO_VALIDATED_PAIRED_BLOCKS",
            "results": [],
        }
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(input_path.open(encoding="utf-8")))
    results: list[dict[str, object]] = []
    for family_name, family in plan["families"].items():
        family_rows = [row for row in rows if row["analysis_family"] == family_name]
        raw_p: list[float] = []
        family_results = []
        for outcome in family["outcomes"]:
            outcome_rows = [
                row for row in family_rows if row["metric"] == outcome
            ]
            baselines = sorted({row["baseline"] for row in outcome_rows})
            if not baselines:
                raise ValueError(
                    f"analysis family {family_name}/{outcome} has no baselines"
                )
            for baseline_name in baselines:
                selected = [
                    row
                    for row in outcome_rows
                    if row["baseline"] == baseline_name
                ]
                seeds, left, right, contexts = _seed_block_values(selected)
                higher_is_better = outcome in {
                    "auprc",
                    "recall_at_0.5pct",
                    "recall_at_1pct",
                    "recall_at_2pct",
                    "budget_curve_area",
                }
                improvement = [
                    method - baseline
                    if higher_is_better
                    else baseline - method
                    for method, baseline in zip(left, right, strict=True)
                ]
                zeros = [0.0] * len(improvement)
                wilcoxon = exact_wilcoxon(
                    improvement,
                    zeros,
                    alternative="greater",
                )
                permutation = paired_permutation(
                    improvement,
                    zeros,
                    alternative="greater",
                )
                interval = bootstrap_seed_blocks(improvement)
                raw_p.append(permutation.p_value)
                family_results.append(
                    {
                        "outcome": outcome,
                        "baseline": baseline_name,
                        "seeds": seeds,
                        "contexts_per_seed": contexts,
                        "mean_improvement": permutation.mean_difference,
                        "exact_wilcoxon_p": wilcoxon.p_value,
                        "paired_permutation_p": permutation.p_value,
                        "seed_block_bootstrap_95": list(interval),
                    }
                )
        if family["correction"] != "holm":
            raise ValueError(
                f"analysis family {family_name} must use declared Holm correction"
            )
        correction = holm(raw_p, family["alpha"])
        for record, adjusted, reject in zip(
            family_results, correction.adjusted, correction.reject, strict=True
        ):
            record.update({"holm_adjusted_p": adjusted, "holm_reject": reject})
        results.append({"family": family_name, "tests": family_results})
    report = {
        "schema": "coregraph_statistical_analysis_v1",
        "status": "PASS",
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "families": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
