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

from coregraph.evaluation.corrections import benjamini_hochberg, holm  # noqa: E402
from coregraph.evaluation.statistics import paired_permutation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-plan", required=True)
    parser.add_argument(
        "--input",
        default="results/coregraph_import/analysis/paired_blocks.csv",
    )
    args = parser.parse_args()
    freeze = json.loads(Path(args.frozen_plan).read_text(encoding="utf-8"))
    plan_path = ROOT / "configs/coregraph/analysis_families.yaml"
    expected = freeze["files"]["configs/coregraph/analysis_families.yaml"]
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("analysis plan hash does not match frozen plan")
    input_path = ROOT / args.input
    if not input_path.exists():
        report = {
            "schema": "coregraph_statistical_analysis_v1",
            "status": "BLOCKED_NO_VALIDATED_PAIRED_BLOCKS",
            "results": [],
        }
        output = ROOT / "results/coregraph_build/STATISTICAL_ANALYSIS.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(input_path.open(encoding="utf-8")))
    results = []
    for family_name, family in plan["families"].items():
        family_rows = [row for row in rows if row["analysis_family"] == family_name]
        raw_p: list[float] = []
        family_results = []
        for hypothesis in family["hypotheses"]:
            selected = [row for row in family_rows if row["hypothesis"] == hypothesis]
            left = [float(row["method_value"]) for row in selected]
            right = [float(row["baseline_value"]) for row in selected]
            inference = paired_permutation(left, right) if selected else None
            raw_p.append(1.0 if inference is None else inference.p_value)
            family_results.append(
                {
                    "hypothesis": hypothesis,
                    "blocks": len(selected),
                    "raw_p": raw_p[-1],
                    "mean_difference": None if inference is None else inference.mean_difference,
                }
            )
        correction = (
            holm(raw_p, family["alpha"])
            if family["correction"] == "holm"
            else benjamini_hochberg(raw_p, family["alpha"])
        )
        for record, adjusted, reject in zip(
            family_results, correction.adjusted, correction.reject, strict=True
        ):
            record.update({"adjusted_p": adjusted, "reject": reject})
        results.append({"family": family_name, "tests": family_results})
    report = {
        "schema": "coregraph_statistical_analysis_v1",
        "status": "PASS",
        "results": results,
    }
    output = ROOT / "results/coregraph_build/STATISTICAL_ANALYSIS.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "families": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
