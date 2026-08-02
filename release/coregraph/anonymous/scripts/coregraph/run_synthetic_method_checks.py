#!/usr/bin/env python3
"""Run tiny deterministic qualitative CoReGraph mechanism checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.data.synthetic import (  # noqa: E402
    SyntheticControls,
    SyntheticRegime,
    generate_contract_shift,
    oracle_expert_ranking,
)


def risk(scores: np.ndarray, labels: np.ndarray, mask=None) -> float:
    keep = np.ones(len(labels), dtype=bool) if mask is None else mask
    return float(np.mean((scores[keep] - labels[keep]) ** 2))


def recall(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(labels) * fraction)))
    selected = np.argsort(-scores, kind="stable")[:k]
    return float(labels[selected].sum() / max(labels.sum(), 1))


def main() -> int:
    controls = SyntheticControls(num_nodes=128, seed=20260729)
    samples = {
        regime: generate_contract_shift(regime, controls)
        for regime in SyntheticRegime
    }
    crossing = samples[SyntheticRegime.ORDERING_CROSSES]
    early = crossing.timestamps < crossing.controls.time_steps // 2
    late = ~early
    budget = samples[SyntheticRegime.BUDGET_CHANGES_EXPERT]
    calibration = samples[SyntheticRegime.CALIBRATION_MISMATCH]
    checks = {
        "graph_expert_wins": (
            oracle_expert_ranking(samples[SyntheticRegime.GRAPH_BEST])[0][0]
            == "graph"
        ),
        "feature_expert_wins": (
            oracle_expert_ranking(samples[SyntheticRegime.FEATURE_BEST])[0][0]
            == "feature"
        ),
        "crossing_expertise": (
            risk(crossing.expert_scores["graph"], crossing.labels, early)
            < risk(crossing.expert_scores["feature"], crossing.labels, early)
            and risk(crossing.expert_scores["feature"], crossing.labels, late)
            < risk(crossing.expert_scores["graph"], crossing.labels, late)
        ),
        "unseen_combination": bool(
            samples[SyntheticRegime.FACTORISED_GENERALISATION]
            .mechanism_report["additive_axis_effect"]
        ),
        "interaction_breaks_factorisation": (
            samples[SyntheticRegime.INTERACTION_BREAKS_FACTORISATION]
            .mechanism_report["interaction_residual"]
            > 0
        ),
        "resource_mask_removes_best": (
            not samples[SyntheticRegime.RESOURCE_MASK].expert_available["graph"]
            and oracle_expert_ranking(
                samples[SyntheticRegime.RESOURCE_MASK]
            )[0][0]
            == "feature"
        ),
        "budget_changes_preferred_expert": (
            recall(budget.expert_scores["graph"], budget.labels, 0.005)
            > recall(budget.expert_scores["feature"], budget.labels, 0.005)
            and recall(budget.expert_scores["feature"], budget.labels, 0.2)
            > recall(budget.expert_scores["graph"], budget.labels, 0.2)
        ),
        "noisy_contract_metadata": (
            samples[SyntheticRegime.NOISY_CONTRACT_METADATA]
            .mechanism_report["contract_metadata_noise"]
            > 0
        ),
        "all_experts_unavailable": not any(
            samples[SyntheticRegime.ALL_EXPERTS_UNAVAILABLE]
            .expert_available.values()
        ),
        "calibration_mismatch": (
            risk(
                calibration.expert_scores["feature"],
                calibration.labels,
            )
            < risk(
                calibration.expert_scores["graph"],
                calibration.labels,
            )
        ),
    }
    report = {
        "schema": "coregraph_synthetic_method_checks_v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "deterministic_seed": controls.seed,
        "examples_per_scenario": controls.num_nodes,
        "checks": checks,
        "real_dataset_used": False,
        "multi_seed_experiment": False,
    }
    output = ROOT / "results/coregraph_build/SYNTHETIC_METHOD_CHECKS.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
