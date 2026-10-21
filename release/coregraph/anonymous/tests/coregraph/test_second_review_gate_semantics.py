from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.coregraph import evaluate_pilot_gate as gate


ROOT = Path(__file__).resolve().parents[2]
FROZEN_GATE_SPEC = (
    ROOT / "results/coregraph_build/PILOT_GATE_FROZEN_SPEC.json"
)
if not FROZEN_GATE_SPEC.is_file():
    FROZEN_GATE_SPEC = ROOT / "specifications/PILOT_GATE_FROZEN_SPEC.json"


def _schema() -> dict[str, object]:
    return {
        "schema_version": "coregraph_pilot_gate_v3",
        "required_datasets": ["elliptic", "dgraphfin"],
        "required_target_contracts": [
            "strict_inductive",
            "isolated_inductive",
        ],
        "required_expert_prediction_seeds": list(range(1, 11)),
        "required_folds": ["fold0"],
        "required_experts": ["feature_mlp", "gcn"],
        "required_strong_baselines": [
            "average_all_feasible",
            "best_source_validation",
            "graphsafe_confidence_abstention_component",
        ],
        "required_ablations": [
            "no_contract",
            "atomic_contract",
            "no_regret",
            "no_budget",
            "no_resource_mask",
            "no_stability",
            "no_abstention",
            "no_diagnostics",
        ],
        "primary_outcomes": [
            "auprc",
            "budget_curve_area",
            "cvar_regret",
            "selective_risk",
            "abstention_cost",
        ],
        "ranking_outcomes": ["auprc", "budget_curve_area"],
        "holm_families": {
            "ranking_and_budget": ["auprc", "budget_curve_area"],
            "robust_risk": ["cvar_regret"],
            "deployment": ["selective_risk", "abstention_cost"],
        },
        "ablation_tests": {
            "no_contract": {
                "metric": "cvar_regret",
                "direction": "lower",
                "minimum_effect": 0.001,
                "required_contribution": True,
            },
            "atomic_contract": {
                "metric": "cvar_regret",
                "direction": "lower",
                "minimum_effect": 0.001,
                "required_contribution": True,
            },
            "no_regret": {
                "metric": "cvar_regret",
                "direction": "lower",
                "minimum_effect": 0.001,
                "required_contribution": True,
            },
            "no_budget": {
                "metric": "budget_curve_area",
                "direction": "higher",
                "minimum_effect": 0.001,
                "required_contribution": True,
            },
            "no_resource_mask": {
                "metric": "cvar_regret",
                "direction": "lower",
                "minimum_effect": 0.001,
                "required_contribution": True,
            },
            "no_stability": {
                "metric": "auprc",
                "direction": "higher",
                "minimum_effect": 0.0,
                "required_contribution": False,
            },
            "no_abstention": {
                "metric": "selective_risk",
                "direction": "lower",
                "minimum_effect": 0.0,
                "required_contribution": False,
            },
            "no_diagnostics": {
                "metric": "auprc",
                "direction": "higher",
                "minimum_effect": 0.0,
                "required_contribution": False,
            },
        },
        "alpha": 0.05,
    }


def _complete_rows(schema: dict[str, object]) -> list[dict[str, object]]:
    methods = [
        "full_corerouter",
        *[f"expert:{name}" for name in schema["required_experts"]],
        *schema["required_strong_baselines"],
        *[f"ablation:{name}" for name in schema["required_ablations"]],
    ]
    rows = []
    for dataset in schema["required_datasets"]:
        for contract in schema["required_target_contracts"]:
            for seed in schema["required_expert_prediction_seeds"]:
                for fold in schema["required_folds"]:
                    for method in methods:
                        router_seed = gate.derive_router_seed(seed, method)
                        for metric in schema["primary_outcomes"]:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "target_contract": contract,
                                    "expert_prediction_seed": seed,
                                    "router_training_seed": router_seed,
                                    "fold": fold,
                                    "method": method,
                                    "metric": metric,
                                    "value": 0.5,
                                    "execution_status": "EXECUTABLE",
                                }
                            )
    return rows


def test_completeness_requires_both_datasets_all_ten_seeds_and_exact_cells() -> None:
    schema = _schema()
    rows = _complete_rows(schema)
    assert gate._validate_complete_coverage(rows, schema)["complete"]

    one_dataset = [row for row in rows if row["dataset"] == "elliptic"]
    assert not gate._validate_complete_coverage(one_dataset, schema)["complete"]

    two_seeds = [
        row for row in rows if row["expert_prediction_seed"] in {1, 2}
    ]
    assert not gate._validate_complete_coverage(two_seeds, schema)["complete"]

    duplicate = [*rows, copy.deepcopy(rows[0])]
    assert not gate._validate_complete_coverage(duplicate, schema)["complete"]


def test_ablation_gate_measures_effects_instead_of_names() -> None:
    schema = _schema()
    rows = _complete_rows(schema)
    flat = gate._evaluate_ablation_effects(rows, schema)
    assert len(flat) == 8
    assert not all(record["meaningful"] for record in flat)

    for row in rows:
        if row["method"] == "full_corerouter":
            if row["metric"] in {"cvar_regret", "selective_risk"}:
                row["value"] = 0.1
            else:
                row["value"] = 0.9
        elif str(row["method"]).startswith("ablation:"):
            if row["metric"] in {"cvar_regret", "selective_risk"}:
                row["value"] = 0.3
            else:
                row["value"] = 0.7
    effects = gate._evaluate_ablation_effects(rows, schema)
    required = [record for record in effects if record["required_contribution"]]
    assert required
    assert all(record["meaningful"] for record in required)


def test_holm_correction_changes_pass_fail_not_just_report_fields() -> None:
    records = [
        {"raw_p": 0.04, "mean_improvement": 0.02, "minimum_effect": 0.01}
        for _ in range(3)
    ]
    corrected = gate._apply_holm_to_effect_records(records, alpha=0.05)
    assert all(record["raw_p"] < 0.05 for record in corrected)
    assert not any(record["holm_reject"] for record in corrected)
    assert not any(record["meaningful"] for record in corrected)


def test_worst_case_is_minimum_of_matched_contract_differences() -> None:
    rows = []
    for seed in (1, 2):
        for contract, full, baseline in (
            ("a", 0.9, 0.7),
            ("b", 0.8, 0.85),
        ):
            for method, value in (
                ("full_corerouter", full),
                ("average_all_feasible", baseline),
            ):
                rows.append(
                    {
                        "dataset": "elliptic",
                        "target_contract": contract,
                        "expert_prediction_seed": seed,
                        "fold": "fold0",
                        "method": method,
                        "metric": "auprc",
                        "value": value,
                    }
                )
    matched = gate._matched_worst_contract_seed_deltas(
        rows,
        "average_all_feasible",
        metric="auprc",
        direction="higher",
    )
    assert matched == {
        ("elliptic", 1): pytest.approx(-0.05),
        ("elliptic", 2): pytest.approx(-0.05),
    }
    independent_minima = min((0.9, 0.8)) - min((0.7, 0.85))
    assert independent_minima > 0


def test_regret_worst_case_is_formed_within_seed_after_contract_regret() -> None:
    rows = []
    for seed in (1, 2):
        for contract, value in (("a", 0.1), ("b", 0.4)):
            rows.append(
                {
                    "dataset": "dgraphfin",
                    "target_contract": contract,
                    "expert_prediction_seed": seed,
                    "fold": "fold0",
                    "method": "full_corerouter",
                    "metric": "contract_regret",
                    "value": value,
                }
            )
    derived = gate._derive_regret_rows(rows)
    maximum = [
        row
        for row in derived
        if row["metric"] == "maximum_regret"
    ]
    assert [row["value"] for row in maximum] == [0.4, 0.4]
    assert all(row["expert_prediction_seed"] in {1, 2} for row in maximum)


def _passing_frozen_result() -> tuple[dict[str, object], dict[str, object]]:
    schema = json.loads(
        FROZEN_GATE_SPEC.read_text(encoding="utf-8")
    )
    methods = [
        "full_corerouter",
        *[f"expert:{name}" for name in schema["required_experts"]],
        *schema["required_strong_baselines"],
        *[f"ablation:{name}" for name in schema["required_ablations"]],
    ]
    rows = []
    for dataset in schema["required_datasets"]:
        for contract in schema["required_target_contracts"]:
            for seed in schema["required_expert_prediction_seeds"]:
                for method in methods:
                    is_full = method == "full_corerouter"
                    for metric in schema["required_contract_metrics"]:
                        lower = metric in {
                            "contract_regret",
                            "selective_risk",
                            "aurc",
                            "abstention_cost",
                            "compute",
                        }
                        value = 0.1 if is_full and lower else 0.8 if is_full else 0.3 if lower else 0.6
                        if metric == "coverage":
                            value = 0.8
                        rows.append(
                            {
                                "dataset": dataset,
                                "target_contract": contract,
                                "seed": seed,
                                "expert_prediction_seed": seed,
                                "router_training_seed": gate.derive_router_seed(
                                    seed,
                                    method,
                                ),
                                "fold": "fold0",
                                "method": method,
                                "metric": metric,
                                "value": value,
                                "execution_status": "EXECUTABLE",
                            }
                        )
    routing = [
        {
            "dataset": dataset,
            "target_contract": contract,
            "expert_prediction_seed": seed,
            "router_training_seed": gate.derive_router_seed(
                seed,
                "full_corerouter",
            ),
            "fold": "fold0",
            "distinct_experts": 2,
            "perturbation_flip_rate": 0.01,
        }
        for dataset in schema["required_datasets"]
        for contract in schema["required_target_contracts"]
        for seed in schema["required_expert_prediction_seeds"]
    ]
    result = {
        "rows": rows,
        "routing": routing,
        "target_label_selection": False,
        "oracle_target_selection": False,
        "headline_oracle": "contract_feasible_oracle",
        "diagnostic_oracle": "instance_clairvoyant_oracle_ceiling",
    }
    return result, schema


def test_frozen_go_no_go_uses_effects_correction_and_all_criteria() -> None:
    result, schema = _passing_frozen_result()
    report = gate.evaluate_pilot_gate(result, schema)
    assert report["passed"]
    assert all(report["criteria"].values())

    weakened = copy.deepcopy(result)
    full_budget = {
        (
            row["dataset"],
            row["target_contract"],
            row["expert_prediction_seed"],
        ): row["value"]
        for row in weakened["rows"]
        if row["method"] == "full_corerouter"
        and row["metric"] == "budget_curve_area"
    }
    for row in weakened["rows"]:
        if (
            row["method"] == "ablation:no_budget"
            and row["metric"] == "budget_curve_area"
        ):
            row["value"] = full_budget[
                (
                    row["dataset"],
                    row["target_contract"],
                    row["expert_prediction_seed"],
                )
            ]
    failed = gate.evaluate_pilot_gate(weakened, schema)
    assert not failed["passed"]
    assert not failed["criteria"]["meaningful_required_ablations"]
    no_budget = next(
        record
        for record in failed["ablation_effects"]
        if record["ablation"] == "no_budget"
    )
    assert not no_budget["holm_reject"]
    assert not no_budget["meaningful"]


def test_gate_cli_blocks_when_no_pilot_result_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_pilot_gate.py",
            "--pilot-result",
            str(tmp_path / "absent.json"),
            "--schema",
            str(FROZEN_GATE_SPEC),
            "--output",
            str(output),
        ],
    )
    assert gate.main() == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED_MISSING_PILOT_RESULT"
    assert report["passed"] is False
