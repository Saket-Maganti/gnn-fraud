#!/usr/bin/env python3
"""Generate frozen future-run matrices without executing experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "configs/coregraph/run_matrices"
FIELDS = (
    "run_key",
    "campaign",
    "analysis_family",
    "dataset",
    "task",
    "source_contract",
    "target_contract",
    "method",
    "objective",
    "seed",
    "access_regime",
    "hardware",
    "resource_class",
    "runtime_status",
    "prerequisite",
    "execution_status",
)

DATASETS = {
    "elliptic_v2": ("node_classification", "fraud"),
    "dgraphfin_v2": ("transaction_classification", "fraud"),
    "tfinance_v2": ("node_classification", "fraud"),
    "ellipticpp_v2": ("edge_classification", "fraud"),
    "ibm_aml_small": ("transaction_classification", "fraud"),
    "ibm_aml_medium": ("transaction_classification", "fraud"),
    "good_cmnist_covariate": ("graph_classification", "good"),
    "good_hiv_scaffold": ("graph_classification", "good"),
    "good_arxiv_time": ("node_classification", "good"),
}
CONTRACT_PAIRS = (
    ("historical_transductive_full", "future_inductive_full"),
    ("historical_transductive_full", "future_inductive_recent"),
    ("historical_inductive_full", "future_missing_graph"),
    ("historical_inductive_full", "future_budget_topk"),
    ("historical_transductive_full", "future_resource_limited"),
)
METHODS = (
    "best_single_feasible",
    "equal_mixture",
    "graphsafe_feature",
    "fraudshift_current_gate",
    "coregraph_atomic",
    "coregraph_no_contract",
    "coregraph_factorised",
    "coregraph_factorised_diagnostics",
)
EXTERNAL = {
    "mowst_official": ("PENDING_INTEGRATION", "official_adapter_parity"),
    "graphmetro_official": ("BLOCKED_LICENSE", "verified_reuse_licence"),
    "ciga_official": ("PENDING_INTEGRATION", "task_parity_and_adapter"),
    "tgn_official": ("PENDING_INTEGRATION", "official_adapter_parity"),
}


def key(values: tuple[object, ...]) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def row(
    campaign: str,
    family: str,
    dataset: str,
    source: str,
    target: str,
    method: str,
    seed: int,
    *,
    objective: str = "composite",
    hardware: str = "single_t4",
    status: str = "PLANNED",
    prerequisite: str = "local_and_data_gates",
) -> dict[str, object]:
    task, _ = DATASETS[dataset]
    values = (campaign, family, dataset, task, source, target, method, objective, seed)
    return {
        "run_key": key(values),
        "campaign": campaign,
        "analysis_family": family,
        "dataset": dataset,
        "task": task,
        "source_contract": source,
        "target_contract": target,
        "method": method,
        "objective": objective,
        "seed": seed,
        "access_regime": "DG_NO_TARGET",
        "hardware": hardware,
        "resource_class": "T4_16GB" if "t4" in hardware else hardware.upper(),
        "runtime_status": "TBD_PROFILE",
        "prerequisite": prerequisite,
        "execution_status": status,
    }


def write(name: str, rows: list[dict[str, object]]) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    screening = [
        row("screening_5seed", "method_screening", dataset, source, target, method, seed)
        for dataset, (task, _) in DATASETS.items()
        if task != "graph_classification"
        for source, target in CONTRACT_PAIRS
        for method in METHODS
        for seed in range(5)
    ]
    final = [
        row("final_10seed", "confirmatory_primary", dataset, source, target, method, seed)
        for dataset, (task, _) in DATASETS.items()
        if task != "graph_classification"
        for source, target in CONTRACT_PAIRS
        for method in (
            "best_single_feasible",
            "graphsafe_feature",
            "coregraph_factorised_diagnostics",
        )
        for seed in range(10)
    ]
    ablations = [
        row(
            "ablation",
            "router_ablation",
            dataset,
            "historical_transductive_full",
            "future_inductive_recent",
            method,
            seed,
        )
        for dataset in ("elliptic_v2", "dgraphfin_v2", "ibm_aml_small")
        for method in (
            "coregraph_no_pairwise",
            "coregraph_no_diagnostics",
            "coregraph_no_resource_mask",
            "coregraph_no_abstention",
            "coregraph_linear",
            "coregraph_mlp",
            "coregraph_attention",
        )
        for seed in range(5)
    ]
    synthetic = [
        row(
            "synthetic_theory",
            "mechanism_validation",
            "elliptic_v2",
            "synthetic_source",
            f"synthetic_{mechanism}",
            method,
            seed,
            hardware="cpu",
            prerequisite="local_gates",
        )
        for mechanism, method, seed in product(
            (
                "graph_best",
                "feature_best",
                "ordering_crosses",
                "fixed_mixture_regret",
                "factorised_generalisation",
                "interaction_breaks_factorisation",
                "resource_mask",
                "budget_changes_expert",
            ),
            ("fixed_mixture", "coregraph_factorised_diagnostics"),
            range(10),
        )
    ]
    good = [
        row(
            "good",
            "graph_ood_external_validity",
            dataset,
            "good_source",
            "good_target",
            method,
            seed,
            status=("PENDING_INTEGRATION" if method == "ciga_official" else "PLANNED"),
            prerequisite=("task_parity_and_adapter" if method == "ciga_official" else "good_data"),
        )
        for dataset in ("good_cmnist_covariate", "good_hiv_scaffold", "good_arxiv_time")
        for method in ("erm", "groupdro", "vrex", "coregraph_factorised", "ciga_official")
        for seed in range(5)
    ]
    fraud = list(screening)
    for dataset in ("elliptic_v2", "dgraphfin_v2", "tfinance_v2"):
        for source, target in CONTRACT_PAIRS[:2]:
            for method, (status, prerequisite) in EXTERNAL.items():
                for seed in range(5):
                    fraud.append(
                        row(
                            "fraud",
                            "official_baseline",
                            dataset,
                            source,
                            target,
                            method,
                            seed,
                            status=status,
                            prerequisite=prerequisite,
                        )
                    )
    resource = [
        row(
            "resource",
            "budget_resource",
            dataset,
            "historical_transductive_full",
            target,
            "coregraph_factorised_diagnostics",
            seed,
            objective="budget_composite",
            hardware=hardware,
            prerequisite="hardware_profile",
        )
        for dataset, target, hardware, seed in product(
            ("elliptic_v2", "dgraphfin_v2", "ibm_aml_medium"),
            ("future_budget_topk", "future_resource_limited", "future_latency_cap"),
            ("cpu", "single_t4", "dual_t4"),
            range(5),
        )
    ]
    master = screening + good + synthetic + resource
    outputs = {
        "MASTER_EXPERIMENT_MATRIX.csv": master,
        "SCREENING_5SEED_GRID.csv": screening,
        "FINAL_10SEED_GRID.csv": final,
        "ABLATION_GRID.csv": ablations,
        "THEORY_SYNTHETIC_GRID.csv": synthetic,
        "GOOD_GRID.csv": good,
        "FRAUD_GRID.csv": fraud,
        "RESOURCE_GRID.csv": resource,
    }
    for name, rows in outputs.items():
        write(name, rows)
    manifest = {
        name: {
            "rows": len(rows),
            "sha256": hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest(),
        }
        for name, rows in outputs.items()
    }
    (OUTPUT / "MATRIX_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
