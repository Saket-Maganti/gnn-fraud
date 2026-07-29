#!/usr/bin/env python3
"""Recompute TKDE analysis surfaces from locked, saved evidence only.

No model is trained.  The script reads canonical result rows, prediction-backed
derived policy tables, and raw dataset metadata already present in the
repository.  It writes deterministic CSV/Markdown analysis inputs and a
scalar-level number provenance map.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tkde_rebuild"
SCRIPT = "scripts/tkde_rebuild/compute_analysis.py"
BOOTSTRAPS = 10_000
RNG_SEED = 20260710

PROVENANCE_FIELDS = [
    "number_id",
    "manuscript_use",
    "artifact",
    "row_key",
    "field",
    "value",
    "units",
    "source_files",
    "aggregation_script",
    "filters",
    "seed_set",
    "aggregation",
    "output_location",
    "evidence_lock",
    "claim_ids",
]


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_ci(values: Iterable[float], salt: int = 0) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1 or np.all(arr == arr[0]):
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(RNG_SEED + salt)
    means = rng.choice(arr, size=(BOOTSTRAPS, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def wilcoxon_p(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0 or np.allclose(arr, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(arr, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(values: Iterable[float], salt: int = 0) -> dict[str, float | int]:
    arr = np.asarray([float(v) for v in values if pd.notna(v)], dtype=float)
    lo, hi = bootstrap_ci(arr, salt=salt)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else math.nan,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else math.nan,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def write_df(df: pd.DataFrame, name: str, description: str) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    compact = df.copy()
    if len(compact) > 250:
        compact = compact.head(250)
        suffix = f"\n\n_Showing 250 of {len(df)} rows; the CSV is complete._\n"
    else:
        suffix = "\n"
    try:
        table = compact.to_markdown(index=False)
    except Exception:
        table = compact.to_csv(index=False)
    (OUT / f"{name}.md").write_text(f"# {name.replace('_', ' ').title()}\n\n{description}\n\n{table}{suffix}", encoding="utf-8")
    return path


def dataset_statistics() -> pd.DataFrame:
    features = pd.read_csv(ROOT / "data/raw/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(ROOT / "data/raw/elliptic_txs_classes.csv")
    edges = pd.read_csv(ROOT / "data/raw/elliptic_txs_edgelist.csv")
    rb09 = pd.read_csv(ROOT / "results/runs_rb09v3/runs.csv")
    rows: list[dict[str, Any]] = []
    ell = rb09[rb09.dataset.eq("elliptic")].iloc[0]
    ell_class = classes["class"].astype(str).value_counts()
    rows.append(
        {
            "dataset": "Elliptic",
            "variant": "canonical",
            "origin": "real pseudonymized Bitcoin transactions",
            "prediction_unit": "transaction node",
            "nodes_or_accounts": int(features.shape[0]),
            "source_edges_or_transactions": int(edges.shape[0]),
            "message_passing_arcs": 468710,
            "node_feature_dim": int(features.shape[1] - 2),
            "edge_feature_dim": 0,
            "time_definition": "49 discrete time steps",
            "protocol": "strict/isolated/transductive; train t1-30, val t31-34, test t35-49",
            "train_units": int(ell.train_nodes),
            "validation_units": int(ell.val_nodes),
            "test_units": int(ell.test_nodes),
            "train_positives": int(ell.positives_train),
            "validation_positives": int(ell.positives_val),
            "test_positives": int(ell.positives_test),
            "test_positive_rate": float(ell.positives_test / ell.test_nodes),
            "total_labeled": int(ell_class.get("1", 0) + ell_class.get("2", 0)),
            "total_positives": int(ell_class.get("1", 0)),
            "label_mapping": "raw 1=illicit->positive 1; raw 2=licit->supervised 0 (repository code 2); unknown excluded",
            "source": "data/raw/elliptic_txs_*.csv; results/runs_rb09v3/runs.csv",
        }
    )

    archive = np.load(ROOT / "data/raw/dgraphfin.npz")
    y = archive["y"]
    dgr = rb09[rb09.dataset.eq("dgraphfin")].iloc[0]
    rows.append(
        {
            "dataset": "DGraphFin",
            "variant": "canonical",
            "origin": "real anonymized financial user graph",
            "prediction_unit": "account/user node",
            "nodes_or_accounts": int(archive["x"].shape[0]),
            "source_edges_or_transactions": int(archive["edge_index"].shape[1]),
            "message_passing_arcs": 7994520,
            "node_feature_dim": int(archive["x"].shape[1]),
            "edge_feature_dim": 0,
            "time_definition": "20 equal-count incident-edge timestamp buckets",
            "protocol": "strict/isolated/transductive; train t1-14, val t15-16, test t17-20",
            "train_units": int(dgr.train_nodes),
            "validation_units": int(dgr.val_nodes),
            "test_units": int(dgr.test_nodes),
            "train_positives": int(dgr.positives_train),
            "validation_positives": int(dgr.positives_val),
            "test_positives": int(dgr.positives_test),
            "test_positive_rate": float(dgr.positives_test / dgr.test_nodes),
            "total_labeled": int(np.isin(y, [0, 1]).sum()),
            "total_positives": int((y == 1).sum()),
            "label_mapping": "raw 1=fraud->positive 1; raw 0=normal->supervised 0 (repository code 2); raw 2/3 background excluded",
            "source": "data/raw/dgraphfin.npz; results/runs_rb09v3/runs.csv",
        }
    )

    ibm = collect_ibm_rows()
    for (variant, protocol), group in ibm.groupby(["variant", "protocol"], sort=True):
        sample = load_json(ROOT / group.iloc[0].source_path)
        split = sample["split_stats"]
        graph = sample["graph_stats"]
        rows.append(
            {
                "dataset": "IBM AML-Data",
                "variant": variant,
                "origin": "synthetic financial transactions",
                "prediction_unit": "transaction edge",
                "nodes_or_accounts": int(graph["num_accounts"]),
                "source_edges_or_transactions": int(graph["num_transactions"]),
                "message_passing_arcs": int(graph["num_transactions"]),
                "node_feature_dim": 8,
                "edge_feature_dim": int(graph["edge_feature_dim"]),
                "time_definition": "continuous transaction timestamps; stable chronological sort",
                "protocol": f"{protocol}: " + ("50/20/30%" if protocol == "early_to_late_transfer" else "60/20/20%"),
                "train_units": int(split["train"]["edge_count"]),
                "validation_units": int(split["val"]["edge_count"]),
                "test_units": int(split["test"]["edge_count"]),
                "train_positives": int(split["train"]["positive_count"]),
                "validation_positives": int(split["val"]["positive_count"]),
                "test_positives": int(split["test"]["positive_count"]),
                "test_positive_rate": float(split["test"]["positive_rate"]),
                "total_labeled": int(graph["num_transactions"]),
                "total_positives": int(sum(split[name]["positive_count"] for name in ["train", "val", "test"])),
                "label_mapping": "is_laundering=1 positive; 0 negative; no unknown class in materialized task",
                "source": str(group.iloc[0].source_path),
            }
        )
    return pd.DataFrame(rows)


def rb09_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(ROOT / "results/runs_rb09v3/runs.csv")
    data = data[data.protocol.isin(["strict_inductive", "inductive_isolated"])].copy()
    rows: list[dict[str, Any]] = []
    metrics = ["auprc", "auroc", "f1"]
    for (dataset, model), group in data.groupby(["dataset", "model"], sort=True):
        for metric_index, metric in enumerate(metrics):
            pivot = group.pivot(index="seed", columns="protocol", values=metric).dropna()
            strict = pivot["strict_inductive"].to_numpy(dtype=float)
            isolated = pivot["inductive_isolated"].to_numpy(dtype=float)
            delta = isolated - strict
            strict_ci = bootstrap_ci(strict, salt=100 + len(rows))
            isolated_ci = bootstrap_ci(isolated, salt=200 + len(rows))
            delta_ci = bootstrap_ci(delta, salt=300 + len(rows))
            delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "n_pairs": len(delta),
                    "strict_mean": strict.mean(),
                    "strict_std": strict.std(ddof=1),
                    "strict_ci95_low": strict_ci[0],
                    "strict_ci95_high": strict_ci[1],
                    "isolated_mean": isolated.mean(),
                    "isolated_std": isolated.std(ddof=1),
                    "isolated_ci95_low": isolated_ci[0],
                    "isolated_ci95_high": isolated_ci[1],
                    "delta_isolated_minus_strict": delta.mean(),
                    "delta_ci95_low": delta_ci[0],
                    "delta_ci95_high": delta_ci[1],
                    "relative_delta_pct": 100.0 * delta.mean() / strict.mean() if strict.mean() else math.nan,
                    "cohen_dz": delta.mean() / delta_std if delta_std > 0 else 0.0,
                    "wilcoxon_p": wilcoxon_p(delta),
                    "source": "results/runs_rb09v3/runs.csv",
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p_within_metric"] = np.nan
    for metric, idx in result.groupby("metric").groups.items():
        result.loc[idx, "holm_p_within_metric"] = holm_adjust(result.loc[idx, "wilcoxon_p"])
    main = result[result.metric.eq("auprc")].copy().reset_index(drop=True)
    main["strict_summary"] = main.apply(lambda x: f"{x.strict_mean:.4f} ± {x.strict_std:.4f}", axis=1)
    main["isolated_summary"] = main.apply(lambda x: f"{x.isolated_mean:.4f} ± {x.isolated_std:.4f}", axis=1)
    return result, main


def v24_duplicate_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in sorted((ROOT / "results/v24_imported").glob("rb41_*/RESULT_INDEX.csv"))]
    data = pd.concat(frames, ignore_index=True)
    metric_cols = ["f1", "auprc", "auroc", "recall_at_1pct", "precision_at_1pct", "recall", "precision"]
    rows = []
    representatives = []
    keys = ["dataset", "protocol", "model", "seed"]
    for key, group in data.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(
            {
                "n_stress_labels": group.stress_config.nunique(),
                "stress_labels": ";".join(sorted(group.stress_config.unique())),
                "all_performance_metrics_identical": all(group[col].nunique(dropna=False) == 1 for col in metric_cols),
                "runtime_values_identical": group.runtime_seconds.nunique(dropna=False) == 1,
                "runner_passes_stress_to_harness": False,
                "scientific_status": "DUPLICATE_METADATA_LABELS_NOT_DISTINCT_CONDITIONS",
                "source": ";".join(sorted(set(group.run_id))),
            }
        )
        rows.append(row)
        representative = group.sort_values("stress_config").iloc[0].to_dict()
        representative["stress_config"] = "deduplicated_v24_rerun_contract"
        representative["deduplication_note"] = "one of three metric-identical label copies retained; stress labels were not passed to harness"
        representatives.append(representative)
    return pd.DataFrame(rows), pd.DataFrame(representatives)


def collect_ibm_rows() -> pd.DataFrame:
    paths: list[tuple[str, Path]] = []
    paths.extend(("V26", path) for path in sorted((ROOT / "results/v26_imported/ibm_aml").glob("**/json/*.json")))
    paths.extend(("V27", path) for path in sorted((ROOT / "results/v27_imported/imported_json").glob("*.json")))
    paths.extend(("V28", path) for path in sorted((ROOT / "results/v28_imported/imported_json").glob("*.json")))
    rows = []
    for version, path in paths:
        payload = load_json(path)
        metrics = payload["metrics"]
        test = payload["split_stats"]["test"]
        config = payload.get("v28_config") or payload["model"]
        rows.append(
            {
                "version": version,
                "variant": payload["variant"],
                "size": payload["size"],
                "regime": payload["illicit_ratio_group"],
                "protocol": payload["protocol"],
                "config": config,
                "seed": int(payload["seed"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "positive_rate": float(test["positive_rate"]),
                "positive_count": int(test["positive_count"]),
                "test_count": int(test["edge_count"]),
                "actual_backend": payload.get("actual_backend", ""),
                "source_path": rel(path),
            }
        )
    frame = pd.DataFrame(rows)
    expected = {"V26": 240, "V27": 80, "V28": 520}
    actual = frame.version.value_counts().to_dict()
    if actual != expected:
        raise RuntimeError(f"IBM evidence counts changed: {actual} != {expected}")
    if not frame.groupby(["version", "variant", "protocol", "config"]).seed.nunique().eq(10).all():
        raise RuntimeError("IBM full10 grouping failed")
    return frame


def ibm_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["auprc", "auroc", "f1", "precision", "recall", "balanced_accuracy", "runtime_seconds"]
    for key, group in data.groupby(["version", "variant", "size", "regime", "protocol", "config"], sort=True):
        base = dict(zip(["version", "variant", "size", "regime", "protocol", "config"], key))
        base.update({"n": len(group), "positive_rate_mean": group.positive_rate.mean(), "source_paths": ";".join(sorted(group.source_path))})
        for offset, metric in enumerate(metrics):
            summary = summarize(group[metric], salt=len(rows) * 17 + offset)
            base[f"{metric}_mean"] = summary["mean"]
            base[f"{metric}_std"] = summary["std"]
            base[f"{metric}_ci95_low"] = summary["ci95_low"]
            base[f"{metric}_ci95_high"] = summary["ci95_high"]
        base["normalized_auprc"] = base["auprc_mean"] / base["positive_rate_mean"]
        base["auprc_lift_above_prevalence"] = base["auprc_mean"] - base["positive_rate_mean"]
        rows.append(base)
    return pd.DataFrame(rows)


def graph_grid(data: pd.DataFrame) -> pd.DataFrame:
    ref = data[data.version.eq("V27")].copy()
    v28 = data[data.version.eq("V28") & ~data.config.eq("account_account_sender_receiver")].copy()
    return pd.concat([ref, v28], ignore_index=True)


def rank_divergence(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = data[data.version.eq("V26")].copy()
    baseline["family"] = "baseline_grid"
    graph = graph_grid(data)
    graph["family"] = "graph_grid"
    combined = pd.concat([baseline, graph], ignore_index=True)
    means = combined.groupby(["family", "variant", "protocol", "config"], as_index=False)[["auprc", "auroc", "f1"]].mean()
    rank_rows = []
    cell_rows = []
    for (family, variant, protocol), group in means.groupby(["family", "variant", "protocol"], sort=True):
        group = group.copy()
        for metric in ["auprc", "auroc", "f1"]:
            group[f"{metric}_rank"] = group[metric].rank(method="min", ascending=False)
        for _, row in group.iterrows():
            rank_rows.append(row.to_dict())
        rho_pr_f1, p_pr_f1 = stats.spearmanr(group.auprc_rank, group.f1_rank)
        rho_roc_pr, p_roc_pr = stats.spearmanr(group.auroc_rank, group.auprc_rank)
        winners = {metric: group.sort_values([metric, "config"], ascending=[False, True]).iloc[0].config for metric in ["auprc", "auroc", "f1"]}
        cell_rows.append(
            {
                "family": family,
                "variant": variant,
                "protocol": protocol,
                "n_configurations": len(group),
                "auprc_winner": winners["auprc"],
                "auroc_winner": winners["auroc"],
                "f1_winner": winners["f1"],
                "auprc_f1_winner_disagree": winners["auprc"] != winners["f1"],
                "auroc_auprc_winner_disagree": winners["auroc"] != winners["auprc"],
                "spearman_auprc_vs_f1": float(rho_pr_f1),
                "spearman_auprc_vs_f1_p": float(p_pr_f1),
                "spearman_auroc_vs_auprc": float(rho_roc_pr),
                "spearman_auroc_vs_auprc_p": float(p_roc_pr),
            }
        )
    return pd.DataFrame(rank_rows), pd.DataFrame(cell_rows)


def matched_ablation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare V28 candidates with the V27 reference on a fixed context grid.

    The four variant/protocol contexts are fixed benchmark conditions, not 40
    exchangeable experimental units.  We therefore average the four matched
    context differences within each seed before bootstrap and Wilcoxon
    inference (n=10 seed blocks).  A second surface preserves every
    context-specific ten-seed effect so heterogeneity is not hidden by that
    fixed-grid average.
    """
    ref = data[data.version.eq("V27")].copy()
    candidates = data[data.version.eq("V28")].copy()
    candidate_names = [
        "account_account_sender_receiver",
        "degree_capped_bipartite",
        "edge_aware_graphsage_h64_degree_only",
        "edge_aware_graphsage_h64_no_edge_features",
        "edge_aware_graphsage_h64_shuffled_edge_features",
        "gine_light_h64",
        "recent_window_only_graph",
    ]
    metrics = ["auprc", "auroc", "f1", "runtime_seconds"]
    rows = []
    context_rows = []
    keys = ["variant", "protocol", "seed"]
    ref_cols = keys + metrics
    for config in candidate_names:
        cand = candidates[candidates.config.eq(config)]
        merged = cand[keys + ["size"] + metrics + ["source_path"]].merge(
            ref[ref_cols + ["source_path"]], on=keys, suffixes=("_candidate", "_reference")
        )
        for size, sized in merged.groupby("size", sort=True):
            for metric in metrics:
                sized = sized.copy()
                sized["delta"] = sized[f"{metric}_candidate"] - sized[f"{metric}_reference"]
                seed_blocks = sized.groupby("seed", sort=True)["delta"].mean()
                lo, hi = bootstrap_ci(seed_blocks, salt=7000 + len(rows))
                std = float(seed_blocks.std(ddof=1)) if len(seed_blocks) > 1 else 0.0
                context_deltas = []
                for (variant, protocol), context in sized.groupby(["variant", "protocol"], sort=True):
                    delta = context["delta"]
                    context_lo, context_hi = bootstrap_ci(delta, salt=9000 + len(context_rows))
                    context_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
                    context_delta = float(delta.mean())
                    context_deltas.append(context_delta)
                    context_rows.append(
                        {
                            "config": config,
                            "size": size,
                            "metric": metric,
                            "variant": variant,
                            "protocol": protocol,
                            "n_seed_pairs": len(delta),
                            "reference_mean": context[f"{metric}_reference"].mean(),
                            "candidate_mean": context[f"{metric}_candidate"].mean(),
                            "mean_delta": context_delta,
                            "delta_ci95_low": context_lo,
                            "delta_ci95_high": context_hi,
                            "cohen_dz": context_delta / context_std if context_std > 0 else 0.0,
                            "wilcoxon_p_descriptive": wilcoxon_p(delta),
                            "source_paths": ";".join(sorted(set(context.source_path_candidate) | set(context.source_path_reference))),
                        }
                    )
                positive = sum(value > 1e-15 for value in context_deltas)
                negative = sum(value < -1e-15 for value in context_deltas)
                zero = len(context_deltas) - positive - negative
                rows.append(
                    {
                        "config": config,
                        "size": size,
                        "metric": metric,
                        "n_pairs": len(seed_blocks),
                        "n_raw_context_seed_pairs": len(sized),
                        "n_variant_protocol_cells": sized[["variant", "protocol"]].drop_duplicates().shape[0],
                        "reference_mean": sized[f"{metric}_reference"].mean(),
                        "candidate_mean": sized[f"{metric}_candidate"].mean(),
                        "mean_delta": seed_blocks.mean(),
                        "delta_ci95_low": lo,
                        "delta_ci95_high": hi,
                        "relative_delta_pct": 100 * seed_blocks.mean() / sized[f"{metric}_reference"].mean() if sized[f"{metric}_reference"].mean() else math.nan,
                        "cohen_dz": seed_blocks.mean() / std if std > 0 else 0.0,
                        "wilcoxon_p": wilcoxon_p(seed_blocks),
                        "context_delta_min": min(context_deltas),
                        "context_delta_max": max(context_deltas),
                        "n_contexts_positive": positive,
                        "n_contexts_negative": negative,
                        "n_contexts_zero": zero,
                        "all_nonzero_contexts_same_sign": positive == 0 or negative == 0,
                        "source_paths": ";".join(sorted(set(sized.source_path_candidate) | set(sized.source_path_reference))),
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_p_within_size_metric"] = np.nan
    for _, idx in result.groupby(["size", "metric"]).groups.items():
        result.loc[idx, "holm_p_within_size_metric"] = holm_adjust(result.loc[idx, "wilcoxon_p"])
    return result, pd.DataFrame(context_rows)


def runtime_feasibility(data: pd.DataFrame) -> pd.DataFrame:
    graph = graph_grid(data)
    rows = []
    for key, group in graph.groupby(["variant", "size", "regime", "protocol", "config"], sort=True):
        variant, size, regime, protocol, config = key
        rows.append(
            {
                "variant": variant,
                "size": size,
                "regime": regime,
                "protocol": protocol,
                "config": config,
                "status": "MEASURED_FULL10",
                "n": len(group),
                "auprc_mean": group.auprc.mean(),
                "auprc_std": group.auprc.std(ddof=1),
                "normalized_auprc_mean": (group.auprc / group.positive_rate).mean(),
                "auroc_mean": group.auroc.mean(),
                "f1_mean": group.f1.mean(),
                "runtime_seconds_mean": group.runtime_seconds.mean(),
                "runtime_seconds_min": group.runtime_seconds.min(),
                "runtime_seconds_max": group.runtime_seconds.max(),
                "source_paths": ";".join(sorted(group.source_path)),
            }
        )
    result = pd.DataFrame(rows)
    result["pareto_auprc_runtime_within_cell"] = False
    for _, idx in result.groupby(["variant", "protocol"]).groups.items():
        group = result.loc[idx]
        for index, row in group.iterrows():
            dominated = ((group.runtime_seconds_mean <= row.runtime_seconds_mean) & (group.auprc_mean >= row.auprc_mean) & ((group.runtime_seconds_mean < row.runtime_seconds_mean) | (group.auprc_mean > row.auprc_mean))).any()
            result.loc[index, "pareto_auprc_runtime_within_cell"] = not bool(dominated)
    blocked = pd.DataFrame(
        [
            {
                "variant": variant,
                "size": "medium",
                "regime": variant.split("-")[0],
                "protocol": "both planned protocols",
                "config": "gine_light_h64",
                "status": "RESOURCE_BLOCKED_T4_CUDA_OOM",
                "n": 0,
                "auprc_mean": math.nan,
                "auprc_std": math.nan,
                "normalized_auprc_mean": math.nan,
                "auroc_mean": math.nan,
                "f1_mean": math.nan,
                "runtime_seconds_mean": math.nan,
                "runtime_seconds_min": math.nan,
                "runtime_seconds_max": math.nan,
                "source_paths": "results/v28_imported/V28_ALL_RUNS_EVIDENCE_LOCK.json",
                "pareto_auprc_runtime_within_cell": False,
            }
            for variant in ["hi-medium", "li-medium"]
        ]
    )
    return pd.concat([result, blocked], ignore_index=True)


def graphsafe_surfaces() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(ROOT / "results/runs_rb17_review_budget_worst_block/rb17_results.csv")
    methods = ["feature_only", "graph_only", "best_val_branch", "simple_average", "graphsafe_conservative"]
    data = data[data.method.isin(methods)].copy()
    metrics = ["f1", "auprc", "recall_at_1pct", "cost_sensitive_risk", "worst_block_regret", "average_block_regret"]
    summaries = []
    for (dataset, method), group in data.groupby(["dataset", "method"], sort=True):
        seed_blocks = group.groupby("seed", as_index=False)[metrics].mean()
        row = {"dataset": dataset, "method": method, "n": len(seed_blocks), "n_context_rows": len(group)}
        for offset, metric in enumerate(metrics):
            s = summarize(seed_blocks[metric], salt=9000 + len(summaries) * 13 + offset)
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
            row[f"{metric}_ci95_low"] = s["ci95_low"]
            row[f"{metric}_ci95_high"] = s["ci95_high"]
        row["source"] = "results/runs_rb17_review_budget_worst_block/rb17_results.csv"
        summaries.append(row)
    summary = pd.DataFrame(summaries)

    paired_rows = []
    block_keys = ["dataset", "seed"]
    base = data[data.method.eq("simple_average")].groupby(block_keys, as_index=False)[metrics].mean()
    for method in [m for m in methods if m != "simple_average"]:
        method_blocks = data[data.method.eq(method)].groupby(block_keys, as_index=False)[metrics].mean()
        merged = method_blocks.merge(base, on=block_keys, suffixes=("_method", "_average"))
        for dataset, group in merged.groupby("dataset", sort=True):
            for metric in metrics:
                direction = -1.0 if metric in {"cost_sensitive_risk", "worst_block_regret", "average_block_regret"} else 1.0
                raw_delta = group[f"{metric}_method"] - group[f"{metric}_average"]
                improvement = direction * raw_delta
                lo, hi = bootstrap_ci(improvement, salt=11000 + len(paired_rows))
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "comparator": "simple_average",
                        "metric": metric,
                        "direction": "lower" if direction < 0 else "higher",
                        "n_pairs": len(group),
                        "mean_improvement": improvement.mean(),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "wilcoxon_p": wilcoxon_p(improvement),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired["holm_p_all_graphsafe_vs_average"] = holm_adjust(paired.wilcoxon_p)

    budget_rows = []
    for (dataset, method), group in data.groupby(["dataset", "method"], sort=True):
        for budget, pcol, rcol in [
            (0.5, "precision_at_0_5pct", "recall_at_0_5pct"),
            (1.0, "precision_at_1pct", "recall_at_1pct"),
            (2.0, "precision_at_2pct", "recall_at_2pct"),
        ]:
            blocks = group.groupby("seed", as_index=False)[[pcol, rcol]].mean()
            budget_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "budget_pct": budget,
                    "n": len(blocks),
                    "n_context_rows": len(group),
                    "precision_mean": blocks[pcol].mean(),
                    "precision_std": blocks[pcol].std(ddof=1),
                    "recall_mean": blocks[rcol].mean(),
                    "recall_std": blocks[rcol].std(ddof=1),
                    "source": "results/runs_rb17_review_budget_worst_block/rb17_results.csv",
                }
            )
    budgets = pd.DataFrame(budget_rows)

    cal = pd.read_csv(ROOT / "results/runs_rb16_graphsafe_best_branch_strengthening/rb16_results.csv")
    cal = cal[cal.method.isin(methods)].copy()
    cal_rows = []
    for (dataset, method), group in cal.groupby(["dataset", "method"], sort=True):
        blocks = group.groupby("seed", as_index=False)[["ece", "brier"]].mean()
        cal_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(blocks),
                "n_context_rows": len(group),
                "ece_mean": blocks.ece.mean(),
                "ece_std": blocks.ece.std(ddof=1),
                "brier_mean": blocks.brier.mean(),
                "brier_std": blocks.brier.std(ddof=1),
                "source": "results/runs_rb16_graphsafe_best_branch_strengthening/rb16_results.csv",
            }
        )
    calibration = pd.DataFrame(cal_rows)
    return summary, paired, budgets, calibration


def static_method_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    axes = pd.DataFrame(
        [
            ["T", "temporal partition/order", "train/validation/test windows and ordering", "prevents future-label/time leakage and fixes horizon"],
            ["V", "graph visibility/access", "which nodes/edges are available at train and inference", "separates full, train-subgraph, and isolated access"],
            ["C", "graph construction", "entities, edges, attributes, caps, and recency window", "fixes the information and inductive bias supplied to a model"],
            ["S", "selection cleanliness", "validation data, metric, threshold, and policy selection", "prevents test-driven model or policy choice"],
            ["B", "decision budget", "threshold, top-K or coverage, and explicit costs", "distinguishes ranking from operational decisions"],
            ["R", "resource envelope", "hardware, memory/disk guard, runtime, and failure status", "separates measured feasibility from predictive performance"],
        ],
        columns=["symbol", "axis", "contract_contents", "why_it_matters"],
    )
    protocols = pd.DataFrame(
        [
            ["strict-inductive", "chronological", "training nodes/edges only during training; held-out structure available at evaluation", "test labels hidden", "validation F1", "Elliptic, DGraphFin"],
            ["isolated-inductive", "same chronological masks", "held-out nodes isolated from training-period cross-split edges at evaluation", "test labels hidden", "validation F1", "Elliptic, DGraphFin"],
            ["transductive", "same masks", "full graph structure visible", "test labels hidden", "validation F1", "Elliptic, DGraphFin"],
            ["late-window holdout", "60/20/20 chronological", "shared node-history map uses first 60%, which is this protocol's training interval", "test transaction labels hidden", "fixed 0.5 threshold for saved F1", "IBM AML-Data"],
            ["early-to-late transfer", "50/20/30 chronological", "classifier labels use first 50%; shared label-free node-history map uses first 60%, including covariates from 50-60%", "test transaction labels hidden", "fixed 0.5 threshold for saved F1", "IBM AML-Data"],
        ],
        columns=["protocol", "temporal_masks", "graph_visibility", "label_availability", "selection_or_threshold", "instantiated_on"],
    )
    models = pd.DataFrame(
        [
            ["MLP", "node features only", "negative control for graph-visibility intervention", "Elliptic h256/l3; DGraphFin h64/l2"],
            ["GCN", "normalized neighborhood aggregation", "tests spectral-style smoothing under visibility change", "Elliptic h256/l3; DGraphFin h64/l2"],
            ["GraphSAGE", "mean neighborhood aggregation", "tests inductive aggregation under visibility change", "Elliptic h256/l3; DGraphFin h64/l2"],
            ["Logistic regression", "standardized transaction features", "linear non-graph baseline", "balanced classes; max_iter=500"],
            ["Histogram gradient boosting", "transaction features", "nonlinear tabular baseline", "160 iterations; learning rate 0.06"],
            ["GraphSAGE-derived edge classifier h32", "training-history endpoint/neighborhood summaries plus edge features", "small IBM graph baseline", "20 epochs; batch 65,536"],
            ["Edge-aware GraphSAGE-derived edge classifier h64", "training-history endpoint/neighborhood summaries plus edge features", "reference for graph-feature/construction ablations", "30 epochs; batch 32,768"],
            ["GINE h64", "one GINE layer and edge head", "edge-conditioned message-passing architecture", "30 epochs; batch 32,768; Small only"],
            ["NoEdge", "zero transaction attributes; structure retained", "tests edge-attribute contribution", "matched to h64 reference"],
            ["ShuffledEdge", "edge attributes permuted within temporal masks", "destroys edge-feature alignment while preserving marginal values", "matched to h64 reference"],
            ["DegreeOnly", "endpoint-degree attributes replace transaction features", "tests structural summaries without original attributes", "matched to h64 reference"],
            ["DegreeCap", "q=.995 cap on training structure", "tests hub sensitivity and resource/performance tradeoff", "matched to h64 reference"],
            ["RecentWindow", "most recent 50% of training edges", "tests recency and reduced computation", "matched to h64 reference"],
        ],
        columns=["method_or_construction", "computational_form", "hypothesis", "configuration"],
    )
    training = pd.DataFrame(
        [
            ["Elliptic protocol grid", "AdamW", "1e-3", "5e-4", "class-weighted CE", "200", "40 checks (inactive within cap)", "validation F1", "argmax over 3 repository label codes", "full graph forward", "h256/l3/dropout .5"],
            ["DGraphFin protocol grid", "AdamW", "1e-3", "5e-4", "class-weighted CE", "200", "40 checks (inactive within cap)", "validation F1", "argmax over 3 repository label codes", "full graph forward", "h64/l2/dropout .5"],
            ["IBM LR", "LBFGS family (scikit-learn)", "library default", "library default", "balanced class weight", "max_iter 500", "none", "training only", "score >= .5", "CPU", "standardized features"],
            ["IBM HistGB", "histogram gradient boosting", ".06", "n/a", "binary log loss", "160", "none", "training only", "score >= .5", "CPU", "library defaults otherwise"],
            ["IBM GraphSAGE-derived h32", "AdamW", "1e-3", "1e-4", "positive-weighted BCE", "20", "none", "fixed seed", "score >= .5", "edge minibatches 65,536", "h32/dropout .1"],
            ["IBM h64 graph grid", "AdamW", "1e-3", "1e-4", "positive-weighted BCE", "30", "none", "fixed seed", "score >= .5", "edge minibatches 32,768", "h64/dropout .1"],
        ],
        columns=["family", "optimizer", "learning_rate", "weight_decay", "loss", "max_epochs_or_iterations", "early_stopping", "selection", "threshold_rule", "batch_or_forward", "architecture"],
    )
    return axes, protocols, models, training


def resource_boundaries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["IBM AML HI-Large", "safe resource guard", "0", "0", "SAFE_RESOURCE_BLOCKED", "not in predictive ranking"],
            ["IBM AML LI-Large", "safe resource guard", "0", "0", "SAFE_RESOURCE_BLOCKED", "not in predictive ranking"],
            ["IBM AML HI-Medium GINE h64", "single Tesla T4 CUDA OOM", "0 of 20 planned", "0", "RESOURCE_BLOCKED_T4_CUDA_OOM", "not in predictive ranking"],
            ["IBM AML LI-Medium GINE h64", "single Tesla T4 CUDA OOM", "0 of 20 planned", "0", "RESOURCE_BLOCKED_T4_CUDA_OOM", "not in predictive ranking"],
            ["DGraphFin GAT h64/l2", "Tesla T4 CUDA OOM", "0 of 20 planned", "0", "BLOCKED_T4_OOM", "memory-reduced h32/l1 diagnostic is not a replacement"],
            ["DGraphFin GraphSAGE max-pool rerun", "larger GPU required", "0", "0", "BLOCKED_WAITING_FOR_GPU", "no imported result CSV"],
        ],
        columns=["cell", "resource_envelope_or_reason", "result_outputs", "prediction_exports", "status", "interpretation"],
    )


def add_scalar_provenance(
    records: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    artifact: str,
    key_fields: list[str],
    numeric_fields: list[str],
    sources: str,
    filters: str,
    seed_set: str,
    aggregation: str,
    lock: str,
    claims: str,
    manuscript_use: str,
) -> None:
    for _, row in df.iterrows():
        key = "|".join(f"{field}={row[field]}" for field in key_fields)
        row_sources = str(row.get("source_paths", row.get("source", sources)))
        for field in numeric_fields:
            if field not in row or pd.isna(row[field]):
                continue
            slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{artifact}_{key}_{field}").strip("_")
            records.append(
                {
                    "number_id": f"N_{slug}",
                    "manuscript_use": manuscript_use,
                    "artifact": artifact,
                    "row_key": key,
                    "field": field,
                    "value": row[field],
                    "units": "seconds" if "runtime" in field else "percent" if field.endswith("_pct") else "count" if field in {"n", "n_pairs", "n_seeds"} or field.endswith("_units") or field.endswith("_positives") else "proportion/score",
                    "source_files": row_sources or sources,
                    "aggregation_script": SCRIPT,
                    "filters": filters,
                    "seed_set": seed_set,
                    "aggregation": aggregation,
                    "output_location": f"results/tkde_rebuild/{artifact}.csv",
                    "evidence_lock": lock,
                    "claim_ids": claims,
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, Any]] = []

    datasets = dataset_statistics()
    write_df(datasets, "DATASET_TASK_STATISTICS", "Exact task units, label semantics, graph sizes, temporal splits, and class priors. IBM rows are protocol-specific because test prevalence changes with the window.")
    add_scalar_provenance(provenance, datasets, artifact="DATASET_TASK_STATISTICS", key_fields=["dataset", "variant", "protocol"], numeric_fields=["nodes_or_accounts", "source_edges_or_transactions", "message_passing_arcs", "node_feature_dim", "edge_feature_dim", "train_units", "validation_units", "test_units", "train_positives", "validation_positives", "test_positives", "test_positive_rate", "total_labeled", "total_positives"], sources="raw dataset files and locked result split counts", filters="eligible labeled units only for Elliptic/DGraphFin", seed_set="not applicable", aggregation="exact counts; one canonical payload for invariant graph/split metadata", lock="RB09 artifact family; V26 lock", claims="C01-C12", manuscript_use="dataset/task table and methods")

    effects, rb09_main = rb09_effects()
    write_df(effects, "RB09_PROTOCOL_EFFECTS", "Paired isolated-minus-strict effects over the ten shared seeds. Holm adjustment is applied separately within each metric's six dataset/model comparisons.")
    write_df(rb09_main, "RB09_AUPRC_MAIN", "Main-paper AUPRC view of the paired protocol effects; MLP is the graph-visibility negative control.")
    add_scalar_provenance(provenance, effects, artifact="RB09_PROTOCOL_EFFECTS", key_fields=["dataset", "model", "metric"], numeric_fields=[c for c in effects.columns if c not in {"dataset", "model", "metric", "source", "strict_summary", "isolated_summary"}], sources="results/runs_rb09v3/runs.csv", filters="protocol in {strict_inductive,inductive_isolated}; pair on seed", seed_set="1-10", aggregation="mean/std; deterministic percentile bootstrap; Wilcoxon; Holm within metric", lock="results/runs_rb09v3/ARTIFACT_FAMILY.json", claims="C01;C02;C03;C04", manuscript_use="RQ1/RQ2 table and figure")

    dup, dedup = v24_duplicate_audit()
    write_df(dup, "V24_DUPLICATE_STRESS_AUDIT", "Construct-validity audit. The V24 runner records three stress labels but does not pass them to the harness; all performance metrics repeat within each base cell.")
    write_df(dedup, "V24_DEDUPLICATED_RERUN_RESULTS", "One representative retained per dataset/protocol/model/seed for supplementary reproducibility only. The stress label has no scientific interpretation.")
    add_scalar_provenance(provenance, dup, artifact="V24_DUPLICATE_STRESS_AUDIT", key_fields=["dataset", "protocol", "model", "seed"], numeric_fields=["n_stress_labels"], sources="results/v24_imported/*/RESULT_INDEX.csv; scripts/tkde_max/run_v24_temporal_stress.py", filters="RB41 only", seed_set="1-10", aggregation="exact equality audit", lock="results/v24_imported/V24_IMPORTED_EVIDENCE_LOCK.json", claims="C22", manuscript_use="framework validation and supplement exclusion audit")

    ibm = collect_ibm_rows()
    write_df(ibm, "IBM_IMPORTED_SEED_ROWS", "Canonical seed-level V26-V28 IBM AML evidence. This is regenerated directly from the 840 imported JSON files.")
    summary = ibm_summary(ibm)
    write_df(summary, "IBM_CELL_SUMMARY", "Full ten-seed means, standard deviations, deterministic bootstrap intervals, prevalence normalization, and configuration-specific runtime for all 84 measured cells.")
    add_scalar_provenance(provenance, summary, artifact="IBM_CELL_SUMMARY", key_fields=["version", "variant", "protocol", "config"], numeric_fields=[c for c in summary.columns if c not in {"version", "variant", "size", "regime", "protocol", "config", "source_paths"}], sources="840 canonical imported JSON files", filters="COUNTABLE_RESULT; execute=true; dry_run=false; return_code=0", seed_set="1-10", aggregation="mean/std and deterministic percentile bootstrap per exact cell", lock="V26/V27/V28 canonical imported evidence locks", claims="C05-C14;C21", manuscript_use="IBM main/full result tables and figures")

    ranks, rank_cells = rank_divergence(ibm)
    write_df(ranks, "IBM_METRIC_RANKS", "Configuration ranks within identical variant/protocol/feasibility families. The V28 sender-receiver contract alias is excluded as non-independent.")
    write_df(rank_cells, "IBM_RANK_DIVERGENCE", "Winner disagreement and Spearman rank correlation between ranking and fixed-threshold metrics.")
    add_scalar_provenance(provenance, rank_cells, artifact="IBM_RANK_DIVERGENCE", key_fields=["family", "variant", "protocol"], numeric_fields=["n_configurations", "spearman_auprc_vs_f1", "spearman_auprc_vs_f1_p", "spearman_auroc_vs_auprc", "spearman_auroc_vs_auprc_p"], sources="results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv", filters="rank within family/variant/protocol; exclude contract alias", seed_set="cell means over seeds 1-10", aggregation="mean then descending rank; Spearman correlation", lock="V26/V27/V28 locks", claims="C06", manuscript_use="rank-versus-decision table and figure")

    ablations, ablation_contexts = matched_ablation(ibm)
    write_df(
        ablations,
        "IBM_MATCHED_ABLATION_EFFECTS",
        "Candidate-minus-reference effects over the fixed four-context grid. Contexts are averaged within seed before inference (n=10 seed blocks); Small and Medium are separate, GINE has Small rows only, and Holm correction is within each size/metric family.",
    )
    write_df(
        ablation_contexts,
        "IBM_MATCHED_ABLATION_CONTEXT_EFFECTS",
        "Context-specific ten-seed candidate-minus-reference effects. These rows expose variant/protocol heterogeneity and are descriptive; confirmatory inference uses the seed-blocked fixed-grid aggregate.",
    )
    add_scalar_provenance(provenance, ablations, artifact="IBM_MATCHED_ABLATION_EFFECTS", key_fields=["config", "size", "metric"], numeric_fields=[c for c in ablations.columns if c not in {"config", "size", "metric", "source_paths"}], sources="V27 reference plus V28 candidate JSON files", filters="pair by variant/protocol/seed; use intersecting feasible cells only", seed_set="1-10, with four fixed contexts averaged within seed", aggregation="mean fixed-grid delta by seed; seed-block bootstrap and Wilcoxon; Holm within size/metric", lock="V27/V28 locks", claims="C07-C10;C13", manuscript_use="dependence-aware matched ablation table and figure")
    add_scalar_provenance(provenance, ablation_contexts, artifact="IBM_MATCHED_ABLATION_CONTEXT_EFFECTS", key_fields=["config", "size", "metric", "variant", "protocol"], numeric_fields=[c for c in ablation_contexts.columns if c not in {"config", "size", "metric", "variant", "protocol", "source_paths"}], sources="V27 reference plus V28 candidate JSON files", filters="pair within exact variant/protocol/seed context; use intersecting feasible cells only", seed_set="1-10 per exact context", aggregation="context-specific paired mean, bootstrap, and descriptive Wilcoxon", lock="V27/V28 locks", claims="C07-C10;C13", manuscript_use="supplementary dependence and heterogeneity sensitivity")

    runtime = runtime_feasibility(ibm)
    write_df(runtime, "IBM_RUNTIME_FEASIBILITY", "Configuration-specific performance and runtime. Pareto status is computed only within the same variant/protocol. Blocked GINE rows have no performance values.")
    add_scalar_provenance(provenance, runtime, artifact="IBM_RUNTIME_FEASIBILITY", key_fields=["variant", "protocol", "config", "status"], numeric_fields=["n", "auprc_mean", "auprc_std", "normalized_auprc_mean", "auroc_mean", "f1_mean", "runtime_seconds_mean", "runtime_seconds_min", "runtime_seconds_max"], sources="V27/V28 JSON and V28 blocked-run lock", filters="measured graph grid excluding contract alias; blocked rows separate", seed_set="1-10 for measured; none for blocked", aggregation="mean/min/max within exact cell; nondominance within variant/protocol", lock="V27/V28 locks", claims="C09-C12;C20", manuscript_use="runtime/resource table and figure")

    gs_summary, gs_paired, budgets, calibration = graphsafe_surfaces()
    write_df(gs_summary, "GRAPHSAFE_BOUNDED_SUMMARY", "Validated saved-output aggregate for five comparators. Each dataset/method has 60 source protocol/model/seed rows.")
    write_df(gs_paired, "GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE", "Paired method improvement over simple averaging; lower-is-better metrics are sign-normalized so positive is favorable. Holm adjustment spans the declared 48-comparison family.")
    write_df(budgets, "REVIEW_BUDGET_CURVES", "Precision and recall at 0.5%, 1%, and 2% review capacities from saved predictions/policy rows.")
    write_df(calibration, "GRAPHSAFE_CALIBRATION_SUMMARY", "ECE and Brier diagnostics for the bounded saved-output case study.")
    add_scalar_provenance(provenance, gs_summary, artifact="GRAPHSAFE_BOUNDED_SUMMARY", key_fields=["dataset", "method"], numeric_fields=[c for c in gs_summary.columns if c not in {"dataset", "method", "source"}], sources="results/runs_rb17_review_budget_worst_block/rb17_results.csv", filters="selected five methods", seed_set="1-10 across source protocol/model cells", aggregation="descriptive mean/std/bootstrap", lock="RB17 import and validation manifests", claims="C15-C17", manuscript_use="bounded GraphSafe case table")
    add_scalar_provenance(provenance, budgets, artifact="REVIEW_BUDGET_CURVES", key_fields=["dataset", "method", "budget_pct"], numeric_fields=["n", "precision_mean", "precision_std", "recall_mean", "recall_std"], sources="results/runs_rb17_review_budget_worst_block/rb17_results.csv", filters="selected five methods", seed_set="1-10 across source protocol/model cells", aggregation="mean/std by budget", lock="RB17 import and validation manifests", claims="C06;C15-C17", manuscript_use="review-budget figure and supplement")

    axes, protocols, models, training = static_method_tables()
    write_df(axes, "DEPLOYMENT_CONTRACT_AXES", "Separately recorded, non-substitutable coordinates of the formal deployment contract Π=(T,V,C,S,B,R).")
    write_df(protocols, "PROTOCOL_DEFINITIONS", "Instantiated protocol contracts; no hierarchy is implied among separately recorded coordinates.")
    write_df(models, "MODEL_CONSTRUCTION_INVENTORY", "Scientific method names, computational forms, hypotheses, and exact configurations.")
    write_df(training, "TRAINING_CONFIGURATION", "Recovered optimizer, loss, threshold, batching, width, depth, and selection settings by family.")
    resources = resource_boundaries()
    write_df(resources, "RESOURCE_BOUNDARIES", "Resource status is not predictive performance. Rows with zero outputs remain outside rankings.")
    add_scalar_provenance(provenance, resources, artifact="RESOURCE_BOUNDARIES", key_fields=["cell", "status"], numeric_fields=[], sources="V22/V24/V26/V28/RB18 locks", filters="blocked/resource-blocked records", seed_set="none", aggregation="exact status", lock="canonical locks", claims="C10-C12;C20", manuscript_use="resource-feasibility table")

    provenance_df = pd.DataFrame(provenance, columns=PROVENANCE_FIELDS).sort_values("number_id").reset_index(drop=True)
    if provenance_df.number_id.duplicated().any():
        dupes = provenance_df[provenance_df.number_id.duplicated()].number_id.tolist()
        raise RuntimeError(f"duplicate provenance ids: {dupes[:5]}")
    provenance_df.to_csv(OUT / "NUMBER_PROVENANCE_MAP.csv", index=False)
    (OUT / "NUMBER_PROVENANCE_MAP.md").write_text(
        "# Number Provenance Map\n\n"
        f"The CSV contains **{len(provenance_df)} scalar records**. Each number generated for a main or supplementary quantitative surface maps to source files, filters, seed coverage, aggregation code, output, evidence lock, and claim IDs. Long source lists remain in the CSV to keep this document readable.\n",
        encoding="utf-8",
    )
    print(f"analysis complete: {len(provenance_df)} scalar provenance records")


if __name__ == "__main__":
    main()
