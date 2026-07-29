#!/usr/bin/env python3
"""Shared helpers for RB16/RB17 GraphSafe strengthening artifacts.

All functions operate on saved RB15/RB15b CSV outputs. They do not train,
launch GPU work, or tune on test labels for policy selection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.result_analysis import ensure_dir, markdown_table, utc_now, write_csv, write_json, write_tex  # noqa: E402

try:  # pragma: no cover - depends on local env.
    from scipy.stats import ttest_rel, wilcoxon

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

KEYS = ["dataset", "protocol", "graph_model", "seed"]

HIGHER_IS_BETTER = {
    "f1",
    "precision",
    "recall",
    "auroc",
    "auprc",
    "precision_at_k",
    "recall_at_k",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_0_005",
    "recall_at_0_005",
    "precision_at_0_5pct",
    "recall_at_0_5pct",
    "precision_at_0_01",
    "recall_at_0_01",
    "precision_at_0_02",
    "recall_at_0_02",
    "precision_at_2pct",
    "recall_at_2pct",
    "worst_window_f1",
    "worst_window_recall_at_k",
    "worst_window_recall_at_1pct",
}

LOWER_IS_BETTER = {
    "ece",
    "brier",
    "cost_sensitive_risk",
    "worst_block_regret",
    "average_block_regret",
    "worst_window_cost_sensitive_risk",
    "worst_block_f1_regret",
}

PRIMARY_METRICS = [
    "f1",
    "auprc",
    "auroc",
    "ece",
    "brier",
    "precision_at_0_005",
    "recall_at_0_005",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_0_02",
    "recall_at_0_02",
    "cost_sensitive_risk",
    "worst_block_regret",
    "average_block_regret",
    "worst_window_f1",
    "worst_window_recall_at_k",
    "worst_window_cost_sensitive_risk",
]

BEST_BRANCH_COMPARATORS = [
    "best_val_branch",
    "tpc_tta_graph",
    "feature_only",
    "graph_only",
    "simple_average",
]

BASELINE_METHODS = [
    "feature_only",
    "graph_only",
    "best_val_branch",
    "simple_average",
    "validation_weighted_ensemble",
    "temperature_graph",
    "prior_ratio_graph",
    "threshold_tta_graph",
    "tpc_tta_graph",
    "graphsafe_tta_full",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    renames = {
        "graph_alpha_x": "graph_alpha",
        "best_val_branch_x": "best_val_branch",
        "graph_alpha_y": "policy_graph_alpha",
        "best_val_branch_y": "policy_best_val_branch",
    }
    for old, new in renames.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    if "worst_window_recall_at_k" in out.columns and "worst_window_recall_at_1pct" not in out.columns:
        out["worst_window_recall_at_1pct"] = out["worst_window_recall_at_k"]
    if "selection_split" not in out.columns:
        out["selection_split"] = "train/val only"
    if "evaluation_split" not in out.columns:
        out["evaluation_split"] = "test"
    return out


def load_rb15_bundle(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rb15 = root / "results" / "runs_rb15_graphsafe_tta"
    rb15b = root / "results" / "runs_rb15b_graphsafe_conservative_policy"
    main = normalize_results(read_csv(rb15 / "graphsafe_tta_results.csv"))
    ablation = normalize_results(read_csv(rb15 / "graphsafe_tta_ablation_results.csv"))
    reliability = read_csv(rb15 / "graphsafe_tta_reliability_scores.csv")
    conservative = normalize_results(read_csv(rb15b / "conservative_policy_results.csv"))
    manifest = {
        "rb15": _load_json(rb15 / "import_manifest.json"),
        "rb15b": _load_json(rb15b / "import_manifest.json"),
    }
    return main, ablation, reliability, conservative, manifest


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def val_reliability_summary(reliability: pd.DataFrame) -> pd.DataFrame:
    val = reliability.loc[reliability["split"].astype(str) == "val"].copy()
    if val.empty:
        val = reliability.copy()
    grouped = (
        val.groupby(KEYS, as_index=False)
        .agg(
            val_mean_reliability_risk=("mean_reliability_risk", "mean"),
            val_max_reliability_risk=("mean_reliability_risk", "max"),
            val_mean_safe_graph_rate=("safe_graph_rate", "mean"),
            val_min_safe_graph_rate=("safe_graph_rate", "min"),
            val_mean_score_disagreement=("mean_score_disagreement", "mean"),
            val_mean_rank_disagreement=("mean_rank_disagreement", "mean"),
            val_n_blocks=("timestep", "nunique"),
        )
    )
    risk_cutoff = float(grouped["val_mean_reliability_risk"].median()) if not grouped.empty else 0.0
    block_risk_cutoff = float(grouped["val_max_reliability_risk"].quantile(0.75)) if not grouped.empty else 0.0
    safe_cutoff = float(grouped["val_mean_safe_graph_rate"].quantile(0.25)) if not grouped.empty else 0.0
    grouped["rb16_high_risk_cutoff"] = risk_cutoff
    grouped["rb16_block_risk_cutoff"] = block_risk_cutoff
    grouped["rb16_safe_rate_cutoff"] = safe_cutoff
    grouped["rb16_high_risk"] = grouped["val_mean_reliability_risk"] >= risk_cutoff
    grouped["rb16_block_high_risk"] = (grouped["val_max_reliability_risk"] >= block_risk_cutoff) | (
        grouped["val_mean_safe_graph_rate"] <= safe_cutoff
    )
    return grouped


def method_index(df: pd.DataFrame) -> Dict[Tuple[Any, ...], pd.Series]:
    return {tuple(row[k] for k in KEYS): row for _, row in df.iterrows()}


def copy_policy_row(row: pd.Series, method: str, source_method: str, **extra: Any) -> Dict[str, Any]:
    out = row.to_dict()
    out["method"] = method
    out["rb16_policy"] = method
    out["source_method"] = source_method
    out.setdefault("selection_split", "train/val only")
    out.setdefault("evaluation_split", "test")
    out.update(extra)
    return out


def choose_between(
    preferred: pd.DataFrame,
    fallback: pd.DataFrame,
    decisions: pd.DataFrame,
    choose_col: str,
    method: str,
    source_when_preferred: str,
    source_when_fallback: str,
    reason_col: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    extra = dict(extra or {})
    pref = method_index(preferred)
    fb = method_index(fallback)
    rows: List[Dict[str, Any]] = []
    for _, dec in decisions.iterrows():
        key = tuple(dec[k] for k in KEYS)
        if key not in pref or key not in fb:
            continue
        use_pref = bool(dec[choose_col])
        chosen = pref[key] if use_pref else fb[key]
        source = source_when_preferred if use_pref else source_when_fallback
        dec_extra = {
            "policy_reason": dec.get(reason_col, ""),
            "policy_selected_graphsafe": use_pref,
            "val_mean_reliability_risk": dec.get("val_mean_reliability_risk", ""),
            "val_mean_safe_graph_rate": dec.get("val_mean_safe_graph_rate", ""),
            **extra,
        }
        rows.append(copy_policy_row(chosen, method, source, **dec_extra))
    return rows


def _normal_pvalue(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size < 2:
        return 1.0
    sd = float(values.std(ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(float(values.mean())) > 1e-12 else 1.0
    z = float(values.mean()) / (sd / math.sqrt(values.size))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _paired_pvalues(improvement: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if len(improvement) < 2:
        return 1.0, 1.0
    if np.all(np.abs(a - b) <= 1e-12):
        return 1.0, 1.0
    if HAVE_SCIPY:
        try:
            try:
                wp = float(wilcoxon(improvement, zero_method="wilcox", correction=False, alternative="two-sided", method="asymptotic").pvalue)
            except TypeError:
                wp = float(wilcoxon(improvement, zero_method="wilcox", correction=False, alternative="two-sided", method="approx").pvalue)
        except Exception:
            wp = 1.0
        try:
            tp = float(ttest_rel(a, b).pvalue)
        except Exception:
            tp = _normal_pvalue(improvement)
        return wp, tp
    return 1.0, _normal_pvalue(improvement)


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n: int = 1000) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _holm(pvals: Sequence[float]) -> List[float]:
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adjusted = [1.0] * n
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (n - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _bh(pvals: Sequence[float]) -> List[float]:
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adjusted = [1.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = idx[rank]
        prev = min(prev, pvals[i] * n / max(rank + 1, 1))
        adjusted[i] = min(1.0, prev)
    return adjusted


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    raise KeyError(f"unknown metric direction: {metric}")


def paired_tests(
    rows: pd.DataFrame,
    methods: Sequence[str],
    comparators: Sequence[str],
    metrics: Sequence[str],
    *,
    comparison_prefix: str,
) -> pd.DataFrame:
    stat_rows: List[Dict[str, Any]] = []
    for method in methods:
        a_all = rows.loc[rows["method"].astype(str) == method]
        if a_all.empty:
            continue
        for comparator in comparators:
            if comparator == method:
                continue
            b_all = rows.loc[rows["method"].astype(str) == comparator]
            if b_all.empty:
                continue
            for (dataset, protocol, graph_model), a_group in a_all.groupby(["dataset", "protocol", "graph_model"]):
                b_group = b_all.loc[
                    (b_all["dataset"] == dataset)
                    & (b_all["protocol"] == protocol)
                    & (b_all["graph_model"] == graph_model)
                ]
                a_idx = a_group.set_index("seed")
                b_idx = b_group.set_index("seed")
                seeds = sorted(set(a_idx.index) & set(b_idx.index))
                if len(seeds) < 2:
                    continue
                for metric in metrics:
                    if metric not in a_idx.columns or metric not in b_idx.columns:
                        continue
                    direction = metric_direction(metric)
                    av = pd.to_numeric(a_idx.loc[seeds, metric], errors="coerce").to_numpy(dtype=float)
                    bv = pd.to_numeric(b_idx.loc[seeds, metric], errors="coerce").to_numpy(dtype=float)
                    mask = ~(np.isnan(av) | np.isnan(bv))
                    av = av[mask]
                    bv = bv[mask]
                    if len(av) < 2:
                        continue
                    improvement = av - bv if direction == "higher" else bv - av
                    wp, tp = _paired_pvalues(improvement, av, bv)
                    lo, hi = _bootstrap_ci(improvement)
                    stat_rows.append(
                        {
                            "dataset": dataset,
                            "protocol": protocol,
                            "graph_model": graph_model,
                            "comparison": f"{comparison_prefix}_{method}_vs_{comparator}",
                            "method": method,
                            "comparator": comparator,
                            "metric": metric,
                            "direction": direction,
                            "method_mean": float(np.mean(av)),
                            "comparator_mean": float(np.mean(bv)),
                            "mean_improvement": float(np.mean(improvement)),
                            "bootstrap_ci95_lo": lo,
                            "bootstrap_ci95_hi": hi,
                            "wilcoxon_p": wp,
                            "ttest_p": tp,
                            "n_paired_seeds": int(len(av)),
                        }
                    )
    if not stat_rows:
        return pd.DataFrame()
    pvals = [float(r["ttest_p"]) for r in stat_rows]
    holm = _holm(pvals)
    bh = _bh(pvals)
    for row, hp, bp in zip(stat_rows, holm, bh):
        row["holm_p"] = hp
        row["bh_p"] = bp
        positive = float(row["mean_improvement"]) > 0
        if positive and hp < 0.05:
            row["win_label"] = "holm_corrected_win"
        elif positive and bp < 0.05:
            row["win_label"] = "fdr_only_win"
        elif positive:
            row["win_label"] = "diagnostic_only_positive"
        else:
            row["win_label"] = "negative_or_no_gain"
    return pd.DataFrame(stat_rows)


def aggregate_method_table(rows: pd.DataFrame, methods: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    sub = rows if methods is None else rows.loc[rows["method"].isin(methods)]
    out: List[Dict[str, Any]] = []
    metrics = [
        "f1",
        "auprc",
        "auroc",
        "ece",
        "brier",
        "precision_at_0_005",
        "recall_at_0_005",
        "precision_at_1pct",
        "recall_at_1pct",
        "precision_at_0_02",
        "recall_at_0_02",
        "cost_sensitive_risk",
        "worst_block_regret",
        "average_block_regret",
        "worst_window_f1",
        "worst_window_recall_at_1pct",
        "worst_window_cost_sensitive_risk",
    ]
    for (dataset, method), group in sub.groupby(["dataset", "method"]):
        row: Dict[str, Any] = {"dataset": dataset, "method": method, "n": int(len(group))}
        for metric in metrics:
            if metric in group.columns:
                row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        out.append(row)
    return sorted(out, key=lambda r: (str(r["dataset"]), str(r["method"])))


def rows_to_markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return markdown_table(rows, columns)
