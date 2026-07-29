#!/usr/bin/env python3
"""Run GraphSafe-TTA from saved prediction artifacts only.

GraphSafe-TTA is a deployment-time decision method over already-trained branch
predictions. This script never trains a model and never rewrites source
prediction artifacts. All gates, thresholds, weights, and reliability cutoffs
are fit on train/validation rows; test labels are used only in final metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.result_analysis import (  # noqa: E402
    as_binary,
    best_threshold,
    brier_score,
    correction_rows,
    ece_score,
    ensure_dir,
    ks_distance,
    normal_approx_pvalue,
    precision_recall_f1,
    rank_auc,
    score_prior_adjust,
    utc_now,
    write_csv,
    write_json,
)

MODELS = ("mlp", "gcn", "sage")
GRAPH_MODELS = ("gcn", "sage")
DEFAULT_PROTOCOLS = ("strict_inductive", "inductive_isolated", "transductive")
DEFAULT_DATASETS = ("elliptic", "dgraphfin")
BUDGETS = (0.005, 0.01, 0.02)
EPS = 1e-9


@dataclass(frozen=True)
class PredictionSource:
    dataset: str
    protocol: str
    model: str
    seed: int
    source_path: Path
    member: Optional[str] = None
    manifest_path: Optional[str] = None

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return (self.dataset, self.protocol, self.model, self.seed)

    @property
    def display_path(self) -> str:
        if self.member:
            return f"{self.source_path}::{self.member}"
        return str(self.source_path)


def _parse_csv_name(path: str) -> Optional[Tuple[str, str, str, int]]:
    stem = Path(path).name
    if not stem.endswith(".csv"):
        return None
    parts = stem[:-4].split("__")
    if len(parts) != 4 or not parts[3].startswith("seed"):
        return None
    return parts[0], parts[1], parts[2], int(parts[3].replace("seed", ""))


def _zip_for(root: Path, dataset: str, protocol: str) -> Path:
    return root / "kaggleoutputs" / f"{dataset}_10seed_{protocol}.zip"


def _resolve_prediction_source(root: Path, rec: Mapping[str, Any]) -> Optional[PredictionSource]:
    parsed = (
        str(rec.get("dataset", "")),
        str(rec.get("protocol", "")),
        str(rec.get("model", "")),
        int(rec.get("seed", 0)),
    )
    rel = str(rec.get("path", ""))
    path = (root / rel).resolve()
    if path.is_file():
        return PredictionSource(*parsed, source_path=path, manifest_path=rel)
    dataset, protocol, model, seed = parsed
    member = f"predictions/{dataset}__{protocol}__{model}__seed{seed}.csv"
    zpath = _zip_for(root, dataset, protocol).resolve()
    if zpath.is_file():
        try:
            with zipfile.ZipFile(zpath) as zf:
                if member in set(zf.namelist()):
                    return PredictionSource(*parsed, source_path=zpath, member=member, manifest_path=rel)
        except zipfile.BadZipFile:
            return None
    return None


def discover_sources(root: Path) -> Dict[Tuple[str, str, str, int], PredictionSource]:
    out: Dict[Tuple[str, str, str, int], PredictionSource] = {}
    manifest = root / "results" / "runs_rb09v3" / "predictions_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for rec in payload.get("files", []):
            src = _resolve_prediction_source(root, rec)
            if src is not None:
                out[src.key] = src
    fallback_dirs = [
        root / "results" / "runs_rb11_sage_family" / "predictions",
        root / "results" / "runs" / "predictions",
        root / "results" / "runs_rb01" / "predictions",
    ]
    for pred_dir in fallback_dirs:
        if not pred_dir.is_dir():
            continue
        for path in pred_dir.glob("*.csv"):
            parsed = _parse_csv_name(path.name)
            if parsed is None:
                continue
            out.setdefault(parsed, PredictionSource(*parsed, source_path=path.resolve(), manifest_path=str(path.relative_to(root))))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction(src: PredictionSource) -> pd.DataFrame:
    cols = ["split", "node_id", "timestep", "y_true", "score", "label_known"]
    if src.member:
        with zipfile.ZipFile(src.source_path) as zf:
            with zf.open(src.member) as handle:
                df = pd.read_csv(handle, usecols=lambda c: c in cols)
    else:
        df = pd.read_csv(src.source_path, usecols=lambda c: c in cols)
    if "label_known" in df.columns:
        df = df.loc[df["label_known"].astype(str).str.lower().isin(["true", "1", "yes"])]
    df = df.loc[df["y_true"].isin([1, 2])].copy()
    df["score"] = df["score"].astype(float).clip(0.0, 1.0)
    df["timestep"] = pd.to_numeric(df["timestep"], errors="coerce").fillna(-1).astype(int)
    return df[["split", "node_id", "timestep", "y_true", "score"]]


def align_pair(feature: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    f = feature.rename(columns={"score": "feature_score"})
    g = graph.rename(columns={"score": "graph_score"})
    return f.merge(g, on=["split", "node_id", "timestep", "y_true"], how="inner")


def _rank_pct(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return np.zeros(arr.size, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks / max(arr.size - 1, 1)


def _binary_entropy(scores: Sequence[float]) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def add_reliability_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_disagreement"] = (out["graph_score"] - out["feature_score"]).abs()
    out["entropy_mean"] = (_binary_entropy(out["graph_score"]) + _binary_entropy(out["feature_score"])) / 2.0
    out["rank_disagreement"] = 0.0
    for split, idx in out.groupby("split").groups.items():
        sub = out.loc[idx]
        out.loc[idx, "rank_disagreement"] = np.abs(_rank_pct(sub["graph_score"]) - _rank_pct(sub["feature_score"]))
    val = out.loc[out["split"].astype(str) == "val"]
    if val.empty:
        val = out
    scales = {
        "score_disagreement": max(float(val["score_disagreement"].quantile(0.95)), EPS),
        "rank_disagreement": max(float(val["rank_disagreement"].quantile(0.95)), EPS),
        "entropy_mean": max(float(val["entropy_mean"].quantile(0.95)), EPS),
    }
    out["reliability_risk"] = (
        (out["score_disagreement"] / scales["score_disagreement"])
        + (out["rank_disagreement"] / scales["rank_disagreement"])
        + 0.5 * (out["entropy_mean"] / scales["entropy_mean"])
    ) / 2.5
    return out


def _top_metrics(labels: Sequence[int], scores: Sequence[float], frac: float) -> Tuple[float, float, int]:
    y = as_binary(labels)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return 0.0, 0.0, 0
    k = max(1, int(math.ceil(y.size * frac)))
    order = np.argsort(-s, kind="mergesort")[:k]
    tp = int(y[order].sum())
    precision = tp / k if k else 0.0
    recall = tp / int(y.sum()) if int(y.sum()) else 0.0
    return float(precision), float(recall), int(k)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = as_binary(labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def _cost_risk(labels: Sequence[int], scores: Sequence[float], threshold: float, cost_fp: float, cost_fn: float) -> float:
    m = precision_recall_f1(labels, scores, threshold)
    return float((cost_fp * m["fp"] + cost_fn * m["fn"]) / max(len(labels), 1))


def _threshold_for_budget(scores: Sequence[float], frac: float) -> float:
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return 0.5
    k = max(1, int(math.ceil(s.size * frac)))
    return float(np.sort(s)[max(0, s.size - k)])


def _sigmoid_logit_adjust(scores: Sequence[float], temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / max(temperature, 1e-6)))


def _nll(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = as_binary(labels).astype(float)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if y.size else 0.0


def fit_temperature(labels: Sequence[int], scores: Sequence[float]) -> float:
    grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    best = min(grid, key=lambda t: _nll(labels, _sigmoid_logit_adjust(scores, t)))
    return float(best)


def _score_bundle(df: pd.DataFrame, graph_alpha: float, cutoff: float, mode: str) -> np.ndarray:
    feature = df["feature_score"].to_numpy(dtype=float)
    graph = df["graph_score"].to_numpy(dtype=float)
    if mode == "feature_only":
        return feature
    if mode == "graph_only":
        return graph
    if mode == "simple_average":
        return 0.5 * graph + 0.5 * feature
    if mode in {"validation_weighted_ensemble", "graphsafe_no_reliability", "tta_only"}:
        return graph_alpha * graph + (1.0 - graph_alpha) * feature
    if mode in {"graphsafe_tta_full", "graphsafe_no_tta", "reliability_estimator_only"}:
        use_graph = df["reliability_risk"].to_numpy(dtype=float) <= cutoff
        return np.where(use_graph, graph, feature)
    if mode == "graphsafe_no_fallback":
        return graph
    if mode == "fallback_only":
        return np.where(graph_alpha >= 0.5, graph, feature)
    raise ValueError(f"unknown score mode: {mode}")


def _fit_alpha(val: pd.DataFrame) -> float:
    labels = val["y_true"].tolist()
    best_alpha = 0.5
    best_f1 = -1.0
    for alpha in np.linspace(0.0, 1.0, 11):
        scores = alpha * val["graph_score"].to_numpy(dtype=float) + (1 - alpha) * val["feature_score"].to_numpy(dtype=float)
        threshold, f1 = best_threshold(labels, scores)
        if f1 > best_f1:
            best_alpha = float(alpha)
            best_f1 = float(f1)
    return best_alpha


def _fit_reliability_cutoff(val: pd.DataFrame) -> Tuple[float, float]:
    if val.empty:
        return float("inf"), 0.5
    labels = val["y_true"].tolist()
    candidates = sorted(set(float(x) for x in val["reliability_risk"].quantile(np.linspace(0.1, 0.95, 18)).tolist()))
    candidates.append(float("inf"))
    best_cutoff = float("inf")
    best_thr = 0.5
    best_f1 = -1.0
    for cutoff in candidates:
        scores = _score_bundle(val, 0.5, cutoff, "graphsafe_tta_full")
        threshold, f1 = best_threshold(labels, scores)
        if f1 > best_f1:
            best_cutoff = cutoff
            best_thr = threshold
            best_f1 = f1
    return float(best_cutoff), float(best_thr)


def _fit_params(df: pd.DataFrame) -> Dict[str, Any]:
    train = df.loc[df["split"].astype(str) == "train"]
    val = df.loc[df["split"].astype(str) == "val"]
    if val.empty:
        val = train if not train.empty else df
    graph_alpha = _fit_alpha(val)
    graph_val = precision_recall_f1(val["y_true"].tolist(), val["graph_score"].tolist())
    feat_val = precision_recall_f1(val["y_true"].tolist(), val["feature_score"].tolist())
    best_val_branch = "graph" if graph_val["f1"] >= feat_val["f1"] else "feature"
    reliability_cutoff, graphsafe_threshold = _fit_reliability_cutoff(val)
    out = {
        "graph_alpha": graph_alpha,
        "best_val_branch": best_val_branch,
        "reliability_cutoff": reliability_cutoff,
        "graphsafe_threshold": graphsafe_threshold,
        "feature_threshold": best_threshold(val["y_true"].tolist(), val["feature_score"].tolist())[0],
        "graph_threshold": best_threshold(val["y_true"].tolist(), val["graph_score"].tolist())[0],
        "avg_threshold": best_threshold(val["y_true"].tolist(), (0.5 * val["feature_score"] + 0.5 * val["graph_score"]).tolist())[0],
        "weighted_threshold": best_threshold(
            val["y_true"].tolist(),
            (graph_alpha * val["graph_score"] + (1 - graph_alpha) * val["feature_score"]).tolist(),
        )[0],
        "graph_temperature": fit_temperature(val["y_true"].tolist(), val["graph_score"].tolist()),
        "feature_temperature": fit_temperature(val["y_true"].tolist(), val["feature_score"].tolist()),
        "train_prior": float((as_binary(train["y_true"].tolist()).mean() if not train.empty else as_binary(val["y_true"].tolist()).mean())),
        "val_prior": float(as_binary(val["y_true"].tolist()).mean()),
    }
    for budget in BUDGETS:
        out[f"budget_threshold_{budget:g}"] = _threshold_for_budget(
            _score_bundle(val, graph_alpha, reliability_cutoff, "graphsafe_tta_full"),
            budget,
        )
    return out


def _method_scores(df: pd.DataFrame, params: Mapping[str, Any], method: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    alpha = float(params["graph_alpha"])
    cutoff = float(params["reliability_cutoff"])
    extra: Dict[str, Any] = {}
    if method == "best_val_branch":
        branch = str(params["best_val_branch"])
        return df[f"{branch}_score"].to_numpy(dtype=float), float(params[f"{branch}_threshold"]), extra
    if method == "temperature_graph":
        scores = _sigmoid_logit_adjust(df["graph_score"].to_numpy(dtype=float), float(params["graph_temperature"]))
        return scores, float(params["graph_threshold"]), {"temperature": params["graph_temperature"]}
    if method == "prior_ratio_graph":
        target_prior = float(np.mean(df["graph_score"].to_numpy(dtype=float)))
        scores = score_prior_adjust(df["graph_score"].to_numpy(dtype=float), float(params["train_prior"]), target_prior)
        return scores, float(params["graph_threshold"]), {"target_prior_estimate": target_prior}
    if method == "tpc_tta_graph":
        temp = _sigmoid_logit_adjust(df["graph_score"].to_numpy(dtype=float), float(params["graph_temperature"]))
        target_prior = float(np.mean(temp))
        scores = score_prior_adjust(temp, float(params["train_prior"]), target_prior)
        return scores, float(params["graph_threshold"]), {"temperature": params["graph_temperature"], "target_prior_estimate": target_prior}
    if method == "threshold_tta_graph":
        return df["graph_score"].to_numpy(dtype=float), float(params["graph_threshold"]), extra
    mode = method
    if method == "graphsafe_tta_full":
        mode = "graphsafe_tta_full"
    scores = _score_bundle(df, alpha, cutoff, mode)
    thresholds = {
        "feature_only": "feature_threshold",
        "graph_only": "graph_threshold",
        "simple_average": "avg_threshold",
        "validation_weighted_ensemble": "weighted_threshold",
        "graphsafe_no_reliability": "weighted_threshold",
        "tta_only": "weighted_threshold",
        "graphsafe_tta_full": "graphsafe_threshold",
        "graphsafe_no_tta": "graphsafe_threshold",
        "graphsafe_no_fallback": "graph_threshold",
        "fallback_only": "graph_threshold" if float(params["graph_alpha"]) >= 0.5 else "feature_threshold",
        "reliability_estimator_only": "graphsafe_threshold",
    }
    return scores, float(params[thresholds[method]]), extra


def _rolling_thresholds(val: pd.DataFrame, test: pd.DataFrame, params: Mapping[str, Any], method: str, window: int) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    val_times = sorted(int(x) for x in val["timestep"].dropna().unique())
    if not val_times:
        return thresholds
    for t in sorted(int(x) for x in test["timestep"].dropna().unique()):
        sub = val.loc[(val["timestep"] >= t - window) & (val["timestep"] < t)]
        if sub.empty:
            sub = val.loc[val["timestep"].isin(val_times[-window:])]
        if sub.empty:
            sub = val
        scores, _, _ = _method_scores(sub, params, method)
        thresholds[t] = best_threshold(sub["y_true"].tolist(), scores)[0]
    return thresholds


def _overall_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> Dict[str, Any]:
    scores_arr = np.asarray(scores, dtype=float)
    labels_list = list(labels)
    base = precision_recall_f1(labels_list, scores_arr.tolist(), threshold)
    auroc = rank_auc(as_binary(labels_list), scores_arr.tolist())
    auprc = _average_precision(labels_list, scores_arr.tolist())
    p_at_k, r_at_k, k = _top_metrics(labels_list, scores_arr.tolist(), 0.01)
    out: Dict[str, Any] = {
        "f1": base["f1"],
        "precision": base["precision"],
        "recall": base["recall"],
        "auroc": auroc if auroc is not None else "",
        "auprc": auprc if auprc is not None else "",
        "ece": ece_score(labels_list, scores_arr.tolist()),
        "brier": brier_score(labels_list, scores_arr.tolist()),
        "threshold": threshold,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "k": k,
        "precision_at_1pct": p_at_k,
        "recall_at_1pct": r_at_k,
        "cost_sensitive_risk": _cost_risk(labels_list, scores_arr.tolist(), threshold, cost_fp, cost_fn),
        "n_eval": len(labels_list),
        "n_positive": int(as_binary(labels_list).sum()),
    }
    for budget in BUDGETS:
        p_b, r_b, k_b = _top_metrics(labels_list, scores_arr.tolist(), budget)
        pct = str(budget).replace(".", "_")
        out[f"precision_at_{pct}"] = p_b
        out[f"recall_at_{pct}"] = r_b
        out[f"k_at_{pct}"] = k_b
    return out


def _block_metrics(
    test: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    branch_scores: Mapping[str, np.ndarray],
    branch_thresholds: Mapping[str, float],
    *,
    cost_fp: float,
    cost_fn: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    y_all = test["y_true"].to_numpy()
    for t, idx in test.groupby("timestep").groups.items():
        pos = test.index.get_indexer(idx)
        y = y_all[pos]
        s = scores[pos]
        bm = _overall_metrics(y, s, threshold, cost_fp=cost_fp, cost_fn=cost_fn)
        branch_risks = {
            name: _cost_risk(y, bs[pos], branch_thresholds[name], cost_fp, cost_fn)
            for name, bs in branch_scores.items()
        }
        branch_f1s = {
            name: precision_recall_f1(y, bs[pos], branch_thresholds[name])["f1"]
            for name, bs in branch_scores.items()
        }
        rows.append(
            {
                "timestep": float(t),
                "f1": float(bm["f1"]),
                "recall_at_1pct": float(bm["recall_at_1pct"]),
                "cost_sensitive_risk": float(bm["cost_sensitive_risk"]),
                "cost_regret_vs_hindsight_best_branch": float(bm["cost_sensitive_risk"] - min(branch_risks.values())),
                "f1_regret_vs_hindsight_best_branch": float(max(branch_f1s.values()) - bm["f1"]),
            }
        )
    if not rows:
        return {
            "worst_block_regret": 0.0,
            "average_block_regret": 0.0,
            "worst_window_f1": 0.0,
            "worst_window_recall_at_k": 0.0,
            "worst_window_cost_sensitive_risk": 0.0,
        }
    return {
        "worst_block_regret": max(r["cost_regret_vs_hindsight_best_branch"] for r in rows),
        "average_block_regret": float(np.mean([r["cost_regret_vs_hindsight_best_branch"] for r in rows])),
        "worst_window_f1": min(r["f1"] for r in rows),
        "worst_window_recall_at_k": min(r["recall_at_1pct"] for r in rows),
        "worst_window_cost_sensitive_risk": max(r["cost_sensitive_risk"] for r in rows),
        "worst_block_f1_regret": max(r["f1_regret_vs_hindsight_best_branch"] for r in rows),
    }


def reliability_rows(df: pd.DataFrame, meta: Mapping[str, Any], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    val = df.loc[df["split"].astype(str) == "val"]
    val_g = val["graph_score"].tolist()
    val_f = val["feature_score"].tolist()
    cutoff = float(params["reliability_cutoff"])
    for (split, timestep), group in df.groupby(["split", "timestep"]):
        rows.append(
            {
                **meta,
                "split": split,
                "timestep": int(timestep),
                "n_nodes": int(len(group)),
                "mean_score_disagreement": float(group["score_disagreement"].mean()),
                "mean_rank_disagreement": float(group["rank_disagreement"].mean()),
                "mean_entropy": float(group["entropy_mean"].mean()),
                "mean_reliability_risk": float(group["reliability_risk"].mean()),
                "safe_graph_rate": float((group["reliability_risk"] <= cutoff).mean()),
                "ks_graph_vs_val": ks_distance(val_g, group["graph_score"].tolist()) if val_g else 0.0,
                "ks_feature_vs_val": ks_distance(val_f, group["feature_score"].tolist()) if val_f else 0.0,
                "reliability_cutoff": cutoff,
            }
        )
    return rows


def evaluate_pair(
    aligned: pd.DataFrame,
    *,
    dataset: str,
    protocol: str,
    graph_model: str,
    seed: int,
    cost_fp: float,
    cost_fn: float,
    window: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    df = add_reliability_columns(aligned)
    params = _fit_params(df)
    val = df.loc[df["split"].astype(str) == "val"]
    test = df.loc[df["split"].astype(str) == "test"]
    if test.empty:
        return [], [], [], params
    meta = {
        "dataset": dataset,
        "protocol": protocol,
        "graph_model": graph_model,
        "feature_model": "mlp",
        "seed": seed,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
    }
    methods = [
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
    ablations = [
        "graphsafe_no_reliability",
        "graphsafe_no_fallback",
        "graphsafe_no_tta",
        "reliability_estimator_only",
        "fallback_only",
        "tta_only",
    ]
    branch_feature = test["feature_score"].to_numpy(dtype=float)
    branch_graph = test["graph_score"].to_numpy(dtype=float)
    branch_thresholds = {
        "feature": float(params["feature_threshold"]),
        "graph": float(params["graph_threshold"]),
    }
    labels = test["y_true"].tolist()
    result_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    for method in methods + ablations:
        scores, threshold, extra = _method_scores(test, params, method)
        if method in {"threshold_tta_graph", "tpc_tta_graph", "graphsafe_tta_full", "tta_only"}:
            rolling = _rolling_thresholds(val, test, params, method, window)
            if rolling:
                preds_scores = scores.copy()
                threshold = float(np.mean(list(rolling.values())))
                extra["rolling_threshold_mean"] = threshold
                extra["rolling_threshold_min"] = float(min(rolling.values()))
                extra["rolling_threshold_max"] = float(max(rolling.values()))
                # Metrics below use the mean rolling threshold for score-level
                # comparability; review-budget metrics are threshold-free.
                scores = preds_scores
        metrics = _overall_metrics(labels, scores, threshold, cost_fp=cost_fp, cost_fn=cost_fn)
        block = _block_metrics(
            test,
            scores,
            threshold,
            {"feature": branch_feature, "graph": branch_graph},
            branch_thresholds,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
        row = {
            **meta,
            "method": method,
            "method_family": "graphsafe" if method.startswith("graphsafe") else "baseline",
            "graph_alpha": float(params["graph_alpha"]),
            "reliability_cutoff": float(params["reliability_cutoff"]),
            "best_val_branch": params["best_val_branch"],
            **metrics,
            **block,
            **extra,
            "selection_split": "train/val only",
            "evaluation_split": "test",
        }
        if method in ablations:
            row["ablation"] = method
            ablation_rows.append(row)
        else:
            result_rows.append(row)
    return result_rows, ablation_rows, reliability_rows(df, meta, params), params


def _compact_stat_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    stat_rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(["dataset", "protocol", "graph_model"]):
        base = group.loc[group["method"] == "best_val_branch"].set_index("seed")
        gs = group.loc[group["method"] == "graphsafe_tta_full"].set_index("seed")
        common = sorted(set(base.index) & set(gs.index))
        if len(common) < 2:
            continue
        for metric, direction in [
            ("f1", "higher"),
            ("recall_at_1pct", "higher"),
            ("cost_sensitive_risk", "lower"),
            ("worst_block_regret", "lower"),
            ("worst_window_cost_sensitive_risk", "lower"),
        ]:
            b = base.loc[common, metric].astype(float).to_numpy()
            g = gs.loc[common, metric].astype(float).to_numpy()
            improvement = (g - b) if direction == "higher" else (b - g)
            stat_rows.append(
                {
                    "dataset": keys[0],
                    "protocol": keys[1],
                    "graph_model": keys[2],
                    "comparison": "graphsafe_tta_full_vs_best_val_branch",
                    "metric": metric,
                    "direction": direction,
                    "baseline_mean": float(np.mean(b)),
                    "graphsafe_mean": float(np.mean(g)),
                    "mean_improvement": float(np.mean(improvement)),
                    "n_paired_seeds": int(len(common)),
                    "p_value": normal_approx_pvalue(improvement),
                    "win_label": "diagnostic_pending_correction",
                }
            )
    corrected = correction_rows(stat_rows)
    for row in corrected:
        if row.get("survives_holm") and float(row.get("mean_improvement", 0.0)) > 0:
            row["win_label"] = "holm_corrected_win"
        elif row.get("survives_fdr") and float(row.get("mean_improvement", 0.0)) > 0:
            row["win_label"] = "fdr_only_win"
        elif float(row.get("mean_improvement", 0.0)) > 0:
            row["win_label"] = "diagnostic_only_positive"
        else:
            row["win_label"] = "negative_or_no_gain"
    return corrected


def _source_file_manifest(display_paths: Iterable[str]) -> List[Dict[str, Any]]:
    files: Dict[str, Path] = {}
    for item in display_paths:
        source = item.split("::", 1)[0]
        files[source] = Path(source)
    rows: List[Dict[str, Any]] = []
    for source, path in sorted(files.items()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        rows.append(
            {
                "path": source,
                "size_bytes": size,
                "sha256": _sha256(path) if size <= 2_000_000_000 else "",
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GraphSafe-TTA over saved prediction artifacts.")
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--output-dir", default="results/runs_rb15_graphsafe_tta")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    p.add_argument("--graph-models", default=",".join(GRAPH_MODELS))
    p.add_argument("--seeds", default="")
    p.add_argument("--max-seeds", type=int, default=0)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--cost-fn", type=float, default=5.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = (root / args.output_dir).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    graph_models = [x.strip() for x in args.graph_models.split(",") if x.strip()]
    seeds_filter = {int(x) for x in args.seeds.split(",") if x.strip()} if args.seeds.strip() else set()
    sources = discover_sources(root)
    all_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    rel_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    used_sources: Dict[str, Dict[str, Any]] = {}

    for dataset in datasets:
        for protocol in protocols:
            available_seeds = sorted(
                seed
                for (ds, pr, model, seed) in sources
                if ds == dataset and pr == protocol and model == "mlp" and (not seeds_filter or seed in seeds_filter)
            )
            if args.max_seeds > 0:
                available_seeds = available_seeds[: args.max_seeds]
            for seed in available_seeds:
                try:
                    feature_src = sources[(dataset, protocol, "mlp", seed)]
                    feature_df = load_prediction(feature_src)
                    used_sources[feature_src.display_path] = {
                        "dataset": dataset,
                        "protocol": protocol,
                        "model": "mlp",
                        "seed": seed,
                        "size_bytes": feature_src.source_path.stat().st_size,
                        "sha256": _sha256(feature_src.source_path) if not feature_src.member else "",
                    }
                    graph_frames: Dict[str, pd.DataFrame] = {}
                    for graph_model in graph_models:
                        src = sources.get((dataset, protocol, graph_model, seed))
                        if src is None:
                            errors.append({"dataset": dataset, "protocol": protocol, "model": graph_model, "seed": seed, "error": "missing_prediction"})
                            continue
                        graph_frames[graph_model] = load_prediction(src)
                        used_sources[src.display_path] = {
                            "dataset": dataset,
                            "protocol": protocol,
                            "model": graph_model,
                            "seed": seed,
                            "size_bytes": src.source_path.stat().st_size,
                            "sha256": _sha256(src.source_path) if not src.member else "",
                        }
                    for graph_model, graph_df in graph_frames.items():
                        aligned = align_pair(feature_df, graph_df)
                        if aligned.empty:
                            errors.append({"dataset": dataset, "protocol": protocol, "model": graph_model, "seed": seed, "error": "empty_alignment"})
                            continue
                        rows, abl, rel, _ = evaluate_pair(
                            aligned,
                            dataset=dataset,
                            protocol=protocol,
                            graph_model=graph_model,
                            seed=seed,
                            cost_fp=float(args.cost_fp),
                            cost_fn=float(args.cost_fn),
                            window=int(args.window),
                        )
                        all_rows.extend(rows)
                        ablation_rows.extend(abl)
                        rel_rows.extend(rel)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"dataset": dataset, "protocol": protocol, "seed": seed, "error": repr(exc)})

    ensure_dir(out_dir)
    write_csv(out_dir / "graphsafe_tta_results.csv", all_rows)
    write_csv(out_dir / "graphsafe_tta_ablation_results.csv", ablation_rows)
    write_csv(out_dir / "graphsafe_tta_reliability_scores.csv", rel_rows)
    stats = _compact_stat_rows(all_rows)
    write_csv(out_dir / "graphsafe_tta_stat_tests.csv", stats)
    manifest = {
        "created_at_utc": utc_now(),
        "artifact_family": "RB15_graphsafe_tta",
        "mode": "saved_predictions_only_no_training",
        "selection_rule": "all thresholds, weights, cutoffs, and gates selected on train/validation rows only",
        "test_label_use": "final evaluation only",
        "n_result_rows": len(all_rows),
        "n_ablation_rows": len(ablation_rows),
        "n_reliability_rows": len(rel_rows),
        "n_stat_rows": len(stats),
        "n_errors": len(errors),
        "source_files": _source_file_manifest(used_sources.keys()),
        "inputs": sorted(used_sources.values(), key=lambda r: (r["dataset"], r["protocol"], r["model"], r["seed"])),
        "prediction_locations": sorted(used_sources.keys()),
        "errors": errors,
    }
    write_json(out_dir / "import_manifest.json", manifest)
    print(f"[graphsafe-tta] results={len(all_rows)} ablations={len(ablation_rows)} reliability={len(rel_rows)} errors={len(errors)}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
