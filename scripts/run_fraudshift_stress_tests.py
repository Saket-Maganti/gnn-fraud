#!/usr/bin/env python3
"""FraudShiftBench CPU-only stress-test simulator.

The simulator perturbs saved prediction scores from existing RB02 artifacts.
It does not train, import new result families, call Kaggle, or overwrite raw
prediction exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.io import parse_prediction_filename  # noqa: E402
from fraudshiftbench.metrics import (  # noqa: E402
    fraud_recall_at_budget,
    rank_reversal_score,
)

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 2
STRESSORS = (
    "class_prior_shift",
    "score_calibration_drift",
    "threshold_drift",
    "review_budget_halved",
    "graph_signal_decay_proxy",
    "label_delay_late_period",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure(path.parent)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def simulation_manifest() -> Dict[str, Any]:
    """Return invariant safety metadata used by tests and reports."""

    return {
        "framework": "FraudShiftBench",
        "simulation_only": True,
        "trained_models": False,
        "gpu_used": False,
        "kaggle_used": False,
        "raw_predictions_overwritten": False,
        "claim_level": "sensitivity",
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibration_drift_scores(scores: Sequence[float], *, scale: float = 0.75, bias: float = -0.15) -> np.ndarray:
    """Apply a deterministic logit-space calibration drift to saved scores."""

    arr = np.asarray(scores, dtype=float)
    arr = np.clip(arr, 1e-6, 1 - 1e-6)
    logits = np.log(arr / (1.0 - arr))
    return _sigmoid(scale * logits + bias)


def weighted_f1(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Weighted positive-class F1 for Elliptic label convention."""

    y = np.asarray([1 if int(v) == POSITIVE_LABEL else 0 for v in labels], dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = float(w[(y == 1) & (pred == 1)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom <= 0 else (2 * tp) / denom


def apply_stressor(
    frame: pd.DataFrame,
    stressor: str,
    *,
    model_harm_rates: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute baseline and stressed utility for one saved prediction frame."""

    if stressor not in STRESSORS:
        raise ValueError(f"Unknown stressor: {stressor}")
    test = frame.loc[frame.get("split", "test").astype(str) == "test"].copy()
    if test.empty:
        test = frame.copy()
    test = test.loc[test["y_true"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    if test.empty:
        return {"baseline_metric": 0.0, "stressed_metric": 0.0, "metric_delta": 0.0}
    labels = test["y_true"].astype(int).tolist()
    scores = test["score"].astype(float).clip(0, 1).to_numpy()
    model = str(test["model"].iloc[0]) if "model" in test else "unknown"

    if stressor == "review_budget_halved":
        baseline = fraud_recall_at_budget(labels, scores, 1000)
        stressed = fraud_recall_at_budget(labels, scores, 500)
    elif stressor == "class_prior_shift":
        baseline = weighted_f1(labels, scores)
        weights = [1.5 if int(y) == POSITIVE_LABEL else 0.75 for y in labels]
        stressed = weighted_f1(labels, scores, weights=weights)
    elif stressor == "score_calibration_drift":
        baseline = weighted_f1(labels, scores)
        stressed = weighted_f1(labels, calibration_drift_scores(scores))
    elif stressor == "threshold_drift":
        baseline = weighted_f1(labels, scores)
        stressed = weighted_f1(labels, scores, threshold=0.6)
    elif stressor == "graph_signal_decay_proxy":
        baseline = weighted_f1(labels, scores)
        harm_rate = float((model_harm_rates or {}).get(model, 0.0))
        shrink = max(0.5, 1.0 - min(0.5, harm_rate * 5.0))
        stressed_scores = 0.5 + (scores - 0.5) * shrink
        stressed = weighted_f1(labels, stressed_scores)
    else:
        baseline = weighted_f1(labels, scores)
        if "timestep" in test:
            late = test.loc[test["timestep"].astype(float) >= 45]
            if late.empty:
                cutoff = float(test["timestep"].astype(float).quantile(0.75))
                late = test.loc[test["timestep"].astype(float) >= cutoff]
        else:
            late = test.tail(max(1, len(test) // 4))
        stressed = weighted_f1(late["y_true"].astype(int).tolist(), late["score"].astype(float).tolist())

    return {
        "baseline_metric": float(baseline),
        "stressed_metric": float(stressed),
        "metric_delta": float(stressed - baseline),
    }


def _load_harm_rates(root: Path) -> Dict[str, float]:
    path = root / "results" / "runs" / "rb01_graph_harm" / "graph_harm_by_model.csv"
    if not path.is_file():
        return {}
    rows = pd.read_csv(path)
    if "gnn_model" not in rows or "harm_rate" not in rows:
        return {}
    return dict(zip(rows["gnn_model"].astype(str), rows["harm_rate"].astype(float)))


def _prediction_files(root: Path) -> List[Path]:
    pred_dir = root / "results" / "runs" / "predictions"
    if not pred_dir.is_dir():
        return []
    return sorted(pred_dir.glob("*.csv"))


def run_stress_tests(root: Path) -> Dict[str, Any]:
    harm_rates = _load_harm_rates(root)
    result_rows: List[Dict[str, Any]] = []
    for path in _prediction_files(root):
        parsed = parse_prediction_filename(path)
        if parsed is None:
            continue
        dataset, protocol, model, seed = parsed
        frame = pd.read_csv(path, usecols=lambda c: c in {"split", "y_true", "score", "model", "timestep"})
        for stressor in STRESSORS:
            metrics = apply_stressor(frame, stressor, model_harm_rates=harm_rates)
            result_rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "model": model,
                    "seed": seed,
                    "stressor": stressor,
                    **metrics,
                    "fragility": max(0.0, -float(metrics["metric_delta"])),
                    "source_prediction_file": str(path.relative_to(root)),
                    "simulation_only": True,
                }
            )

    result_df = pd.DataFrame(result_rows)
    if result_df.empty:
        summary_df = pd.DataFrame()
        rank_rows: List[Dict[str, Any]] = []
    else:
        summary_df = (
            result_df.groupby(["stressor", "dataset", "protocol", "model"], as_index=False)
            .agg(
                baseline_metric=("baseline_metric", "mean"),
                stressed_metric=("stressed_metric", "mean"),
                metric_delta=("metric_delta", "mean"),
                fragility=("fragility", "mean"),
                n_prediction_sets=("seed", "count"),
            )
            .sort_values(["stressor", "protocol", "fragility"], ascending=[True, True, False])
        )
        rank_rows = []
        for (stressor, protocol), group in summary_df.groupby(["stressor", "protocol"]):
            baseline = dict(zip(group["model"], group["baseline_metric"]))
            stressed = dict(zip(group["model"], group["stressed_metric"]))
            instability = rank_reversal_score(baseline, stressed)
            rank_rows.append(
                {
                    "stressor": stressor,
                    "protocol": protocol,
                    "rank_instability": 0.0 if instability is None else instability,
                    "n_models": int(group["model"].nunique()),
                    "simulation_only": True,
                }
            )

    return {
        "manifest": {
            **simulation_manifest(),
            "created_at_utc": utc_now(),
            "stressors": list(STRESSORS),
            "source_prediction_dir": "results/runs/predictions",
            "source_harm_proxy": "results/runs/rb01_graph_harm/graph_harm_by_model.csv",
        },
        "results": result_rows,
        "summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
        "rank_instability": rank_rows,
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = pd.DataFrame(payload.get("summary", []))
    rank = pd.DataFrame(payload.get("rank_instability", []))
    lines = [
        "# FraudShiftBench Stress-Test Report",
        "",
        "This report is generated from existing prediction outputs only. It is a sensitivity simulator, not new empirical training.",
        "",
        "## Claim Boundary",
        "",
        "- Evidence level: `sensitivity`.",
        "- No GPU, Kaggle, training, or raw prediction overwrite is used.",
        "- Stressors perturb saved scores, thresholds, review budgets, or evaluation subsets.",
        "",
        "## Highest Fragility Rows",
        "",
    ]
    if summary.empty:
        lines.append("_No stress-test rows were generated._")
    else:
        top = summary.sort_values("fragility", ascending=False).head(8)
        lines.append("| Stressor | Protocol | Model | Fragility | Delta |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in top.to_dict(orient="records"):
            lines.append(
                f"| {row['stressor']} | {row['protocol']} | {row['model']} | "
                f"{float(row['fragility']):.4f} | {float(row['metric_delta']):.4f} |"
            )
    lines.extend(["", "## Rank Instability", ""])
    if rank.empty:
        lines.append("_No rank-instability rows were generated._")
    else:
        lines.append("| Stressor | Protocol | Rank instability |")
        lines.append("| --- | --- | --- |")
        for row in rank.sort_values("rank_instability", ascending=False).head(10).to_dict(orient="records"):
            lines.append(f"| {row['stressor']} | {row['protocol']} | {float(row['rank_instability']):.4f} |")
    lines.extend(
        [
            "",
            "## Narrative Use",
            "",
            "FraudShiftBench turns the saved benchmark into a reusable stress harness: a reader can ask which saved models are fragile under class-prior shift, calibration drift, review-budget changes, graph-signal decay proxies, or late-period evaluation. The language remains sensitivity-level unless future runs validate the same behavior under new empirical experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(summary_rows: Sequence[Mapping[str, Any]], figure_base: Path) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows)
    pivot = df.groupby(["stressor", "model"])["fragility"].mean().unstack(fill_value=0.0)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("FraudShiftBench simulated fragility from saved RB02 predictions")
    ax.set_xlabel("Model")
    ax.set_ylabel("Stressor")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean metric drop (positive fragility)")
    fig.tight_layout()
    _ensure(figure_base.parent)
    fig.savefig(figure_base.with_suffix(".png"), dpi=220)
    fig.savefig(figure_base.with_suffix(".pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CPU-only FraudShiftBench stress simulations.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "runs" / "fraudshift_stress_tests"))
    parser.add_argument("--figure-base", default=str(REPO_ROOT / "figures" / "fraudshift_stress_tests"))
    parser.add_argument("--report", default=str(REPO_ROOT / "aaai_upgrade" / "FRAUDSHIFT_STRESS_TEST_REPORT.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    payload = run_stress_tests(root)
    out_dir = Path(args.output_dir)
    _ensure(out_dir)
    _write_json(out_dir / "manifest.json", payload["manifest"])
    _write_csv(out_dir / "stress_test_results.csv", payload["results"])
    _write_csv(out_dir / "stress_test_summary.csv", payload["summary"])
    _write_csv(out_dir / "rank_instability.csv", payload["rank_instability"])
    _write_json(
        out_dir / "stress_test_summary.json",
        {
            "created_at_utc": utc_now(),
            "manifest": payload["manifest"],
            "n_result_rows": len(payload["results"]),
            "n_summary_rows": len(payload["summary"]),
            "rank_instability": payload["rank_instability"],
        },
    )
    _plot(payload["summary"], Path(args.figure_base))
    report_path = Path(args.report)
    _ensure(report_path.parent)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(f"[fraudshift-stress] rows={len(payload['results'])} summary={len(payload['summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
