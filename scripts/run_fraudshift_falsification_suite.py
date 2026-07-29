#!/usr/bin/env python3
"""FraudShiftBench falsification and negative-control suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.metrics import rank_reversal_score  # noqa: E402


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


def rank_risk_from_runs(frame: pd.DataFrame, protocol_a: str, protocol_b: str) -> float:
    rows = frame.loc[frame["protocol"].isin([protocol_a, protocol_b])].copy()
    means = rows.groupby(["protocol", "model"])["f1"].mean().unstack(0)
    if protocol_a not in means or protocol_b not in means:
        return 0.0
    score = rank_reversal_score(means[protocol_a].dropna().to_dict(), means[protocol_b].dropna().to_dict())
    return 0.0 if score is None else float(score)


def shuffle_column(frame: pd.DataFrame, column: str, *, seed: int = 13) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(seed)
    out[column] = rng.permutation(out[column].to_numpy())
    return out


def rank_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    sorted_scores = s[order]
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def graph_harm_shuffle_auc(features: pd.DataFrame, *, seed: int = 17) -> Dict[str, float]:
    y = (features["harm_label"].astype(str) == "graph_harm").astype(int).to_numpy()
    scores = features["training_free_harm_risk_score"].astype(float).to_numpy()
    original = rank_auc(y, scores)
    rng = np.random.default_rng(seed)
    shuffled = rank_auc(rng.permutation(y), scores)
    return {
        "original_auc": 0.0 if original is None else float(original),
        "shuffled_auc": 0.0 if shuffled is None else float(shuffled),
        "auc_drop": (0.0 if original is None else float(original)) - (0.0 if shuffled is None else float(shuffled)),
    }


def _load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for rel in [
        "results/runs/matched_gnn_protocol/runs.csv",
        "results/runs/multi_dataset_protocol/runs.csv",
    ]:
        path = root / rel
        if path.is_file():
            frame = pd.read_csv(path)
            frame["source_artifact"] = rel
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_harm_features(root: Path) -> pd.DataFrame:
    path = root / "results" / "runs" / "rb01_graph_harm" / "harm_predictor_features.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(
        path,
        usecols=["harm_label", "training_free_harm_risk_score"],
    ).sample(n=200000, random_state=23)


def run_falsification(root: Path) -> Dict[str, Any]:
    runs = _load_runs(root)
    results: List[Dict[str, Any]] = []
    protocol_pairs = [
        ("random", "chronological", "RB02_matched_protocol"),
        ("transductive", "strict_inductive", "RB02_multi_protocol"),
    ]
    if not runs.empty:
        for optimistic, faithful, family in protocol_pairs:
            original = rank_risk_from_runs(runs, optimistic, faithful)
            shuffled_model = rank_risk_from_runs(shuffle_column(runs, "model", seed=41), optimistic, faithful)
            shuffled_seed = rank_risk_from_runs(shuffle_column(runs, "seed", seed=43), optimistic, faithful)
            randomized_protocol = rank_risk_from_runs(shuffle_column(runs, "protocol", seed=47), optimistic, faithful)
            results.extend(
                [
                    {
                        "control": "shuffle_model_labels",
                        "artifact_family": family,
                        "original_metric": original,
                        "falsified_metric": shuffled_model,
                        "interpretation": "Model identity negative control changes leaderboard-risk semantics.",
                    },
                    {
                        "control": "shuffle_seed_labels",
                        "artifact_family": family,
                        "original_metric": original,
                        "falsified_metric": shuffled_seed,
                        "interpretation": "Seed-label shuffle should not create new protocol coverage.",
                    },
                    {
                        "control": "randomize_protocol_labels",
                        "artifact_family": family,
                        "original_metric": original,
                        "falsified_metric": randomized_protocol,
                        "interpretation": "Protocol-label randomization is expected to be unstable or nonsensical.",
                    },
                ]
            )

    pred_files = sorted((root / "results" / "runs" / "predictions").glob("*.csv"))
    if pred_files:
        sample = pd.read_csv(pred_files[0], usecols=lambda c: c in {"split", "timestep", "y_true"})
        sample = sample.loc[sample["split"].astype(str) == "test"].copy()
        original_prior_by_time = sample.groupby("timestep")["y_true"].apply(lambda x: (x == 1).mean())
        shuffled = sample.copy()
        shuffled["timestep"] = np.random.default_rng(53).permutation(shuffled["timestep"].to_numpy())
        shuffled_prior_by_time = shuffled.groupby("timestep")["y_true"].apply(lambda x: (x == 1).mean())
        results.append(
            {
                "control": "destroy_timestep_order",
                "artifact_family": "RB02_predictions",
                "original_metric": float(original_prior_by_time.std(ddof=0)),
                "falsified_metric": float(shuffled_prior_by_time.std(ddof=0)),
                "interpretation": "Temporal-order destruction targets class-prior drift structure.",
            }
        )

    harm = _load_harm_features(root)
    if not harm.empty:
        aucs = graph_harm_shuffle_auc(harm)
        results.append(
            {
                "control": "graph_harm_label_shuffle",
                "artifact_family": "RB01",
                "original_metric": aucs["original_auc"],
                "falsified_metric": aucs["shuffled_auc"],
                "interpretation": "Graph-harm predictability should degrade when diagnostic labels are shuffled.",
            }
        )

    return {
        "manifest": {
            "created_at_utc": utc_now(),
            "framework": "FraudShiftBench",
            "negative_controls_only": True,
            "trained_models": False,
            "gpu_used": False,
            "raw_predictions_overwritten": False,
        },
        "results": results,
    }


def _render_report(payload: Mapping[str, Any]) -> str:
    rows = payload.get("results", [])
    lines = [
        "# FraudShiftBench Falsification Suite Report",
        "",
        "This suite runs negative controls over existing artifacts to check whether the framework behaves sensibly when model, seed, protocol, timestep, or harm labels are disrupted.",
        "",
        "## Claim Boundary",
        "",
        "- These are negative controls, not new training results.",
        "- Passing this suite supports the framework's auditability, not a universal model claim.",
        "",
        "## Controls",
        "",
    ]
    if not rows:
        lines.append("_No controls were generated._")
    else:
        lines.append("| Control | Artifact family | Original | Falsified | Interpretation |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['control']} | {row['artifact_family']} | {float(row['original_metric']):.4f} | "
                f"{float(row['falsified_metric']):.4f} | {row['interpretation']} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FraudShiftBench falsification controls.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "runs" / "falsification_suite"))
    parser.add_argument("--report", default=str(REPO_ROOT / "aaai_upgrade" / "FALSIFICATION_SUITE_REPORT.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_falsification(Path(args.root).resolve())
    out_dir = Path(args.output_dir)
    _ensure(out_dir)
    _write_json(out_dir / "falsification_summary.json", payload)
    _write_csv(out_dir / "falsification_results.csv", payload["results"])
    report = Path(args.report)
    _ensure(report.parent)
    report.write_text(_render_report(payload), encoding="utf-8")
    print(f"[falsification] controls={len(payload['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
