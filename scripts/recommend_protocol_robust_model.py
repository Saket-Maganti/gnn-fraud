#!/usr/bin/env python3
"""FraudShiftBench protocol-robust model recommender.

This is decision support over existing RB01/RB02 summaries. It does not train a
model and does not claim deployment performance beyond the saved artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OBJECTIVES = (
    "maximize_f1",
    "maximize_recall_at_budget",
    "minimize_false_positives",
    "minimize_protocol_regret",
    "prioritize_robustness",
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


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _faithful_protocols(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "protocol" not in frame:
        return frame
    return frame.loc[frame["protocol"].isin(["strict_inductive", "chronological"])].copy()


def _select_f1_model(multi: pd.DataFrame, matched: pd.DataFrame) -> str:
    frames = [_faithful_protocols(x) for x in (multi, matched) if not x.empty]
    if not frames:
        return "unavailable"
    rows = pd.concat(frames, ignore_index=True)
    grouped = rows.groupby("model")["f1"].mean().sort_values(ascending=False)
    return str(grouped.index[0]) if not grouped.empty else "unavailable"


def _select_recall_model(utility: pd.DataFrame, budget: str) -> str:
    if utility.empty:
        return "unavailable"
    rows = utility.loc[
        (utility["budget"].astype(str) == budget)
        & (utility["protocol"].astype(str).isin(["strict_inductive", "chronological"]))
    ].copy()
    if rows.empty:
        rows = utility.loc[utility["budget"].astype(str) == budget].copy()
    grouped = rows.groupby("model")["fraud_recall_at_budget"].mean().sort_values(ascending=False)
    return str(grouped.index[0]) if not grouped.empty else "unavailable"


def _select_low_fp_model(review: pd.DataFrame, budget: str) -> str:
    if review.empty:
        return "unavailable"
    rows = review.loc[
        (review["budget"].astype(str) == budget)
        & (review["protocol"].astype(str).isin(["strict_inductive", "chronological"]))
    ].copy()
    if rows.empty:
        rows = review.loc[review["budget"].astype(str) == budget].copy()
    grouped = rows.groupby("model")["false_positives"].mean().sort_values(ascending=True)
    return str(grouped.index[0]) if not grouped.empty else "unavailable"


def _select_robust_model(selection: pd.DataFrame) -> str:
    if selection.empty:
        return "unavailable"
    rows = selection.sort_values(["robust_worst_case_regret", "robust_average_regret"], ascending=True)
    return str(rows.iloc[0]["robust_model"])


def unsafe_protocols(protocol_risk: pd.DataFrame, *, threshold: float = 0.2) -> List[Dict[str, Any]]:
    if protocol_risk.empty or "protocol_risk_index" not in protocol_risk:
        return []
    rows = protocol_risk.loc[protocol_risk["protocol_risk_index"].astype(float) >= threshold].copy()
    out: List[Dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        out.append(
            {
                "artifact_family": row.get("artifact_family"),
                "optimistic_protocol": row.get("optimistic_protocol"),
                "faithful_protocol": row.get("faithful_protocol"),
                "protocol_risk_index": float(row.get("protocol_risk_index", 0.0)),
                "highest_risk_component": row.get("highest_risk_component"),
            }
        )
    return out


def optimistic_selection_regret(selection: pd.DataFrame) -> List[Dict[str, Any]]:
    if selection.empty:
        return []
    cols = [
        "artifact_family",
        "optimistic_protocol",
        "faithful_protocol",
        "optimistic_winner",
        "faithful_winner",
        "optimistic_winner_worst_case_regret",
        "optimistic_winner_average_regret",
    ]
    rows: List[Dict[str, Any]] = []
    for row in selection[cols].to_dict(orient="records"):
        rows.append({k: (float(v) if str(k).endswith("_regret") else v) for k, v in row.items()})
    return rows


def recommend(
    objective: str,
    *,
    budget: str,
    multi: pd.DataFrame,
    matched: pd.DataFrame,
    selection: pd.DataFrame,
    utility_rankings: pd.DataFrame,
    review_metrics: pd.DataFrame,
    protocol_risk: pd.DataFrame,
) -> Dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "maximize_f1":
        model = _select_f1_model(multi, matched)
    elif objective == "maximize_recall_at_budget":
        model = _select_recall_model(utility_rankings, budget)
    elif objective == "minimize_false_positives":
        model = _select_low_fp_model(review_metrics, budget)
    else:
        model = _select_robust_model(selection)
    return {
        "objective": objective,
        "budget": budget,
        "recommended_model": model,
        "evidence_level": "supported_elliptic_decision_support",
        "decision_boundary": "Decision support over existing RB01/RB02 artifacts; not a new trained model.",
        "unsafe_protocols": unsafe_protocols(protocol_risk),
        "optimistic_selection_regret": optimistic_selection_regret(selection),
    }


def load_inputs(root: Path) -> Dict[str, pd.DataFrame]:
    return {
        "multi": _load(root / "results" / "runs" / "multi_dataset_protocol" / "runs.csv"),
        "matched": _load(root / "results" / "runs" / "matched_gnn_protocol" / "runs.csv"),
        "selection": _load(root / "results" / "runs" / "protocol_robust_selection_v2" / "selection_summary.csv"),
        "utility_rankings": _load(root / "results" / "runs" / "fraud_review_utility" / "utility_rankings.csv"),
        "review_metrics": _load(root / "results" / "runs" / "fraud_review_utility" / "review_budget_metrics.csv"),
        "protocol_risk": _load(root / "results" / "runs" / "protocol_risk_index" / "protocol_risk_index.csv"),
    }


def _render_report(recommendations: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Protocol Decision Policy Report",
        "",
        "FraudShiftBench turns the saved benchmark into a policy recommender for model selection under temporal protocol risk.",
        "",
        "## Claim Boundary",
        "",
        "- This is decision support over existing RB01/RB02 artifacts.",
        "- It does not train a new model or claim second-dataset confirmation.",
        "- Protocols flagged as unsafe are high-risk within the saved Elliptic artifact families.",
        "",
        "## Recommendations",
        "",
        "| Objective | Budget | Recommended model | Evidence level |",
        "| --- | --- | --- | --- |",
    ]
    for row in recommendations:
        lines.append(
            f"| {row['objective']} | {row['budget']} | {row['recommended_model']} | {row['evidence_level']} |"
        )
    lines.extend(["", "## Unsafe Protocol Comparisons", ""])
    unsafe = recommendations[0].get("unsafe_protocols", []) if recommendations else []
    if not unsafe:
        lines.append("_No protocol-risk row crossed the configured threshold._")
    else:
        lines.append("| Artifact family | Optimistic protocol | Faithful protocol | Risk index | Driver |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in unsafe:
            lines.append(
                f"| {row['artifact_family']} | {row['optimistic_protocol']} | {row['faithful_protocol']} | "
                f"{float(row['protocol_risk_index']):.4f} | {row['highest_risk_component']} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend a protocol-robust model from saved artifacts.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--objective", choices=[*OBJECTIVES, "all"], default="all")
    parser.add_argument("--budget", default="top_1000")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "runs" / "protocol_decision_policy"))
    parser.add_argument("--report", default=str(REPO_ROOT / "aaai_upgrade" / "PROTOCOL_DECISION_POLICY_REPORT.md"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    inputs = load_inputs(root)
    objectives = list(OBJECTIVES) if args.objective == "all" else [args.objective]
    recommendations = [
        recommend(objective, budget=args.budget, **inputs)
        for objective in objectives
    ]
    out_dir = Path(args.output_dir)
    _ensure(out_dir)
    _write_json(
        out_dir / "recommendations.json",
        {"created_at_utc": utc_now(), "framework": "FraudShiftBench", "recommendations": recommendations},
    )
    flat_rows = [
        {
            "objective": row["objective"],
            "budget": row["budget"],
            "recommended_model": row["recommended_model"],
            "evidence_level": row["evidence_level"],
        }
        for row in recommendations
    ]
    _write_csv(out_dir / "recommendations.csv", flat_rows)
    _write_csv(out_dir / "unsafe_protocols.csv", recommendations[0].get("unsafe_protocols", []) if recommendations else [])
    _write_csv(
        out_dir / "optimistic_selection_regret.csv",
        recommendations[0].get("optimistic_selection_regret", []) if recommendations else [],
    )
    report = Path(args.report)
    _ensure(report.parent)
    report.write_text(_render_report(recommendations), encoding="utf-8")
    print(f"[protocol-policy] recommendations={len(recommendations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
