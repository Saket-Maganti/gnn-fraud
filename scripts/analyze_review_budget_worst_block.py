#!/usr/bin/env python3
"""Analyze RB17 review-budget and worst-block deployment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.result_analysis import ensure_dir, markdown_table, utc_now, write_csv, write_tex  # noqa: E402
from scripts.graphsafe_strengthening_common import paired_tests  # noqa: E402

OUT_DIR = "results/runs_rb17_review_budget_worst_block"
FOCUS_METHODS = [
    "graphsafe_tta_full",
    "graphsafe_conservative",
    "graphsafe_high_risk_only",
    "graphsafe_dgraphfin_risk_focused",
    "no_prior_graphsafe",
    "no_rank_claim_graphsafe",
    "reliability_weighted_branch_selector",
]
COMPARATORS = ["best_val_branch", "tpc_tta_graph", "feature_only", "graph_only", "simple_average"]
METRICS = [
    "precision_at_0_5pct",
    "recall_at_0_5pct",
    "precision_at_1pct",
    "recall_at_1pct",
    "precision_at_2pct",
    "recall_at_2pct",
    "cost_sensitive_risk",
    "worst_block_regret",
    "average_block_regret",
    "worst_window_f1",
    "worst_window_recall_at_1pct",
    "worst_window_cost_sensitive_risk",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze RB17 review-budget/worst-block artifacts.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    return parser


def _mean_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keep = set(FOCUS_METHODS + ["best_val_branch", "tpc_tta_graph", "feature_only", "graph_only", "simple_average"])
    sub = df.loc[df["method"].astype(str).isin(keep)]
    for (dataset, method), group in sub.groupby(["dataset", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": int(len(group)),
                "precision_at_0_5pct": float(pd.to_numeric(group["precision_at_0_5pct"], errors="coerce").mean()),
                "recall_at_0_5pct": float(pd.to_numeric(group["recall_at_0_5pct"], errors="coerce").mean()),
                "precision_at_1pct": float(pd.to_numeric(group["precision_at_1pct"], errors="coerce").mean()),
                "recall_at_1pct": float(pd.to_numeric(group["recall_at_1pct"], errors="coerce").mean()),
                "precision_at_2pct": float(pd.to_numeric(group["precision_at_2pct"], errors="coerce").mean()),
                "recall_at_2pct": float(pd.to_numeric(group["recall_at_2pct"], errors="coerce").mean()),
                "cost_sensitive_risk": float(pd.to_numeric(group["cost_sensitive_risk"], errors="coerce").mean()),
                "worst_block_regret": float(pd.to_numeric(group["worst_block_regret"], errors="coerce").mean()),
                "worst_window_recall_at_1pct": float(pd.to_numeric(group["worst_window_recall_at_1pct"], errors="coerce").mean()),
                "worst_window_cost_sensitive_risk": float(pd.to_numeric(group["worst_window_cost_sensitive_risk"], errors="coerce").mean()),
            }
        )
    return sorted(rows, key=lambda r: (r["dataset"], r["method"]))


def _cost_table(cost: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keep = set(FOCUS_METHODS + ["best_val_branch", "tpc_tta_graph", "feature_only", "graph_only", "simple_average"])
    sub = cost.loc[cost["method"].astype(str).isin(keep)]
    for (dataset, method, ratio), group in sub.groupby(["dataset", "method", "cost_ratio_fn_to_fp"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "cost_ratio_fn_to_fp": ratio,
                "n": int(len(group)),
                "reconstructed_cost_sensitive_risk": float(pd.to_numeric(group["reconstructed_cost_sensitive_risk"], errors="coerce").mean()),
            }
        )
    return sorted(rows, key=lambda r: (r["dataset"], r["method"], r["cost_ratio_fn_to_fp"]))


def _write_figures(root: Path, budget_rows: Sequence[Mapping[str, Any]], worst_rows: Sequence[Mapping[str, Any]], cost_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    ensure_dir(fig_dir)
    budget = pd.DataFrame(budget_rows)
    focus_methods = ["best_val_branch", "graphsafe_conservative", "no_prior_graphsafe", "reliability_weighted_branch_selector"]
    focus = budget.loc[budget["method"].isin(focus_methods)].copy()
    if not focus.empty:
        labels = [f"{r.dataset}:{r.method}" for r in focus.itertuples()]
        plt.figure(figsize=(max(9, len(labels) * 0.35), 4))
        plt.bar(range(len(focus)), focus["recall_at_1pct"])
        plt.xticks(range(len(focus)), labels, rotation=65, ha="right", fontsize=7)
        plt.ylabel("Recall@1%")
        plt.title("RB17 review-budget recall")
        plt.tight_layout()
        plt.savefig(fig_dir / "review_budget_recall_precision.png", dpi=160)
        plt.savefig(fig_dir / "review_budget_recall_precision.pdf")
        plt.close()
    worst = pd.DataFrame(worst_rows)
    focus = worst.loc[worst["method"].isin(focus_methods)].copy()
    if not focus.empty:
        labels = [f"{r.dataset}:{r.protocol}:{r.graph_model}:{r.method}" for r in focus.itertuples()]
        plt.figure(figsize=(max(10, len(labels) * 0.24), 4))
        plt.bar(range(len(focus)), focus["worst_block_regret"])
        plt.xticks(range(len(focus)), labels, rotation=70, ha="right", fontsize=6)
        plt.ylabel("Worst-block regret")
        plt.title("RB17 worst-block regret comparison")
        plt.tight_layout()
        plt.savefig(fig_dir / "worst_block_regret_comparison.png", dpi=160)
        plt.savefig(fig_dir / "worst_block_regret_comparison.pdf")
        plt.close()
    cost = pd.DataFrame(cost_rows)
    focus = cost.loc[cost["method"].isin(["best_val_branch", "graphsafe_conservative", "no_prior_graphsafe"])].copy()
    if not focus.empty:
        plt.figure(figsize=(7, 4))
        for (dataset, method), group in focus.groupby(["dataset", "method"]):
            group = group.sort_values("cost_ratio_fn_to_fp")
            plt.plot(group["cost_ratio_fn_to_fp"], group["reconstructed_cost_sensitive_risk"], marker="o", label=f"{dataset}:{method}")
        plt.xlabel("FN:FP cost ratio")
        plt.ylabel("Reconstructed risk")
        plt.title("RB17 cost-sensitive risk sensitivity")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(fig_dir / "cost_sensitive_risk_sensitivity.png", dpi=160)
        plt.savefig(fig_dir / "cost_sensitive_risk_sensitivity.pdf")
        plt.close()


def _write_report(root: Path, stats: pd.DataFrame, budget_rows: Sequence[Mapping[str, Any]], worst_rows: Sequence[Mapping[str, Any]], cost_rows: Sequence[Mapping[str, Any]]) -> None:
    labels = stats["win_label"].value_counts().to_dict() if not stats.empty else {}
    best = stats.loc[(stats["comparator"] == "best_val_branch") & (stats["method"].isin(FOCUS_METHODS))]
    best_labels = best["win_label"].value_counts().to_dict() if not best.empty else {}
    corrected = best.loc[best["win_label"].isin(["holm_corrected_win", "fdr_only_win"])] if not best.empty else pd.DataFrame()
    risk_metrics = {"cost_sensitive_risk", "worst_block_regret", "recall_at_1pct", "worst_window_cost_sensitive_risk"}
    corrected_risk = corrected.loc[corrected["metric"].isin(risk_metrics)] if not corrected.empty else pd.DataFrame()
    both_dataset = set(corrected_risk.get("dataset", pd.Series(dtype=str))) >= {"elliptic", "dgraphfin"}
    if both_dataset:
        answer = "Yes, but only with the exact corrected rows listed below."
    elif not corrected_risk.empty:
        answer = "Partially: corrected risk/review-budget evidence exists, but not across both datasets."
    else:
        answer = "No: the review-budget/worst-block method claim remains weak against the best validation-selected branch."
    report = [
        "# RB17 Review-Budget And Worst-Block Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "Cost-sensitive risk formula: `(false_positive_cost * FP + false_negative_cost * FN) / n_eval`.",
        "Cost sensitivity is reconstructed from saved precision, recall, n_positive, and n_eval at the saved threshold; it is not a new threshold search.",
        "",
        "## Key Question",
        "",
        f"Can the paper truthfully claim deployment-risk reduction under review-budget and worst-block metrics? **{answer}**",
        "",
        f"Overall RB17 win labels: {labels}.",
        f"Best-branch RB17 win labels: {best_labels}.",
        "",
        "## Corrected Best-Branch Risk Rows",
        "",
        markdown_table(corrected_risk.to_dict("records"), ["dataset", "protocol", "graph_model", "method", "metric", "mean_improvement", "win_label"]),
        "",
        "## Review-Budget Metrics",
        "",
        markdown_table(list(budget_rows), ["dataset", "method", "precision_at_0_5pct", "recall_at_0_5pct", "precision_at_1pct", "recall_at_1pct", "precision_at_2pct", "recall_at_2pct"]),
        "",
        "## Worst-Block Metrics",
        "",
        markdown_table(list(worst_rows)[:80], ["dataset", "protocol", "graph_model", "method", "cost_sensitive_risk", "worst_block_regret", "worst_window_recall_at_1pct", "worst_window_cost_sensitive_risk"]),
        "",
        "## Claim Boundary",
        "",
        "- Blocked: GraphSafe-TTA improves review-budget metrics across datasets as an unqualified statement unless this report has corrected support.",
        "- Blocked: GraphSafe-TTA reduces worst-block regret across datasets as an unqualified statement unless this report has corrected support.",
        "- The method remains decision-level; rank-level AUPRC/AUROC decay is not repaired.",
        "",
    ]
    (root / "aaai_upgrade" / "RB17_REVIEW_BUDGET_WORST_BLOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out_dir = root / OUT_DIR
    results = pd.read_csv(out_dir / "rb17_results.csv")
    worst = pd.read_csv(out_dir / "rb17_worst_block.csv")
    cost = pd.read_csv(out_dir / "rb17_cost_sensitivity.csv")
    stats = paired_tests(results, FOCUS_METHODS, COMPARATORS, METRICS, comparison_prefix="rb17")
    write_csv(out_dir / "rb17_statistical_tests.csv", stats.to_dict("records") if not stats.empty else [])
    budget_rows = _mean_rows(results)
    cost_rows = _cost_table(cost)
    table_dir = root / "results" / "paper_tables"
    write_csv(table_dir / "table_review_budget_metrics.csv", budget_rows)
    write_tex(
        table_dir / "table_review_budget_metrics.tex",
        budget_rows,
        ["dataset", "method", "precision_at_0_5pct", "recall_at_0_5pct", "precision_at_1pct", "recall_at_1pct", "precision_at_2pct", "recall_at_2pct"],
        "RB17 review-budget precision and recall.",
    )
    worst_rows = worst.to_dict("records")
    write_csv(table_dir / "table_worst_block_regret.csv", worst_rows)
    write_tex(
        table_dir / "table_worst_block_regret.tex",
        worst_rows,
        ["dataset", "protocol", "graph_model", "method", "cost_sensitive_risk", "worst_block_regret", "average_block_regret"],
        "RB17 worst-block regret.",
    )
    write_csv(table_dir / "table_cost_sensitive_risk.csv", cost_rows)
    write_tex(
        table_dir / "table_cost_sensitive_risk.tex",
        cost_rows,
        ["dataset", "method", "cost_ratio_fn_to_fp", "reconstructed_cost_sensitive_risk"],
        "RB17 cost-sensitive risk sensitivity.",
    )
    _write_figures(root, budget_rows, worst_rows, cost_rows)
    _write_report(root, stats, budget_rows, worst_rows, cost_rows)
    print(f"[rb17-analyze] stats={len(stats)} budget_rows={len(budget_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
