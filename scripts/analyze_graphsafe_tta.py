#!/usr/bin/env python3
"""Summarize RB15 GraphSafe-TTA outputs into paper tables, figures, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraudshiftbench.result_analysis import ensure_dir, markdown_table, utc_now, write_csv, write_json, write_tex  # noqa: E402

OUT = "results/runs_rb15_graphsafe_tta"


def _mean_table(df: pd.DataFrame, methods: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sub = df.loc[df["method"].isin(methods)].copy()
    if sub.empty:
        return rows
    for (dataset, method), group in sub.groupby(["dataset", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": int(len(group)),
                "f1": float(group["f1"].mean()),
                "auprc": float(pd.to_numeric(group["auprc"], errors="coerce").mean()),
                "auroc": float(pd.to_numeric(group["auroc"], errors="coerce").mean()),
                "ece": float(group["ece"].mean()),
                "brier": float(group["brier"].mean()),
                "precision_at_1pct": float(group["precision_at_1pct"].mean()),
                "recall_at_1pct": float(group["recall_at_1pct"].mean()),
                "cost_sensitive_risk": float(group["cost_sensitive_risk"].mean()),
                "worst_block_regret": float(group["worst_block_regret"].mean()),
                "worst_window_f1": float(group["worst_window_f1"].mean()),
                "worst_window_cost_sensitive_risk": float(group["worst_window_cost_sensitive_risk"].mean()),
            }
        )
    return rows


def _worst_block_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sub = df.loc[df["method"].isin(["best_val_branch", "graphsafe_tta_full", "feature_only", "graph_only"])].copy()
    for (dataset, protocol, graph_model, method), group in sub.groupby(["dataset", "protocol", "graph_model", "method"]):
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "graph_model": graph_model,
                "method": method,
                "n": int(len(group)),
                "worst_block_regret": float(group["worst_block_regret"].mean()),
                "average_block_regret": float(group["average_block_regret"].mean()),
                "worst_window_f1": float(group["worst_window_f1"].mean()),
                "worst_window_recall_at_k": float(group["worst_window_recall_at_k"].mean()),
                "worst_window_cost_sensitive_risk": float(group["worst_window_cost_sensitive_risk"].mean()),
            }
        )
    return rows


def _ablation_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows
    for (dataset, ablation), group in df.groupby(["dataset", "ablation"]):
        rows.append(
            {
                "dataset": dataset,
                "ablation": ablation,
                "n": int(len(group)),
                "f1": float(group["f1"].mean()),
                "recall_at_1pct": float(group["recall_at_1pct"].mean()),
                "cost_sensitive_risk": float(group["cost_sensitive_risk"].mean()),
                "worst_block_regret": float(group["worst_block_regret"].mean()),
            }
        )
    return rows


def _write_figures(root: Path, main_rows: Sequence[Mapping[str, Any]], worst_rows: Sequence[Mapping[str, Any]], rel: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    ensure_dir(fig_dir)
    main = pd.DataFrame(main_rows)
    if not main.empty:
        pivot = main.loc[main["method"].isin(["feature_only", "graph_only", "best_val_branch", "graphsafe_tta_full"])]
        labels = [f"{r.dataset}:{r.method}" for r in pivot.itertuples()]
        vals = pivot["recall_at_1pct"].astype(float).to_numpy()
        plt.figure(figsize=(max(8, len(labels) * 0.35), 4))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        plt.ylabel("Recall@1%")
        plt.title("GraphSafe-TTA review-budget recall")
        plt.tight_layout()
        plt.savefig(fig_dir / "graphsafe_tta_review_budget.png", dpi=160)
        plt.savefig(fig_dir / "graphsafe_tta_review_budget.pdf")
        plt.close()
    worst = pd.DataFrame(worst_rows)
    if not worst.empty:
        pivot = worst.loc[worst["method"].isin(["best_val_branch", "graphsafe_tta_full"])]
        labels = [f"{r.dataset}:{r.protocol}:{r.graph_model}:{r.method}" for r in pivot.itertuples()]
        vals = pivot["worst_block_regret"].astype(float).to_numpy()
        plt.figure(figsize=(max(9, len(labels) * 0.28), 4))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=6)
        plt.ylabel("Worst-block cost regret")
        plt.title("GraphSafe-TTA worst-block regret")
        plt.tight_layout()
        plt.savefig(fig_dir / "graphsafe_tta_worst_block_regret.png", dpi=160)
        plt.savefig(fig_dir / "graphsafe_tta_worst_block_regret.pdf")
        plt.close()
    if not rel.empty:
        sub = rel.loc[rel["split"].astype(str) == "test"].copy()
        grp = sub.groupby(["dataset", "graph_model"])["safe_graph_rate"].mean().reset_index()
        labels = [f"{r.dataset}:{r.graph_model}" for r in grp.itertuples()]
        vals = grp["safe_graph_rate"].astype(float).to_numpy()
        plt.figure(figsize=(max(6, len(labels) * 0.5), 4))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Mean safe graph-use rate")
        plt.title("GraphSafe-TTA reliability gate")
        plt.tight_layout()
        plt.savefig(fig_dir / "graphsafe_tta_reliability_gate.png", dpi=160)
        plt.savefig(fig_dir / "graphsafe_tta_reliability_gate.pdf")
        plt.close()


def _status_from_stats(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "BLOCKED: no paired statistical rows were produced."
    good = stats.loc[(stats["comparison"] == "graphsafe_tta_full_vs_best_val_branch") & (stats["mean_improvement"].astype(float) > 0)]
    holm = good.loc[good["survives_holm"].astype(str).str.lower() == "true"]
    fdr = good.loc[good["survives_fdr"].astype(str).str.lower() == "true"]
    if not holm.empty:
        return "STRENGTHENED: at least one GraphSafe-TTA decision-level improvement survives Holm correction."
    if not fdr.empty:
        return "PARTIAL: at least one decision-level improvement survives FDR but not Holm correction."
    if not good.empty:
        return "DIAGNOSTIC: positive paired deltas exist but do not survive correction."
    return "WEAK/NEGATIVE: GraphSafe-TTA does not beat the validation-selected branch on the tested decision metrics."


def _write_reports(root: Path, main_rows: Sequence[Mapping[str, Any]], worst_rows: Sequence[Mapping[str, Any]], ablation_rows: Sequence[Mapping[str, Any]], stats: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    aaai = root / "aaai_upgrade"
    docs = root / "docs" / "paper_sections"
    ensure_dir(aaai)
    ensure_dir(docs)
    main_cols = ["dataset", "method", "f1", "auprc", "auroc", "ece", "brier", "recall_at_1pct", "cost_sensitive_risk", "worst_block_regret"]
    worst_cols = ["dataset", "protocol", "graph_model", "method", "worst_block_regret", "average_block_regret", "worst_window_f1", "worst_window_recall_at_k", "worst_window_cost_sensitive_risk"]
    abl_cols = ["dataset", "ablation", "f1", "recall_at_1pct", "cost_sensitive_risk", "worst_block_regret"]
    verdict = _status_from_stats(stats)
    (aaai / "GRAPHSAFE_TTA_MAIN_REPORT.md").write_text(
        "\n".join(
            [
                "# GraphSafe-TTA Main Report",
                "",
                f"Generated: {utc_now()}",
                "",
                "GraphSafe-TTA is evaluated as a decision-level deployment method. It is not claimed to repair rank-level structure decay or solve protocol shift.",
                "",
                markdown_table(main_rows, main_cols),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (aaai / "GRAPHSAFE_TTA_WORST_BLOCK_REPORT.md").write_text(
        "# GraphSafe-TTA Worst-Block Report\n\n" + markdown_table(worst_rows, worst_cols) + "\n",
        encoding="utf-8",
    )
    (aaai / "GRAPHSAFE_TTA_ABLATION_REPORT.md").write_text(
        "# GraphSafe-TTA Ablation Report\n\n" + markdown_table(ablation_rows, abl_cols) + "\n",
        encoding="utf-8",
    )
    stats_rows = stats.to_dict("records") if not stats.empty else []
    (aaai / "GRAPHSAFE_TTA_AAAI_CLAIM_VERDICT.md").write_text(
        "\n".join(
            [
                "# GraphSafe-TTA AAAI Claim Verdict",
                "",
                f"Verdict: **{verdict}**",
                "",
                "Allowed claims require support from RB15 tables:",
                "- GraphSafe-TTA improves decision-level deployment reliability when the corresponding corrected row is positive.",
                "- GraphSafe-TTA reduces worst-block regret only where the worst-block table shows lower regret than the validation-selected branch.",
                "- GraphSafe-TTA improves review-budget recall or precision only where the budget table shows an absolute gain.",
                "- GraphSafe-TTA detects high-risk graph-use regimes through unlabeled reliability scores.",
                "",
                "Blocked claims:",
                "- Blocked: GraphSafe-TTA solves protocol shift.",
                "- Blocked: GraphSafe-TTA fixes graph-structure decay.",
                "- Blocked unless exact rank-level evidence exists: GraphSafe-TTA improves rank-level metrics.",
                "- Blocked: GraphSafe-TTA is causal.",
                "- Blocked unless corrected evidence supports it: GraphSafe-TTA beats all baselines.",
                "",
                markdown_table(stats_rows, ["dataset", "protocol", "graph_model", "metric", "mean_improvement", "p_value", "survives_holm", "survives_fdr", "win_label"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (aaai / "PROFESSOR_GRAPHSAFE_TTA_RESULT_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Professor GraphSafe-TTA Result Summary",
                "",
                f"Verdict: **{verdict}**",
                "",
                "## Compact Absolute Metrics",
                "",
                markdown_table(main_rows, main_cols),
                "",
                "## Ablations",
                "",
                markdown_table(ablation_rows, abl_cols),
                "",
                "## Worst-Block Deployment Metrics",
                "",
                markdown_table(worst_rows, worst_cols),
                "",
                "## Reliability Diagnostics",
                "",
                "See `results/runs_rb15_graphsafe_tta/graphsafe_tta_reliability_scores.csv` for unlabeled block-level disagreement, drift, entropy, and gate rates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "graphsafe_tta_method_section.md").write_text(
        "# GraphSafe-TTA Method Section\n\nGraphSafe-TTA is a leakage-safe deployment wrapper for temporal graph fraud detection. It estimates graph-use reliability from unlabeled deployment-window disagreement, rank disagreement, entropy, and validation-to-deployment score drift; it selects graph, feature, or fallback scores using validation-fitted cutoffs; and it reports review-budget-aware decisions under fixed top-budget policies. Parameters are selected on training/validation windows only.\n",
        encoding="utf-8",
    )
    (docs / "graphsafe_tta_results_section.md").write_text(
        "# GraphSafe-TTA Results Section\n\nBenchmark/protocol results motivate the deployment problem. RB15 reports absolute F1, AUPRC, AUROC, ECE, Brier, review-budget metrics, cost-sensitive risk, and worst-block regret for feature-only, GNN-only, adaptation baselines, GraphSafe-TTA, and ablations. Claims should follow the verdict in `aaai_upgrade/GRAPHSAFE_TTA_AAAI_CLAIM_VERDICT.md`.\n",
        encoding="utf-8",
    )
    (docs / "graphsafe_tta_limitations_section.md").write_text(
        "# GraphSafe-TTA Limitations Section\n\nGraphSafe-TTA is a decision-level reliability method. It does not repair rank-level graph-structure decay, does not solve protocol shift, and does not establish causal graph harm. Test labels are used only for final evaluation, so unsupported improvements remain blocked even when diagnostics are suggestive.\n",
        encoding="utf-8",
    )
    (aaai / "CODEX_GRAPHSAFE_TTA_FINAL_AUDIT.md").write_text(
        "\n".join(
            [
                "# Codex GraphSafe-TTA Final Audit",
                "",
                f"Generated: {utc_now()}",
                "",
                f"Verdict: **{verdict}**",
                "",
                f"Artifact family: `{manifest.get('artifact_family', 'RB15_graphsafe_tta')}`",
                f"Prediction locations read: {len(manifest.get('prediction_locations', []))}",
                f"Result rows: {manifest.get('n_result_rows')}",
                f"Ablation rows: {manifest.get('n_ablation_rows')}",
                f"Reliability rows: {manifest.get('n_reliability_rows')}",
                "",
                "Methods compared: feature-only MLP, GNN-only GCN/SAGE, validation-selected branch, simple average, validation-weighted ensemble, temperature/prior/threshold/TPC-style graph adaptations, GraphSafe-TTA full, and GraphSafe-TTA ablations.",
                "",
                "Supported claims and blocked claims are governed by `aaai_upgrade/GRAPHSAFE_TTA_AAAI_CLAIM_VERDICT.md` and `scripts/check_claim_gates.py`.",
                "",
                "Verification commands requested:",
                "- `gnn_env/bin/python scripts/validate_graphsafe_tta_artifacts.py`",
                "- `gnn_env/bin/python scripts/check_claim_gates.py`",
                "- `gnn_env/bin/python scripts/audit_figure_readiness.py`",
                "- `gnn_env/bin/python -m unittest discover -s tests -p \"test_*.py\"`",
                "- `bash scripts/run_cpu_reliability_suite.sh`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze GraphSafe-TTA RB15 artifacts.")
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--input-dir", default=OUT)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    out = root / args.input_dir
    main_csv = out / "graphsafe_tta_results.csv"
    ablation_csv = out / "graphsafe_tta_ablation_results.csv"
    rel_csv = out / "graphsafe_tta_reliability_scores.csv"
    stat_csv = out / "graphsafe_tta_stat_tests.csv"
    manifest_path = out / "import_manifest.json"
    if not main_csv.is_file():
        print(f"[graphsafe-analyze] missing {main_csv}")
        return 1
    df = pd.read_csv(main_csv)
    abl = pd.read_csv(ablation_csv) if ablation_csv.is_file() else pd.DataFrame()
    rel = pd.read_csv(rel_csv) if rel_csv.is_file() else pd.DataFrame()
    stats = pd.read_csv(stat_csv) if stat_csv.is_file() else pd.DataFrame()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    main_methods = [
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
    main_rows = _mean_table(df, main_methods)
    worst_rows = _worst_block_table(df)
    ablation_rows = _ablation_table(abl)
    table_dir = root / "results" / "paper_tables"
    write_csv(table_dir / "table_graphsafe_tta_main.csv", main_rows)
    write_tex(table_dir / "table_graphsafe_tta_main.tex", main_rows, ["dataset", "method", "f1", "auprc", "auroc", "ece", "brier", "recall_at_1pct", "cost_sensitive_risk"], "GraphSafe-TTA main absolute metrics.")
    write_csv(table_dir / "table_graphsafe_tta_worst_block.csv", worst_rows)
    write_tex(table_dir / "table_graphsafe_tta_worst_block.tex", worst_rows, ["dataset", "protocol", "graph_model", "method", "worst_block_regret", "worst_window_f1", "worst_window_cost_sensitive_risk"], "GraphSafe-TTA worst-block deployment metrics.")
    write_csv(table_dir / "table_graphsafe_tta_ablation.csv", ablation_rows)
    write_tex(table_dir / "table_graphsafe_tta_ablation.tex", ablation_rows, ["dataset", "ablation", "f1", "recall_at_1pct", "cost_sensitive_risk", "worst_block_regret"], "GraphSafe-TTA ablations.")
    _write_figures(root, main_rows, worst_rows, rel)
    _write_reports(root, main_rows, worst_rows, ablation_rows, stats, manifest)
    write_json(out / "analysis_manifest.json", {"created_at_utc": utc_now(), "n_main_rows": len(main_rows), "n_worst_rows": len(worst_rows), "n_ablation_rows": len(ablation_rows), "status": _status_from_stats(stats)})
    print(f"[graphsafe-analyze] main={len(main_rows)} worst={len(worst_rows)} ablation={len(ablation_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
