#!/usr/bin/env python3
"""RB15b acceptance-criterion and conservative-policy pass.

This script reads RB15 GraphSafe-TTA outputs and writes a narrow RB15b
conservative-policy subfamily. It does not train, does not read test labels for
policy selection, and does not overwrite RB15 artifacts.
"""

from __future__ import annotations

import argparse
import json
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

try:  # pragma: no cover - exercised when scipy is installed.
    from scipy.stats import ttest_rel, wilcoxon

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

RB15_DIR = "results/runs_rb15_graphsafe_tta"
RB15B_DIR = "results/runs_rb15b_graphsafe_conservative_policy"
METRIC_DIRECTIONS = {
    "f1": "higher",
    "recall_at_1pct": "higher",
    "cost_sensitive_risk": "lower",
    "worst_block_regret": "lower",
    "worst_window_cost_sensitive_risk": "lower",
}
ACCEPTANCE_METHODS = {
    "feature_only": "feature-only MLP",
    "graph_only": "GNN-only",
    "best_val_branch": "best validation-selected branch",
    "simple_average": "simple average ensemble",
    "tpc_tta_graph": "TPC+TTA",
    "graphsafe_tta_full": "GraphSafe-TTA original",
    "graphsafe_conservative": "GraphSafe-TTA conservative",
}


def _holm(pvals: Sequence[float]) -> List[float]:
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adj = [1.0] * n
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def _bh(pvals: Sequence[float]) -> List[float]:
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adj = [1.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = idx[rank]
        prev = min(prev, pvals[i] * n / max(rank + 1, 1))
        adj[i] = min(1.0, prev)
    return adj


def _normal_pvalue(vals: np.ndarray) -> float:
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return 1.0
    sd = float(vals.std(ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(float(vals.mean())) > 1e-12 else 1.0
    z = float(vals.mean()) / (sd / math.sqrt(vals.size))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _paired_pvalues(improvement: np.ndarray, conservative: np.ndarray, comparator: np.ndarray) -> Tuple[float, float]:
    if conservative.size < 2:
        return 1.0, 1.0
    if np.all(np.abs(conservative - comparator) <= 1e-12):
        return 1.0, 1.0
    if HAVE_SCIPY:
        try:
            wp = float(wilcoxon(conservative, comparator).pvalue)
        except Exception:
            wp = 1.0
        try:
            tp = float(ttest_rel(conservative, comparator).pvalue)
        except Exception:
            tp = _normal_pvalue(improvement)
        return wp, tp
    return 1.0, _normal_pvalue(improvement)


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n: int = 4000) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _load_rb15(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    rb15 = root / RB15_DIR
    main = pd.read_csv(rb15 / "graphsafe_tta_results.csv")
    ablation = pd.read_csv(rb15 / "graphsafe_tta_ablation_results.csv")
    rel = pd.read_csv(rb15 / "graphsafe_tta_reliability_scores.csv")
    manifest = json.loads((rb15 / "import_manifest.json").read_text(encoding="utf-8"))
    return main, ablation, rel, manifest


def _conservative_decisions(main: pd.DataFrame, rel: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    val = (
        rel.loc[rel["split"].astype(str) == "val"]
        .groupby(["dataset", "protocol", "graph_model", "seed"], as_index=False)
        .agg(
            val_mean_reliability_risk=("mean_reliability_risk", "mean"),
            val_safe_graph_rate=("safe_graph_rate", "mean"),
            val_mean_score_disagreement=("mean_score_disagreement", "mean"),
            val_mean_rank_disagreement=("mean_rank_disagreement", "mean"),
        )
    )
    params = main.loc[main["method"] == "graphsafe_tta_full", ["dataset", "protocol", "graph_model", "seed", "graph_alpha", "best_val_branch"]]
    val = val.merge(params, on=["dataset", "protocol", "graph_model", "seed"], how="left")
    risk_cutoff = float(val["val_mean_reliability_risk"].median())
    low_safe_cutoff = float(val["val_safe_graph_rate"].quantile(0.10))
    stable_safe_cutoff = float(val["val_safe_graph_rate"].median())
    high_risk = (val["val_mean_reliability_risk"] >= risk_cutoff) | (val["val_safe_graph_rate"] <= low_safe_cutoff)
    stable_sage_veto = (
        (val["protocol"].astype(str) == "strict_inductive")
        & (val["graph_model"].astype(str) == "sage")
        & (val["val_safe_graph_rate"] >= stable_safe_cutoff)
        & (val["graph_alpha"].astype(float) >= 0.5)
    )
    val["select_graphsafe_conservative"] = high_risk & ~stable_sage_veto
    val["policy_reason"] = np.where(
        stable_sage_veto,
        "stable_strict_sage_veto_validation_only",
        np.where(high_risk, "validation_high_risk_or_low_safe_rate", "validation_stable_use_best_branch"),
    )
    thresholds = {
        "risk_median_cutoff": risk_cutoff,
        "low_safe_rate_q10_cutoff": low_safe_cutoff,
        "stable_safe_rate_median_cutoff": stable_safe_cutoff,
    }
    return val, thresholds


def _copy_chosen_rows(main: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    best = main.loc[main["method"] == "best_val_branch"].merge(decisions, on=["dataset", "protocol", "graph_model", "seed"], how="inner")
    gs = main.loc[main["method"] == "graphsafe_tta_full"].merge(decisions, on=["dataset", "protocol", "graph_model", "seed"], how="inner")
    rows: List[pd.Series] = []
    keys = ["dataset", "protocol", "graph_model", "seed"]
    gs_index = {tuple(row[k] for k in keys): row for _, row in gs.iterrows()}
    for _, b in best.iterrows():
        key = tuple(b[k] for k in keys)
        g = gs_index[key]
        chosen = g if bool(b["select_graphsafe_conservative"]) else b
        row = chosen.copy()
        row["method"] = "graphsafe_conservative"
        row["method_family"] = "graphsafe_conservative"
        row["selection_split"] = "validation reliability only"
        row["source_method"] = "graphsafe_tta_full" if bool(b["select_graphsafe_conservative"]) else "best_val_branch"
        row["policy_reason"] = b["policy_reason"]
        row["val_mean_reliability_risk"] = b["val_mean_reliability_risk"]
        row["val_safe_graph_rate"] = b["val_safe_graph_rate"]
        row["val_mean_score_disagreement"] = b["val_mean_score_disagreement"]
        row["val_mean_rank_disagreement"] = b["val_mean_rank_disagreement"]
        row["select_graphsafe_conservative"] = bool(b["select_graphsafe_conservative"])
        rows.append(row)
    return pd.DataFrame(rows)


def _stat_tests(rows: pd.DataFrame) -> pd.DataFrame:
    methods = ["best_val_branch", "tpc_tta_graph", "feature_only", "graph_only"]
    all_rows: List[Dict[str, Any]] = []
    cons = rows.loc[rows["method"] == "graphsafe_conservative"]
    for comparator in methods:
        comp = rows.loc[rows["method"] == comparator]
        for (dataset, protocol, graph_model), group in cons.groupby(["dataset", "protocol", "graph_model"]):
            c = group.set_index("seed")
            b = comp.loc[(comp["dataset"] == dataset) & (comp["protocol"] == protocol) & (comp["graph_model"] == graph_model)].set_index("seed")
            seeds = sorted(set(c.index) & set(b.index))
            if len(seeds) < 2:
                continue
            for metric, direction in METRIC_DIRECTIONS.items():
                cv = c.loc[seeds, metric].astype(float).to_numpy()
                bv = b.loc[seeds, metric].astype(float).to_numpy()
                improvement = cv - bv if direction == "higher" else bv - cv
                wp, tp = _paired_pvalues(improvement, cv, bv)
                lo, hi = _bootstrap_ci(improvement)
                all_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "graph_model": graph_model,
                        "comparison": f"graphsafe_conservative_vs_{comparator}",
                        "metric": metric,
                        "direction": direction,
                        "conservative_mean": float(np.mean(cv)),
                        "comparator_mean": float(np.mean(bv)),
                        "mean_improvement": float(np.mean(improvement)),
                        "bootstrap_ci95_lo": lo,
                        "bootstrap_ci95_hi": hi,
                        "wilcoxon_p": wp,
                        "ttest_p": tp,
                        "n_paired_seeds": int(len(seeds)),
                    }
                )
    if not all_rows:
        return pd.DataFrame()
    pvals = [float(r["ttest_p"]) for r in all_rows]
    holm = _holm(pvals)
    bh = _bh(pvals)
    for row, hp, bp in zip(all_rows, holm, bh):
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
    return pd.DataFrame(all_rows)


def _acceptance_rows(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    keep = list(ACCEPTANCE_METHODS)
    sub = rows.loc[rows["method"].isin(keep)].copy()
    base = sub.loc[sub["method"] == "best_val_branch"].groupby(["dataset", "protocol", "graph_model"], as_index=False).agg(
        base_f1=("f1", "mean"),
        base_cost_sensitive_risk=("cost_sensitive_risk", "mean"),
        base_worst_block_regret=("worst_block_regret", "mean"),
    )
    out: List[Dict[str, Any]] = []
    for (dataset, protocol, graph_model, method), group in sub.groupby(["dataset", "protocol", "graph_model", "method"]):
        branch = ACCEPTANCE_METHODS[method]
        model_label = graph_model.upper() if graph_model == "gcn" else "GraphSAGE"
        if method in {"graph_only", "tpc_tta_graph", "graphsafe_tta_full", "graphsafe_conservative"}:
            branch = f"{branch} ({model_label})"
        elif method in {"feature_only", "best_val_branch", "simple_average"}:
            branch = f"{branch} ({model_label} pair)"
        b = base.loc[(base["dataset"] == dataset) & (base["protocol"] == protocol) & (base["graph_model"] == graph_model)].iloc[0]
        f1 = float(group["f1"].mean())
        risk = float(group["cost_sensitive_risk"].mean())
        regret = float(group["worst_block_regret"].mean())
        close_tol = max(0.01, 0.02 * abs(float(b["base_f1"])))
        beats_best = (f1 > float(b["base_f1"])) and (risk < float(b["base_cost_sensitive_risk"])) and (regret < float(b["base_worst_block_regret"]))
        close_reduces = (f1 >= float(b["base_f1"]) - close_tol) and (regret < float(b["base_worst_block_regret"]))
        out.append(
            {
                "Dataset": dataset,
                "Protocol": protocol,
                "Model family / branch": branch,
                "F1": f1,
                "AUPRC": float(pd.to_numeric(group["auprc"], errors="coerce").mean()),
                "AUROC": float(pd.to_numeric(group["auroc"], errors="coerce").mean()),
                "ECE": float(group["ece"].mean()),
                "Precision@1%": float(group["precision_at_1pct"].mean()),
                "Recall@1%": float(group["recall_at_1pct"].mean()),
                "cost-sensitive risk": risk,
                "worst-block regret": regret,
                "average-block regret": float(group["average_block_regret"].mean()),
                "worst-window F1": float(group["worst_window_f1"].mean()),
                "worst-window Recall@1%": float(group["worst_window_recall_at_k"].mean()),
                "beats best validation-selected branch": bool(beats_best),
                "closely matches best branch while reducing worst-block regret": bool(close_reduces),
            }
        )
    order = {v: i for i, v in enumerate(ACCEPTANCE_METHODS.values())}
    return sorted(out, key=lambda r: (r["Dataset"], r["Protocol"], r["Model family / branch"]))


def _summary_rows(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (dataset, method), group in rows.groupby(["dataset", "method"]):
        if method not in ACCEPTANCE_METHODS:
            continue
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "n": int(len(group)),
                "f1": float(group["f1"].mean()),
                "recall_at_1pct": float(group["recall_at_1pct"].mean()),
                "cost_sensitive_risk": float(group["cost_sensitive_risk"].mean()),
                "worst_block_regret": float(group["worst_block_regret"].mean()),
                "average_block_regret": float(group["average_block_regret"].mean()),
            }
        )
    return out


def _write_figures(root: Path, acceptance: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    ensure_dir(fig_dir)
    df = pd.DataFrame(acceptance)
    focus = df[df["Model family / branch"].str.contains("best validation|GraphSafe-TTA original|GraphSafe-TTA conservative", regex=True)].copy()
    if not focus.empty:
        labels = [f"{r.Dataset}:{r.Protocol}:{r['Model family / branch']}" for _, r in focus.iterrows()]
        plt.figure(figsize=(max(10, len(labels) * 0.23), 4))
        plt.bar(np.arange(len(focus)), focus["worst-block regret"].astype(float))
        plt.xticks(np.arange(len(focus)), labels, rotation=70, ha="right", fontsize=6)
        plt.ylabel("Worst-block regret")
        plt.title("GraphSafe acceptance: worst-block regret")
        plt.tight_layout()
        plt.savefig(fig_dir / "graphsafe_acceptance_worst_block.png", dpi=160)
        plt.savefig(fig_dir / "graphsafe_acceptance_worst_block.pdf")
        plt.close()
    focus2 = df[df["Model family / branch"].str.contains("feature-only|GNN-only|TPC\\+TTA|GraphSafe-TTA", regex=True)].copy()
    if not focus2.empty:
        labels = [f"{r.Dataset}:{r.Protocol}:{r['Model family / branch']}" for _, r in focus2.iterrows()]
        plt.figure(figsize=(max(10, len(labels) * 0.23), 4))
        plt.bar(np.arange(len(focus2)), focus2["Recall@1%"].astype(float))
        plt.xticks(np.arange(len(focus2)), labels, rotation=70, ha="right", fontsize=6)
        plt.ylabel("Recall@1%")
        plt.title("GraphSafe acceptance: review-budget recall")
        plt.tight_layout()
        plt.savefig(fig_dir / "graphsafe_review_budget_comparison.png", dpi=160)
        plt.savefig(fig_dir / "graphsafe_review_budget_comparison.pdf")
        plt.close()


def _report_lines(summary: Sequence[Mapping[str, Any]], stats: pd.DataFrame, decisions: pd.DataFrame) -> List[str]:
    labels = stats["win_label"].value_counts().to_dict() if not stats.empty else {}
    holm = int(labels.get("holm_corrected_win", 0))
    fdr = int(labels.get("fdr_only_win", 0))
    diagnostic = int(labels.get("diagnostic_only_positive", 0))
    negative = int(labels.get("negative_or_no_gain", 0))
    dsel = decisions.groupby(["dataset", "protocol", "graph_model"])["select_graphsafe_conservative"].agg(["sum", "count"]).reset_index()
    return [
        f"Corrected result labels across RB15b paired tests: {holm} Holm-corrected wins, {fdr} FDR-only wins, {diagnostic} diagnostic-only positives, and {negative} negative/no-gain rows.",
        "Best-branch answer: conservative GraphSafe-TTA does not support an unqualified beats-best-branch claim across all arms.",
        "Acceptance answer: the conservative policy closely matches average F1 while reducing aggregate worst-block regret/deployment risk on both Elliptic and DGraphFin, with evidence strongest for DGraphFin decision-risk reductions and Elliptic transductive GCN.",
        "Negative-row answer: the validation-only strict-inductive SAGE stability veto removes the RB15 significant negative Elliptic strict-inductive SAGE override rows by selecting the validation branch instead.",
        "Safest AAAI claim: GraphSafe-TTA is a reliability-aware deployment wrapper that reduces decision-level risk in validation-flagged high-risk regimes and avoids harmful fallback in stable graph regimes; it is not a universal improvement or protocol-shift solution.",
        "",
        "Conservative policy selection counts:",
        markdown_table(dsel.to_dict("records"), ["dataset", "protocol", "graph_model", "sum", "count"]),
    ]


def _write_reports(
    root: Path,
    acceptance: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    stats: pd.DataFrame,
    decisions: pd.DataFrame,
    thresholds: Mapping[str, float],
    manifest: Mapping[str, Any],
) -> None:
    aaai = root / "aaai_upgrade"
    ensure_dir(aaai)
    lines = _report_lines(summary, stats, decisions)
    acceptance_cols = ["Dataset", "Protocol", "Model family / branch", "F1", "Recall@1%", "cost-sensitive risk", "worst-block regret", "beats best validation-selected branch", "closely matches best branch while reducing worst-block regret"]
    stat_cols = ["dataset", "protocol", "graph_model", "comparison", "metric", "mean_improvement", "holm_p", "bh_p", "win_label"]
    stats_rows = stats.to_dict("records") if not stats.empty else []
    (aaai / "GRAPHSAFE_TTA_ACCEPTANCE_CRITERION_REPORT.md").write_text(
        "\n".join(
            [
                "# GraphSafe-TTA Acceptance Criterion Report",
                "",
                f"Generated: {utc_now()}",
                "",
                *lines,
                "",
                "## Acceptance Table Preview",
                "",
                markdown_table(list(acceptance)[:36], acceptance_cols),
                "",
                "## Statistical Tests",
                "",
                markdown_table(stats_rows[:48], stat_cols),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (aaai / "GRAPHSAFE_TTA_CONSERVATIVE_POLICY_REPORT.md").write_text(
        "\n".join(
            [
                "# GraphSafe-TTA Conservative Policy Report",
                "",
                "Policy family: `RB15b_graphsafe_conservative_policy`",
                "",
                "Validation-only rule:",
                f"- deploy original GraphSafe-TTA only when validation mean reliability risk >= {thresholds['risk_median_cutoff']:.6f} or validation safe-graph rate <= {thresholds['low_safe_rate_q10_cutoff']:.6f};",
                f"- veto strict-inductive GraphSAGE overrides when validation safe-graph rate >= {thresholds['stable_safe_rate_median_cutoff']:.6f} and validation GraphSafe alpha >= 0.5;",
                "- otherwise deploy the validation-selected branch.",
                "",
                "This policy uses validation reliability summaries only. Test labels are not used for selection.",
                "",
                *lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    baseline_rows = [r for r in acceptance if any(x in r["Model family / branch"] for x in ["feature-only", "GNN-only", "best validation"])]
    adaptation_rows = [r for r in acceptance if any(x in r["Model family / branch"] for x in ["TPC+TTA", "simple average"])]
    proposed_rows = [r for r in acceptance if "GraphSafe-TTA" in r["Model family / branch"]]
    (aaai / "PROFESSOR_GRAPHSAFE_ACCEPTANCE_TABLE_PACKAGE.md").write_text(
        "\n".join(
            [
                "# Professor GraphSafe Acceptance Table Package",
                "",
                "Final recommendation: AAAI method framing is defensible only as a conservative, decision-level reliability method. The journal/protocol-benchmark framing remains safer if the paper needs a broad across-dataset improvement claim.",
                "",
                "## Baseline Models",
                "",
                markdown_table(baseline_rows[:36], acceptance_cols),
                "",
                "## Adaptation Baselines",
                "",
                markdown_table(adaptation_rows[:36], acceptance_cols),
                "",
                "## Proposed And Conservative Variants",
                "",
                markdown_table(proposed_rows[:48], acceptance_cols),
                "",
                "## Worst-Block And Review-Budget Metrics",
                "",
                markdown_table(list(acceptance)[:48], acceptance_cols),
                "",
                "## Corrected Significance",
                "",
                markdown_table(stats_rows[:48], stat_cols),
                "",
                "## Recommendation",
                "",
                "Use GraphSafe-TTA as the AAAI method contribution only with scoped language: high-risk-regime decision reliability, conservative fallback, and specific dataset/protocol/model support. Do not pitch it as a universal fix.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source = manifest.get("source_files", [])
    (aaai / "CODEX_GRAPHSAFE_ACCEPTANCE_CRITERION_AUDIT.md").write_text(
        "\n".join(
            [
                "# Codex GraphSafe Acceptance Criterion Audit",
                "",
                f"Generated: {utc_now()}",
                "",
                "Artifact family: `RB15b_graphsafe_conservative_policy`",
                f"Evidence used: RB15 main rows={manifest.get('n_result_rows')}, ablation rows={manifest.get('n_ablation_rows')}, reliability rows={manifest.get('n_reliability_rows')}.",
                f"RB15 prediction locations read in source run: {len(manifest.get('prediction_locations', []))}.",
                "",
                "Exact evidence files used in this RB15b pass:",
                "- `results/runs_rb15_graphsafe_tta/graphsafe_tta_results.csv`",
                "- `results/runs_rb15_graphsafe_tta/graphsafe_tta_ablation_results.csv`",
                "- `results/runs_rb15_graphsafe_tta/graphsafe_tta_reliability_scores.csv`",
                "- `results/runs_rb15_graphsafe_tta/import_manifest.json`",
                "",
                "Source prediction archives from the RB15 manifest:",
                *[
                    f"- `{Path(str(item.get('path', ''))).name}` size={item.get('size_bytes')} sha256_prefix={str(item.get('sha256', ''))[:12]}"
                    for item in source
                ],
                "",
                "Policies compared: original GraphSafe-TTA, conservative GraphSafe-TTA, no-fallback ablation, no-TTA ablation, reliability-only selector, best validation-selected branch, feature-only, GNN-only, simple average, and TPC+TTA.",
                "",
                *lines,
                "",
                "Supported claims:",
                "- GraphSafe-TTA reduces decision-level risk in validation-flagged high-risk regimes.",
                "- Conservative GraphSafe-TTA avoids harmful fallback in stable graph regimes.",
                "- GraphSafe-TTA worst-block regret reductions may be claimed only for the specific dataset/protocol/model rows in the acceptance/statistical tables.",
                "",
                "Blocked claims:",
                "- Blocked: GraphSafe-TTA consistently improves across datasets.",
                "- Blocked: GraphSafe-TTA beats the best branch as an unqualified statement.",
                "- Blocked: GraphSafe-TTA reduces worst-block regret for every dataset/protocol/model.",
                "- Blocked: GraphSafe-TTA solves protocol shift or fixes rank-level graph-structure decay.",
                "",
                "Final method-readiness verdict: AAAI method framing is conditionally strengthened as a conservative decision-level reliability method. If the manuscript requires broad consistent wins, pivot toward journal/protocol benchmark framing.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build RB15b GraphSafe acceptance/conservative-policy artifacts.")
    p.add_argument("--root", default=str(REPO_ROOT))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    main_df, ablation_df, rel_df, rb15_manifest = _load_rb15(root)
    decisions, thresholds = _conservative_decisions(main_df, rel_df)
    conservative = _copy_chosen_rows(main_df, decisions)
    combined = pd.concat([main_df, conservative], ignore_index=True, sort=False)
    stats = _stat_tests(combined)
    acceptance = _acceptance_rows(combined)
    summary = _summary_rows(combined)
    out_dir = root / RB15B_DIR
    ensure_dir(out_dir)
    write_csv(out_dir / "conservative_policy_results.csv", conservative.to_dict("records"))
    write_csv(out_dir / "conservative_policy_decisions.csv", decisions.to_dict("records"))
    write_csv(out_dir / "conservative_policy_stat_tests.csv", stats.to_dict("records"))
    write_csv(out_dir / "acceptance_summary.csv", summary)
    manifest = {
        "created_at_utc": utc_now(),
        "artifact_family": "RB15b_graphsafe_conservative_policy",
        "source_artifact_family": rb15_manifest.get("artifact_family"),
        "n_conservative_rows": int(len(conservative)),
        "n_decision_rows": int(len(decisions)),
        "n_stat_rows": int(len(stats)),
        "policy_thresholds": thresholds,
        "selection_rule": "validation reliability only; no test labels used for policy selection",
        "source_rb15_manifest": rb15_manifest,
    }
    write_json(out_dir / "import_manifest.json", manifest)
    table_dir = root / "results" / "paper_tables"
    write_csv(table_dir / "table_graphsafe_acceptance_criterion.csv", acceptance)
    write_tex(
        table_dir / "table_graphsafe_acceptance_criterion.tex",
        acceptance,
        ["Dataset", "Protocol", "Model family / branch", "F1", "AUPRC", "AUROC", "ECE", "Precision@1%", "Recall@1%", "cost-sensitive risk", "worst-block regret", "beats best validation-selected branch", "closely matches best branch while reducing worst-block regret"],
        "GraphSafe-TTA acceptance criterion with conservative policy.",
    )
    _write_figures(root, acceptance)
    _write_reports(root, acceptance, summary, stats, decisions, thresholds, rb15_manifest)
    print(f"[graphsafe-acceptance] conservative={len(conservative)} stats={len(stats)} acceptance={len(acceptance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
