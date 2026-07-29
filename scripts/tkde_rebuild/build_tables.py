#!/usr/bin/env python3
"""Compatibility entry point for the publication-safe V2 table generators.

The historical helper functions below are retained only so older imports do
not break.  Executing this module delegates to the portrait, readable V2
generators and cannot recreate the superseded microscopic raw-row tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tkde_rebuild"
DATA = OUT / "table_data"
TABLES = ROOT / "paper_tkde" / "tables"
SUPP = ROOT / "paper_tkde" / "supplement" / "tables"


def esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def header_tex(value: Any) -> str:
    """Escape plain CSV headers while preserving intentional LaTeX math."""
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    return esc(text)


STATUS_DISPLAY = {
    "PASS_FULL10": "pass; ten seeds",
    "PASS_FULL10_MERGED_WITH_COMPATIBLE_ALIASES": "pass; compatible aliases",
    "SAFE_RESOURCE_BLOCKED": "safe resource block",
    "RESOURCE_BLOCKED_T4_CUDA_OOM": "T4 CUDA OOM",
    "BLOCKED_T4_OOM": "T4 CUDA OOM",
    "BLOCKED_WAITING_FOR_GPU": "awaiting larger GPU",
    "SUPPORTED_WITH_RESOURCE_BOUNDARY": "supported; resource-bounded",
    "SUPPORTED_THEORETICALLY": "supported theoretically",
    "REFUTED_IN_SCOPE": "refuted in scope",
    "DIAGNOSTIC_ONLY": "diagnostic only",
}


def status_display(value: Any) -> str:
    text = str(value)
    return STATUS_DISPLAY.get(text, text.replace("_", " ").lower())


def f4(value: float) -> str:
    return "--" if pd.isna(value) else f"{float(value):.4f}"


def f3(value: float) -> str:
    return "--" if pd.isna(value) else f"{float(value):.3f}"


def summary(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def pvalue(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_data(name: str, df: pd.DataFrame) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def main_related_work() -> tuple[pd.DataFrame, Path]:
    rows = [
        ["GADBench", "graph anomaly", "No", "Limited", "No", "No", "Benchmark", "No"],
        ["TGB / BenchTemp", "temporal graphs", "Yes", "Inductive tasks", "No", "Efficiency", "Benchmark", "No"],
        ["BAG", "dynamic anomaly", "Yes", "Task/model breadth", "No", "Scale reported", "Benchmark", "No"],
        ["GAD in the Wild", "deployment anomaly", "Partial", "Scale/missing attributes", "No", "Scale challenge", "Limited", "No"],
        ["BetterBench / Eval Factsheets", "benchmark governance", "Generic", "Documented", "Documented", "Documented", "Metadata", "Descriptive"],
        ["FraudShiftBench", "temporal graph fraud", "Yes", "Visibility + construction", "Top-K + cost", "Typed blocked status", "Per-seed manifests", "Executable"],
    ]
    df = pd.DataFrame(rows, columns=["work", "domain", "temporal", "graph_controls", "decision_budget", "resource_semantics", "prediction_provenance", "claim_support"])
    path = TABLES / "table01_related_work.tex"
    body = "\n".join(
        f"{esc(r.work)} & {esc(r.domain)} & {esc(r.temporal)} & {esc(r.graph_controls)} & {esc(r.decision_budget)} & {esc(r.resource_semantics)} & {esc(r.prediction_provenance)} & {esc(r.claim_support)} \\\\"
        for _, r in df.iterrows()
    )
    write(
        path,
        r"""\begin{table*}[t]
\caption{Positioning against the closest benchmark families. ``No'' means the feature is not a stated benchmark axis, not that the work lacks documentation or rigor. GAD in the Wild and Eval Factsheets were preprints at the audit date.}
\label{tab:related-comparison}
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabularx}{\textwidth}{@{}p{0.105\textwidth}p{0.105\textwidth}p{0.055\textwidth}p{0.14\textwidth}p{0.10\textwidth}p{0.12\textwidth}p{0.105\textwidth}X@{}}
\toprule
Work & Domain & Temporal & Visibility / construction & Decision budget & Resource semantics & Prediction provenance & Claim support \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""",
    )
    return df, path


def compact_datasets() -> tuple[pd.DataFrame, Path]:
    full = pd.read_csv(OUT / "DATASET_TASK_STATISTICS.csv")
    rows = []
    for dataset in ["Elliptic", "DGraphFin"]:
        r = full[(full.dataset == dataset)].iloc[0]
        rows.append(
            {
                "task": dataset,
                "origin_unit": ("real / tx node" if dataset == "Elliptic" else "real / account node"),
                "nodes": int(r.nodes_or_accounts),
                "edges": int(r.source_edges_or_transactions),
                "features": f"{int(r.node_feature_dim)} node / 0 edge",
                "test": f"{int(r.test_units):,}",
                "positives": f"{int(r.test_positives):,}",
                "test_prevalence": f"{100*r.test_positive_rate:.3f}%",
                "windows": "t1--30 / 31--34 / 35--49" if dataset == "Elliptic" else "t1--14 / 15--16 / 17--20",
            }
        )
    ibm = full[full.dataset.eq("IBM AML-Data")]
    for variant in ["hi-small", "hi-medium", "li-small", "li-medium"]:
        group = ibm[ibm.variant.eq(variant)]
        r = group.iloc[0]
        rows.append(
            {
                "task": f"IBM {variant}",
                "origin_unit": "synthetic / tx edge",
                "nodes": int(r.nodes_or_accounts),
                "edges": int(r.source_edges_or_transactions),
                "features": "8 node / 8 edge",
                "test": f"{int(group.test_units.min()):,}--{int(group.test_units.max()):,}",
                "positives": f"{int(group.test_positives.min()):,}--{int(group.test_positives.max()):,}",
                "test_prevalence": f"{100*group.test_positive_rate.min():.3f}--{100*group.test_positive_rate.max():.3f}%",
                "windows": "50/20/30; 60/20/20",
            }
        )
    df = pd.DataFrame(rows)
    path = TABLES / "table02_dataset_tasks.tex"
    body = "\n".join(
        f"{esc(r.task)} & {esc(r.origin_unit)} & {r.nodes:,} & {r.edges:,} & {esc(r.features)} & {esc(r.test)} & {esc(r.positives)} & {esc(r.test_prevalence)} & {esc(r.windows)} \\\\"
        for _, r in df.iterrows()
    )
    write(
        path,
        r"""\begin{table*}[t]
\caption{Task and graph statistics. Edges are source edges/transactions before the Elliptic and DGraphFin loaders symmetrize the message-passing graph. IBM ranges cover the two chronological protocols. HI/LI denote the synthetic generator's higher/lower illicit-ratio regimes; Large is excluded from performance analysis by the resource guard.}
\label{tab:datasets}
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}l l r r l r r l X@{}}
\toprule
Task & Origin / unit & Nodes & Edges / txs & Features & Test $n$ & Test $+$ & Test prior & Train / val / test \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""",
    )
    return df, path


def main_models() -> tuple[pd.DataFrame, Path]:
    rows = [
        ["MLP", "node features only", "graph-visibility negative control", "h256/l3 Ell.; h64/l2 DGraph"],
        ["GCN", "normalized neighborhood aggregation", "visibility-sensitive smoothing", "same widths/depths as MLP"],
        ["GraphSAGE", "mean aggregation", "inductive aggregation", "same widths/depths as MLP"],
        ["LR / HistGB", "transaction features", "linear / nonlinear tabular controls", "balanced LR; 160-tree-iteration HistGB"],
        ["SAGE-derived h32", "endpoint + train-neighbor + edge features", "small graph baseline", "20 epochs; batch 65,536"],
        ["Edge-aware h64", "endpoint + train-neighbor + edge features", "graph-ablation reference", "30 epochs; batch 32,768"],
        ["GINE h64", "one GINE layer + edge head", "edge-conditioned message passing", "Small measured; Medium T4 OOM"],
        ["Graph controls", "NoEdge, Shuffle, Degree, DegreeCap, Recent", "remove alignment, cap hubs, or restrict history", "matched h64 cells"],
    ]
    df = pd.DataFrame(rows, columns=["method", "operation", "hypothesis", "configuration"])
    path = TABLES / "table03_model_constructions.tex"
    body = "\n".join(f"{esc(r.method)} & {esc(r.operation)} & {esc(r.hypothesis)} & {esc(r.configuration)} \\\\" for _, r in df.iterrows())
    write(
        path,
        r"""\begin{table*}[t]
\caption{Model and construction inventory. Scientific names replace internal run identifiers; full optimization and feature definitions are in the supplement. The IBM ``sender--receiver'' row is an implementation alias of the reference graph and is not counted as an independent method.}
\label{tab:models}
\centering
\scriptsize
\begin{tabularx}{\textwidth}{@{}p{0.15\textwidth}p{0.28\textwidth}p{0.28\textwidth}X@{}}
\toprule
Method / control & Computation & Hypothesis & Configuration / scope \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""",
    )
    return df, path


def main_rb09() -> tuple[pd.DataFrame, Path]:
    df = pd.read_csv(OUT / "RB09_AUPRC_MAIN.csv")
    label = {"gcn": "GCN", "mlp": "MLP", "sage": "GraphSAGE"}
    body = []
    for _, r in df.iterrows():
        body.append(
            f"{esc(r.dataset.title())} & {label[r.model]} & {summary(r.strict_mean,r.strict_std)} & {summary(r.isolated_mean,r.isolated_std)} & {r.delta_isolated_minus_strict:+.4f} [{r.delta_ci95_low:+.4f},{r.delta_ci95_high:+.4f}] & {r.relative_delta_pct:+.1f} & {r.cohen_dz:+.2f} & {pvalue(r.holm_p_within_metric)} \\\\"
        )
    path = TABLES / "table04_rb09_protocol_effects.tex"
    write(
        path,
        r"""\begin{table*}[t]
\caption{AUPRC under matched strict- and isolated-inductive graph visibility (mean $\pm$ SD, ten paired seeds). $\Delta$ is isolated minus strict; brackets are a deterministic 95\% bootstrap interval. Holm correction covers the six AUPRC comparisons. The MLP is unchanged because it does not consume graph structure.}
\label{tab:rb09-effects}
\centering
\scriptsize
\setlength{\tabcolsep}{3.1pt}
\begin{tabular}{@{}llcccrrr@{}}
\toprule
Dataset & Model & Strict & Isolated & $\Delta$ [95\% CI] & Rel. $\Delta$ (\%) & $d_z$ & Holm $p$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}""",
    )
    return df, path


CONFIG = {
    "hist_gradient_boosting_edge_features": "HistGB",
    "logistic_regression_edge_features": "LR",
    "graphsage_edge_minibatch_h32": "SAGE-h32",
    "edge_aware_graphsage_h64": "Ref-h64",
    "edge_aware_graphsage_h64_no_edge_features": "NoEdge",
    "edge_aware_graphsage_h64_shuffled_edge_features": "Shuffle",
    "edge_aware_graphsage_h64_degree_only": "Degree",
    "degree_capped_bipartite": "DegreeCap",
    "recent_window_only_graph": "Recent",
    "gine_light_h64": "GINE",
}


def main_ibm() -> tuple[pd.DataFrame, Path]:
    summary_df = pd.read_csv(OUT / "IBM_CELL_SUMMARY.csv")
    ranks = pd.read_csv(OUT / "IBM_RANK_DIVERGENCE.csv")
    baseline = summary_df[summary_df.version.eq("V26")]
    graph = summary_df[(summary_df.version.eq("V27")) | (summary_df.version.eq("V28") & ~summary_df.config.eq("account_account_sender_receiver"))]
    rows = []
    protocols = ["early_to_late_transfer", "late_window_holdout"]
    variants = ["hi-small", "hi-medium", "li-small", "li-medium"]
    for variant in variants:
        for protocol in protocols:
            g = baseline[(baseline.variant == variant) & (baseline.protocol == protocol)].set_index("config")
            rows.append(
                {
                    "panel": "A",
                    "variant": variant,
                    "protocol": "Early" if protocol.startswith("early") else "Late",
                    "col1": summary(g.loc["hist_gradient_boosting_edge_features", "auprc_mean"], g.loc["hist_gradient_boosting_edge_features", "auprc_std"]),
                    "col2": summary(g.loc["logistic_regression_edge_features", "auprc_mean"], g.loc["logistic_regression_edge_features", "auprc_std"]),
                    "col3": summary(g.loc["graphsage_edge_minibatch_h32", "auprc_mean"], g.loc["graphsage_edge_minibatch_h32", "auprc_std"]),
                    "col4": "HistGB",
                    "col5": CONFIG[ranks[(ranks.family == "baseline_grid") & (ranks.variant == variant) & (ranks.protocol == protocol)].iloc[0].f1_winner],
                }
            )
    for variant in variants:
        for protocol in protocols:
            g = graph[(graph.variant == variant) & (graph.protocol == protocol)].set_index("config")
            rank = ranks[(ranks.family == "graph_grid") & (ranks.variant == variant) & (ranks.protocol == protocol)].iloc[0]
            ap = g.loc[rank.auprc_winner]
            f1 = g.loc[rank.f1_winner]
            roc = g.loc[rank.auroc_winner]
            ref = g.loc["edge_aware_graphsage_h64"]
            rows.append(
                {
                    "panel": "B",
                    "variant": variant,
                    "protocol": "Early" if protocol.startswith("early") else "Late",
                    "col1": summary(ref.auprc_mean, ref.auprc_std),
                    "col2": f"{CONFIG[rank.auprc_winner]} {summary(ap.auprc_mean,ap.auprc_std)}",
                    "col3": f"{CONFIG[rank.auroc_winner]} {summary(roc.auroc_mean,roc.auroc_std)}",
                    "col4": f"{CONFIG[rank.f1_winner]} {summary(f1.f1_mean,f1.f1_std)}",
                    "col5": "Small only" if variant.endswith("small") else "GINE blocked",
                }
            )
    df = pd.DataFrame(rows)
    panel_a = df[df.panel.eq("A")]
    panel_b = df[df.panel.eq("B")]
    body_a = "\n".join(f"{esc(r.variant)} & {r.protocol} & {r.col1} & {r.col2} & {r.col3} & {r.col4} & {r.col5} \\\\" for _, r in panel_a.iterrows())
    body_b = "\n".join(f"{esc(r.variant)} & {r.protocol} & {r.col1} & {r.col2} & {r.col3} & {r.col4} & {esc(r.col5)} \\\\" for _, r in panel_b.iterrows())
    path = TABLES / "table05_ibm_results.tex"
    write(
        path,
        r"""\begin{table*}[t]
\caption{IBM AML-Data results (mean $\pm$ SD, ten seeds). Panel A reports every baseline-grid AUPRC. Panel B reports the h64 reference AUPRC and the best feasible configuration/value for each metric; Fig.~\ref{fig:ibm-results} shows all graph configurations and intervals. Winners are computed only within identical variant/protocol feasibility sets.}
\label{tab:ibm-results}
\centering
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\textbf{A. Baseline grid: AUPRC and metric winners}\\[-1mm]
\begin{tabular}{@{}llccccc@{}}
\toprule
Variant & Protocol & HistGB AUPRC & LR AUPRC & SAGE-h32 AUPRC & AUPRC winner & F1 winner \\
\midrule
""" + body_a + r"""
\bottomrule
\end{tabular}

\vspace{1mm}
\textbf{B. Graph grid: reference and feasible metric winners}\\[-1mm]
\begin{tabular}{@{}llccccc@{}}
\toprule
Variant & Protocol & Ref-h64 AUPRC & Best AUPRC & Best AUROC & Best F1@.5 & Feasibility note \\
\midrule
""" + body_b + r"""
\bottomrule
\end{tabular}
\end{table*}""",
    )
    return df, path


def main_resources() -> tuple[pd.DataFrame, Path]:
    df = pd.read_csv(OUT / "RESOURCE_BOUNDARIES.csv")
    path = TABLES / "table06_resource_boundaries.tex"
    body = "\n".join(
        f"{esc(r.cell)} & {esc(r.resource_envelope_or_reason)} & {esc(r.result_outputs)} / {esc(r.prediction_exports)} & {esc(status_display(r.status))} & {esc(r.interpretation)} \\\\"
        for _, r in df.iterrows()
    )
    write(
        path,
        r"""\begin{table*}[t]
\caption{Visible resource boundaries. A blocked cell is unmeasured, not a low-performing point. It is excluded from ranks, matched means, and Pareto comparisons.}
\label{tab:resource-boundaries}
\centering
\scriptsize
\begin{tabularx}{\textwidth}{@{}p{0.19\textwidth}p{0.20\textwidth}p{0.09\textwidth}p{0.19\textwidth}X@{}}
\toprule
Cell & Envelope / reason & Result / pred. files & Status & Interpretation \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""",
    )
    return df, path


def main_graphsafe() -> tuple[pd.DataFrame, Path]:
    full = pd.read_csv(OUT / "GRAPHSAFE_BOUNDED_SUMMARY.csv")
    df = full[full.method.isin(["best_val_branch", "simple_average", "graphsafe_conservative"])].copy()
    names = {"best_val_branch": "Best val branch", "simple_average": "Simple avg.", "graphsafe_conservative": "GraphSafe cons."}
    body = "\n".join(
        f"{esc(r.dataset.title())} & {names[r.method]} & {f3(r.f1_mean)} & {f3(r.recall_at_1pct_mean)} & {f3(r.cost_sensitive_risk_mean)} \\\\"
        for _, r in df.iterrows()
    )
    path = TABLES / "table07_graphsafe_case.tex"
    write(
        path,
        r"""\begin{table}[t]
\caption{Bounded saved-output case study. Each value is the mean over ten seed blocks; within a seed, six protocol--model contexts are averaged before inference. Lower cost risk is better. Holm-corrected paired tests do not support universal GraphSafe dominance.}
\label{tab:graphsafe}
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llrrr@{}}
\toprule
Data & Method & F1 & R@1\% & Risk \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}""",
    )
    return df, path


def longtable(path: Path, caption: str, label: str, columns: str, headers: list[str], rows: Iterable[list[Any]], *, size: str = r"\scriptsize") -> None:
    header = " & ".join(header_tex(value) for value in headers) + r" \\"
    body = "\n".join(" & ".join(esc(value) for value in row) + r" \\" for row in rows)
    write(
        path,
        rf"""{{{size}
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{longtable}}{{@{{}}{columns}@{{}}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\multicolumn{{{len(headers)}}}{{c}}{{\tablename\ \thetable\ continued}}\\
\toprule
{header}
\midrule
\endhead
\midrule
\multicolumn{{{len(headers)}}}{{r}}{{continued}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
}}""",
    )


def supplement_tables() -> list[tuple[str, Path, str]]:
    outputs: list[tuple[str, Path, str]] = []

    datasets = pd.read_csv(OUT / "DATASET_TASK_STATISTICS.csv")
    p = SUPP / "table_s01_dataset_cards.tex"
    longtable(p, "Full task cards and split statistics.", "tab:s-datasets", "p{0.08\\textwidth}p{0.07\\textwidth}p{0.08\\textwidth}rrrrrrr", ["Dataset", "Variant", "Unit", "Nodes", "Edges/tx", "Train", "Val", "Test", "Test +", "Prior"], ([r.dataset, r.variant, r.prediction_unit, int(r.nodes_or_accounts), int(r.source_edges_or_transactions), int(r.train_units), int(r.validation_units), int(r.test_units), int(r.test_positives), f"{100*r.test_positive_rate:.4f}%"] for _, r in datasets.iterrows()), size=r"\tiny")
    outputs.append(("S01", p, "DATASET_TASK_STATISTICS.csv"))

    protocols = pd.read_csv(OUT / "PROTOCOL_DEFINITIONS.csv")
    p = SUPP / "table_s02_protocols.tex"
    longtable(p, "Complete protocol contracts.", "tab:s-protocols", "p{0.13\\textwidth}p{0.14\\textwidth}p{0.29\\textwidth}p{0.14\\textwidth}p{0.13\\textwidth}p{0.10\\textwidth}", list(protocols.columns), protocols.itertuples(index=False, name=None), size=r"\tiny")
    outputs.append(("S02", p, "PROTOCOL_DEFINITIONS.csv"))

    models = pd.read_csv(OUT / "MODEL_CONSTRUCTION_INVENTORY.csv")
    p = SUPP / "table_s03_models.tex"
    longtable(p, "Full model and graph-construction inventory.", "tab:s-models", "p{0.16\\textwidth}p{0.27\\textwidth}p{0.27\\textwidth}p{0.23\\textwidth}", list(models.columns), models.itertuples(index=False, name=None), size=r"\tiny")
    outputs.append(("S03", p, "MODEL_CONSTRUCTION_INVENTORY.csv"))

    training = pd.read_csv(OUT / "TRAINING_CONFIGURATION.csv")
    p = SUPP / "table_s04_training.tex"
    cols = "p{0.10\\textwidth}p{0.07\\textwidth}p{0.045\\textwidth}p{0.055\\textwidth}p{0.10\\textwidth}p{0.065\\textwidth}p{0.075\\textwidth}p{0.075\\textwidth}p{0.10\\textwidth}p{0.10\\textwidth}p{0.115\\textwidth}"
    training_headers = ["Family", "Opt.", "LR", "WD", "Loss", "Epochs / iter.", "Patience", "Selection", "Threshold", "Batch / forward", "Architecture"]
    longtable(p, "Optimization, selection, and threshold settings.", "tab:s-training", cols, training_headers, training.itertuples(index=False, name=None), size=r"\tiny")
    outputs.append(("S04", p, "TRAINING_CONFIGURATION.csv"))

    rb09 = pd.read_csv(ROOT / "results/runs_rb09v3/runs.csv")
    p = SUPP / "table_s05_rb09_seed.tex"
    longtable(p, "Complete RB09 seed-level metric table.", "tab:s-rb09-seed", "lllrrrrrrr", ["Data", "Protocol", "Model", "Seed", "F1", "AUROC", "AUPRC", "P@1%", "R@1%", "Runtime"], ([r.dataset, r.protocol, r.model, r.seed, f4(r.f1), f4(r.auroc), f4(r.auprc), f4(r.precision_at_1pct), f4(r.recall_at_1pct), f3(r.runtime_seconds)] for _, r in rb09.iterrows()), size=r"\tiny")
    outputs.append(("S05", p, "results/runs_rb09v3/runs.csv"))

    effects = pd.read_csv(OUT / "RB09_PROTOCOL_EFFECTS.csv")
    p = SUPP / "table_s06_rb09_effects.tex"
    longtable(p, "All paired RB09 strict-to-isolated effects.", "tab:s-rb09-effects", "lllrrrrrrrr", ["Data", "Model", "Metric", "$n$", "Strict", "Isolated", "$\\Delta$", "CI low", "CI high", "$d_z$", "Holm p"], ([r.dataset, r.model, r.metric, r.n_pairs, f4(r.strict_mean), f4(r.isolated_mean), f4(r.delta_isolated_minus_strict), f4(r.delta_ci95_low), f4(r.delta_ci95_high), f3(r.cohen_dz), pvalue(r.holm_p_within_metric)] for _, r in effects.iterrows()), size=r"\tiny")
    outputs.append(("S06", p, "RB09_PROTOCOL_EFFECTS.csv"))

    dup = pd.read_csv(OUT / "V24_DUPLICATE_STRESS_AUDIT.csv")
    dup_summary = dup.groupby(["dataset", "protocol", "model"], as_index=False).agg(
        base_cells=("seed", "count"),
        labels=("n_stress_labels", "min"),
        all_metrics_identical=("all_performance_metrics_identical", "all"),
    )
    p = SUPP / "table_s07_v24_duplicate_audit.tex"
    longtable(p, "V24 RB41 construct-validity audit. Every base cell has three metadata labels and identical performance metrics.", "tab:s-v24-duplicate", "lllrrl", ["Data", "Protocol", "Model", "Base cells", "Labels/cell", "All metrics identical"], dup_summary.itertuples(index=False, name=None), size=r"\scriptsize")
    outputs.append(("S07", p, "V24_DUPLICATE_STRESS_AUDIT.csv"))

    v22_lanes = pd.read_csv(ROOT / "manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv")
    p = SUPP / "table_s08_v22_lanes.tex"
    lane_names = {
        "RB28_DGRAPHFIN_LOSS_ROBUSTNESS_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin loss robustness",
        "RB28_ELLIPTIC_LOSS_ROBUSTNESS_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic loss robustness",
        "RB29_DGRAPHFIN_NEGATIVE_CONTROLS_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin negative controls",
        "RB29_ELLIPTIC_NEGATIVE_CONTROLS_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic negative controls",
        "RB30_DGRAPHFIN_EXTRA_ARCH_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin fixed GAT h64/l2",
        "RB30_ELLIPTIC_EXTRA_ARCH_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic GAT/GIN",
    }
    longtable(p, "Canonical V22 full-grid lanes; raw lane identifiers are mapped in Table~\\ref{tab:s-families}.", "tab:s-v22-lanes", "p{0.31\\textwidth}p{0.18\\textwidth}rrrrr", ["Lane", "Status", "JSON", "Expected", "Pred.", "Expected", "Rows"], ([lane_names.get(r.lane_id, r.lane_id), status_display(r.status), r.actual_json, r.expected_json, r.actual_prediction_csv, r.expected_prediction_csv, r.prediction_rows] for _, r in v22_lanes.iterrows()), size=r"\tiny")
    outputs.append(("S08", p, "manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv"))

    v22_stats = pd.read_csv(ROOT / "manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv")
    p = SUPP / "table_s09_v22_stats.tex"
    longtable(p, "Complete V22 paired statistical tests (model identity retained).", "tab:s-v22-stats", "llllllllrrrrr", ["Family", "Data", "Protocol", "Model", "Axis", "Left", "Right", "Metric", "$n$", "$\\Delta$", "CI low", "CI high", "BH p"], ([r.family, r.dataset, r.protocol, r.fixed_model, r.comparison_axis, r.left, r.right, r.metric, r.n, f4(r.mean_diff_left_minus_right), f4(r.bootstrap_ci95_low), f4(r.bootstrap_ci95_high), pvalue(r.p_value_bh)] for _, r in v22_stats.iterrows()), size=r"\tiny")
    outputs.append(("S09", p, "manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv"))

    ibm_seed = pd.read_csv(OUT / "IBM_IMPORTED_SEED_ROWS.csv")
    p = SUPP / "table_s10_ibm_seed.tex"
    longtable(p, "Complete IBM AML-Data seed-level results (840 rows).", "tab:s-ibm-seed", "lllllrrrr", ["Version", "Variant", "Protocol", "Config", "Seed", "AUPRC", "AUROC", "F1", "Runtime"], ([r.version, r.variant, r.protocol, CONFIG.get(r.config, r.config), r.seed, f4(r.auprc), f4(r.auroc), f4(r.f1), f3(r.runtime_seconds)] for _, r in ibm_seed.iterrows()), size=r"\tiny")
    outputs.append(("S10", p, "IBM_IMPORTED_SEED_ROWS.csv"))

    ibm_cell = pd.read_csv(OUT / "IBM_CELL_SUMMARY.csv")
    p = SUPP / "table_s11_ibm_cells.tex"
    longtable(p, "Complete IBM AML-Data cell aggregates and intervals.", "tab:s-ibm-cells", "llllrrrrrr", ["V", "Variant", "Protocol", "Config", "$n$", "AUPRC", "CI low", "CI high", "AUROC", "F1"], ([r.version, r.variant, r.protocol, CONFIG.get(r.config, r.config), r.n, f4(r.auprc_mean), f4(r.auprc_ci95_low), f4(r.auprc_ci95_high), f4(r.auroc_mean), f4(r.f1_mean)] for _, r in ibm_cell.iterrows()), size=r"\tiny")
    outputs.append(("S11", p, "IBM_CELL_SUMMARY.csv"))

    ranks = pd.read_csv(OUT / "IBM_RANK_DIVERGENCE.csv")
    p = SUPP / "table_s12_ranks.tex"
    longtable(p, "Rank-versus-decision disagreement by exact feasibility set.", "tab:s-ranks", "lllrrlllrr", ["Family", "Variant", "Protocol", "$m$", "AUPRC/F1 differ", "AUPRC winner", "AUROC winner", "F1 winner", "$\\rho_{AP,F1}$", "$\\rho_{ROC,AP}$"], ([r.family, r.variant, r.protocol, r.n_configurations, r.auprc_f1_winner_disagree, CONFIG.get(r.auprc_winner, r.auprc_winner), CONFIG.get(r.auroc_winner, r.auroc_winner), CONFIG.get(r.f1_winner, r.f1_winner), f3(r.spearman_auprc_vs_f1), f3(r.spearman_auroc_vs_auprc)] for _, r in ranks.iterrows()), size=r"\tiny")
    outputs.append(("S12", p, "IBM_RANK_DIVERGENCE.csv"))

    abl = pd.read_csv(OUT / "IBM_MATCHED_ABLATION_EFFECTS.csv")
    p = SUPP / "table_s13_ablation.tex"
    longtable(p, "Matched IBM graph-ablation effects. The 40 raw context--seed pairs are reduced to ten seed blocks before interval estimation and testing.", "tab:s-ablation", "lllrrrrrrrrr", ["Config", "Size", "Metric", "Raw", "Blocks", "Ref", "Candidate", "$\\Delta$", "CI low", "CI high", "$d_z$", "Holm p"], ([CONFIG.get(r.config, r.config), r["size"], r.metric, r.n_raw_context_seed_pairs, r.n_pairs, f4(r.reference_mean), f4(r.candidate_mean), f4(r.mean_delta), f4(r.delta_ci95_low), f4(r.delta_ci95_high), f3(r.cohen_dz), pvalue(r.holm_p_within_size_metric)] for _, r in abl.iterrows()), size=r"\tiny")
    outputs.append(("S13", p, "IBM_MATCHED_ABLATION_EFFECTS.csv"))

    context_abl = pd.read_csv(OUT / "IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv")
    p = SUPP / "table_s13b_ablation_contexts.tex"
    longtable(p, "Context-specific IBM graph-ablation sensitivity. These ten-seed rows expose heterogeneity; the seed-blocked fixed-grid tests are in Table~\\ref{tab:s-ablation}.", "tab:s-ablation-contexts", "lllllrrrrrrr", ["Config", "Size", "Metric", "Variant", "Protocol", "$n$", "Ref", "Candidate", "$\\Delta$", "CI low", "CI high", "p (desc.)"], ([CONFIG.get(r.config, r.config), r["size"], r.metric, r.variant, r.protocol, r.n_seed_pairs, f4(r.reference_mean), f4(r.candidate_mean), f4(r.mean_delta), f4(r.delta_ci95_low), f4(r.delta_ci95_high), pvalue(r.wilcoxon_p_descriptive)] for _, r in context_abl.iterrows()), size=r"\tiny")
    outputs.append(("S13b", p, "IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv"))

    runtime = pd.read_csv(OUT / "IBM_RUNTIME_FEASIBILITY.csv")
    p = SUPP / "table_s14_runtime.tex"
    longtable(p, "Configuration-specific performance, runtime, Pareto, and blocked status.", "tab:s-runtime", "lllllrrrrl", ["Variant", "Protocol", "Config", "Status", "$n$", "AUPRC", "AUROC", "F1", "Runtime", "Pareto"], ([r.variant, r.protocol, CONFIG.get(r.config, r.config), status_display(r.status), r.n, f4(r.auprc_mean), f4(r.auroc_mean), f4(r.f1_mean), f3(r.runtime_seconds_mean), r.pareto_auprc_runtime_within_cell] for _, r in runtime.iterrows()), size=r"\tiny")
    outputs.append(("S14", p, "IBM_RUNTIME_FEASIBILITY.csv"))

    gs = pd.read_csv(OUT / "GRAPHSAFE_BOUNDED_SUMMARY.csv")
    p = SUPP / "table_s15_graphsafe.tex"
    longtable(p, "GraphSafe and comparator aggregate summary.", "tab:s-graphsafe", "llrrrrrr", ["Data", "Method", "$n$", "F1", "AUPRC", "R@1%", "Cost risk", "Worst regret"], ([r.dataset, r.method, r.n, f4(r.f1_mean), f4(r.auprc_mean), f4(r.recall_at_1pct_mean), f4(r.cost_sensitive_risk_mean), f4(r.worst_block_regret_mean)] for _, r in gs.iterrows()), size=r"\tiny")
    outputs.append(("S15", p, "GRAPHSAFE_BOUNDED_SUMMARY.csv"))

    gsp = pd.read_csv(OUT / "GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE.csv")
    p = SUPP / "table_s16_graphsafe_tests.tex"
    longtable(p, "Paired GraphSafe-family contrasts against simple averaging.", "tab:s-graphsafe-tests", "llllrrrrr", ["Data", "Method", "Metric", "Direction", "$n$", "Improvement", "CI low", "CI high", "Holm p"], ([r.dataset, r.method, r.metric, r.direction, r.n_pairs, f4(r.mean_improvement), f4(r.ci95_low), f4(r.ci95_high), pvalue(r.holm_p_all_graphsafe_vs_average)] for _, r in gsp.iterrows()), size=r"\tiny")
    outputs.append(("S16", p, "GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE.csv"))

    budget = pd.read_csv(OUT / "REVIEW_BUDGET_CURVES.csv")
    p = SUPP / "table_s17_review_budget.tex"
    longtable(p, "Fixed review-budget precision and recall.", "tab:s-review-budget", "llrrrrr", ["Data", "Method", "Budget %", "$n$", "Precision", "Recall", "Std(P/R)"], ([r.dataset, r.method, r.budget_pct, r.n, f4(r.precision_mean), f4(r.recall_mean), f"{f4(r.precision_std)} / {f4(r.recall_std)}"] for _, r in budget.iterrows()), size=r"\tiny")
    outputs.append(("S17", p, "REVIEW_BUDGET_CURVES.csv"))

    resources = pd.read_csv(OUT / "RESOURCE_BOUNDARIES.csv")
    p = SUPP / "table_s18_resources.tex"
    longtable(p, "Complete resource-boundary records.", "tab:s-resources", "p{0.21\\textwidth}p{0.20\\textwidth}p{0.10\\textwidth}p{0.20\\textwidth}p{0.22\\textwidth}", ["Cell", "Envelope/reason", "Outputs", "Status", "Interpretation"], ([r.cell, r.resource_envelope_or_reason, f"{r.result_outputs} / {r.prediction_exports}", status_display(r.status), r.interpretation] for _, r in resources.iterrows()), size=r"\tiny")
    outputs.append(("S18", p, "RESOURCE_BOUNDARIES.csv"))

    claims = pd.read_csv(OUT / "CLAIM_EVIDENCE_LEDGER.csv")
    p = SUPP / "table_s19_claim_ledger.tex"
    longtable(p, "Typed claim ledger and allowed language.", "tab:s-claims", "p{0.04\\textwidth}p{0.14\\textwidth}p{0.25\\textwidth}p{0.25\\textwidth}p{0.25\\textwidth}", ["ID", "Status", "Scope", "Permitted", "Prohibited"], ([r.claim_id, status_display(r.support_status), r.scope, r.permitted_wording, r.prohibited_wording] for _, r in claims.iterrows()), size=r"\tiny")
    outputs.append(("S19", p, "CLAIM_EVIDENCE_LEDGER.csv"))

    inventory = pd.read_csv(OUT / "EVIDENCE_INVENTORY.csv")
    family = inventory.groupby(["family", "eligibility"], as_index=False).agg(
        cells=("evidence_id", "count"),
        datasets=("dataset", lambda x: ";".join(sorted(set(map(str, x))))),
        locks=("evidence_lock", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    write_data("EVIDENCE_FAMILY_SUMMARY", family)
    p = SUPP / "table_s20_evidence_families.tex"
    longtable(p, "Run-family to scientific-experiment and lock mapping.", "tab:s-families", "p{0.22\\textwidth}p{0.12\\textwidth}rp{0.20\\textwidth}p{0.37\\textwidth}", ["Family", "Eligibility", "Cells", "Datasets", "Lock(s)"], family.itertuples(index=False, name=None), size=r"\tiny")
    outputs.append(("S20", p, "EVIDENCE_FAMILY_SUMMARY.csv"))

    val = pd.read_csv(OUT / "FRAMEWORK_VALIDATION_CASES.csv")
    p = SUPP / "table_s21_framework_validation.tex"
    longtable(p, "Claim mutation and evidence-ablation validation cases.", "tab:s-framework", "llp{0.38\\textwidth}rrp{0.20\\textwidth}l", ["Case", "Claim", "Mutation", "Req.", "Obs.", "Status", "Pass"], ([r.case_id, r.base_claim, r.mutation, r.required_evidence_units, r.observed_evidence_units, status_display(r.observed_status), r["pass"]] for _, r in val.iterrows()), size=r"\tiny")
    outputs.append(("S21", p, "FRAMEWORK_VALIDATION_CASES.csv"))

    fp = pd.read_csv(OUT / "FALSE_PROMOTION_AUDIT.csv")
    p = SUPP / "table_s22_false_promotion.tex"
    longtable(p, "False-promotion prevention audit.", "tab:s-false-promotion", "llrrp{0.22\\textwidth}p{0.18\\textwidth}p{0.25\\textwidth}", ["ID", "Family", "Results", "Preds", "Naive rule", "Correct status", "Reason"], fp.itertuples(index=False, name=None), size=r"\tiny")
    outputs.append(("S22", p, "FALSE_PROMOTION_AUDIT.csv"))
    return outputs


def main() -> None:
    import sys

    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.tkde_visual_rebuild.build_curated_supplement_tables import (
        main as build_supplement,
    )
    from scripts.tkde_visual_rebuild.build_main_tables import build as build_main

    main_provenance = build_main()
    result = build_supplement()
    if result != 0:
        raise RuntimeError(f"Curated supplement generator failed with exit code {result}")
    print(
        f"generated {len(main_provenance)} readable main tables and the curated "
        "portrait supplement through the V2 compatibility entry point"
    )


if __name__ == "__main__":
    main()
