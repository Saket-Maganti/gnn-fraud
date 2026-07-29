#!/usr/bin/env python3
"""Build readable, evidence-preserving tables for the TKDE main paper."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results" / "tkde_rebuild"
OUT = ROOT / "results" / "tkde_visual_rebuild"
DATA = OUT / "table_data"
TABLES = ROOT / "paper_tkde" / "tables"
ROW_END = r" \\"


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


def dataset_name(value: str) -> str:
    """Return publication capitalization for the frozen dataset identifiers."""

    return {
        "dgraphfin": "DGraphFin",
        "elliptic": "Elliptic",
    }.get(str(value).lower(), str(value))


def esc(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\textsc{n/a}"
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
    }
    return "".join(replacements.get(char, char) for char in text)


def f4(value: float) -> str:
    return r"\textsc{n/a}" if pd.isna(value) else f"{float(value):.4f}"


def f3(value: float) -> str:
    return r"\textsc{n/a}" if pd.isna(value) else f"{float(value):.3f}"


def pvalue(value: float) -> str:
    if pd.isna(value):
        return r"\textsc{n/a}"
    return f"{float(value):.3f}" if value >= 0.001 else f"{float(value):.1e}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_data(stem: str, frame: pd.DataFrame) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{stem}.csv"
    frame.to_csv(path, index=False)
    return path


def related_work() -> tuple[pd.DataFrame, Path, list[str]]:
    rows = [
        ["GADBench", "No", "Limited", "No", "No", "Benchmark", "No"],
        ["TGB / BenchTemp", "Yes", "Inductive tasks", "No", "Efficiency", "Benchmark", "No"],
        ["BAG", "Yes", "Task/model breadth", "No", "Scale", "Benchmark", "No"],
        ["GAD in the Wild", "Partial", "Scale/missing attrs.", "No", "Scale", "Limited", "No"],
        ["BetterBench / Eval Factsheets", "Generic", "Documented", "Documented", "Documented", "Metadata", "Descriptive"],
        ["FraudShiftBench", "Yes", "Visibility + construction", "Top-$K$ + cost", "Typed status", "Per-seed", "Executable"],
    ]
    columns = ["work", "temporal", "graph_contract", "capacity", "resources", "predictions", "claim_support"]
    frame = pd.DataFrame(rows, columns=columns)
    body = "\n".join(" & ".join(esc(v) if "$" not in str(v) else str(v) for v in row) + r" \\" for row in rows)
    path = TABLES / "table01_related_work.tex"
    write(path, r"""\begin{table*}[t]
\caption{Closest benchmark families and the distinctions needed here. ``No'' means that the feature is not a stated benchmark axis; it does not question the rigor of the cited work.}
\label{tab:related-comparison}
\centering
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabularx}{\textwidth}{@{}p{0.16\textwidth}c p{0.19\textwidth}p{0.10\textwidth}p{0.11\textwidth}p{0.10\textwidth}X@{}}
\toprule
Work & Temporal & Graph contract & Capacity & Resources & Predictions & Claim support \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""")
    return frame, path, ["results/tkde_rebuild/LITERATURE_MATRIX.csv"]


def dataset_tasks() -> tuple[pd.DataFrame, Path, list[str]]:
    source = pd.read_csv(FROZEN / "DATASET_TASK_STATISTICS.csv")
    rows: list[dict[str, str]] = []
    for dataset in ("Elliptic", "DGraphFin"):
        r = source[source.dataset.eq(dataset)].iloc[0]
        rows.append({
            "dataset": dataset,
            "unit": "transaction node" if dataset == "Elliptic" else "account node",
            "scale": f"{int(r.nodes_or_accounts):,} / {int(r.source_edges_or_transactions):,}",
            "time": "49 steps; 30/4/15" if dataset == "Elliptic" else "20 buckets; 14/2/4",
            "prior": f"{100*r.test_positive_rate:.3f}\\%",
            "features": f"{int(r.node_feature_dim)} node",
            "protocols": "strict; isolated; transductive",
            "scope": "real; node task",
        })
    ibm = source[source.dataset.eq("IBM AML-Data")]
    for variant in ("hi-small", "hi-medium", "li-small", "li-medium"):
        group = ibm[ibm.variant.eq(variant)]
        r = group.iloc[0]
        rows.append({
            "dataset": f"IBM {variant.upper()}",
            "unit": "transaction edge",
            "scale": f"{int(r.nodes_or_accounts):,} / {int(r.source_edges_or_transactions):,}",
            "time": "50/20/30; 60/20/20",
            "prior": f"{100*group.test_positive_rate.min():.3f}--{100*group.test_positive_rate.max():.3f}\\%",
            "features": "8 node + 8 edge",
            "protocols": "early-to-late; late holdout",
            "scope": "synthetic; Small/Medium",
        })
    frame = pd.DataFrame(rows)
    body = "\n".join(
        f"{esc(r.dataset)} & {esc(r.unit)} & {esc(r.scale)} & {esc(r.time)} & {r.prior} & {esc(r.features)} & {esc(r.protocols)} & {esc(r.scope)}" + ROW_END
        for r in frame.itertuples(index=False)
    )
    path = TABLES / "table02_dataset_tasks.tex"
    write(path, r"""\begin{table*}[t]
\caption{Dataset and task cards. Scale is nodes/accounts followed by source edges/transactions. The prediction units, time representations, prevalences, and empirical scopes differ, so values are not pooled into one leaderboard.}
\label{tab:datasets}
\centering
\footnotesize
\setlength{\tabcolsep}{2.8pt}
\begin{tabularx}{\textwidth}{@{}p{0.105\textwidth}p{0.095\textwidth}p{0.13\textwidth}p{0.115\textwidth}p{0.07\textwidth}p{0.105\textwidth}p{0.15\textwidth}X@{}}
\toprule
Dataset & Unit & Scale & Time / split & Test prior & Features & Protocols & Empirical scope \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""")
    return frame, path, ["results/tkde_rebuild/DATASET_TASK_STATISTICS.csv"]


def protocol_visibility() -> tuple[pd.DataFrame, Path, list[str]]:
    rows = [
        ["Strict inductive", "chronological", "train period", "held-out structure only at evaluation", "validation F1", "argmax / rank", "recorded CPU/CUDA"],
        ["Isolated inductive", "same masks", "train period", "remove train--held-out edges", "validation F1", "argmax / rank", "same as strict"],
        ["Transductive", "same masks", "full label-free graph", "full graph; labels hidden", "validation F1", "diagnostic", "recorded CPU/CUDA"],
        ["IBM early-to-late", "50/20/30 labels", "first 60\% covariates", "shared 60\% account history", "fixed schedule", r"$\tau=0.5$ / budgets", "CPU + T4 guards"],
        ["IBM late holdout", "60/20/20 labels", "first 60\% covariates", "training-period account history", "fixed schedule", r"$\tau=0.5$ / budgets", "CPU + T4 guards"],
    ]
    columns = ["protocol", "label_windows", "label_free_history", "graph_visibility", "selection", "decision", "resources"]
    frame = pd.DataFrame(rows, columns=columns)
    body = "\n".join(" & ".join(str(v) if ("$" in str(v) or "\\%" in str(v)) else esc(v) for v in row) + r" \\" for row in rows)
    path = TABLES / "table03_protocol_visibility.tex"
    write(path, r"""\begin{table*}[t]
\caption{Independent deployment-contract coordinates. IBM early-to-late uses labels from the first 50\% but the shared label-free account-history map from the first 60\%; it is not a first-50\%-only feature contract.}
\label{tab:protocol-visibility}
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}p{0.12\textwidth}p{0.12\textwidth}p{0.14\textwidth}p{0.19\textwidth}p{0.11\textwidth}p{0.12\textwidth}X@{}}
\toprule
Protocol & Label windows & Label-free history & Graph visibility & Selection & Decision & Resource envelope \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""")
    return frame, path, ["results/tkde_rebuild/PROTOCOL_DEFINITIONS.csv"]


def protocol_effects() -> tuple[pd.DataFrame, Path, list[str]]:
    frame = pd.read_csv(FROZEN / "RB09_AUPRC_MAIN.csv")
    labels = {"gcn": "GCN", "mlp": "MLP", "sage": "GraphSAGE"}
    body = []
    for r in frame.itertuples(index=False):
        body.append(
            f"{esc(dataset_name(r.dataset))} & {labels[r.model]} & {r.strict_mean:.4f} & {r.isolated_mean:.4f} & {r.delta_isolated_minus_strict:+.4f} & [{r.delta_ci95_low:+.4f}, {r.delta_ci95_high:+.4f}] & {r.relative_delta_pct:+.1f} & {r.cohen_dz:+.2f} & {pvalue(r.holm_p_within_metric)}" + ROW_END
        )
    path = TABLES / "table04_rb09_protocol_effects.tex"
    write(path, r"""\begin{table*}[t]
\caption{Paired AUPRC effects of strict versus isolated graph visibility over ten shared seeds. $\Delta$ is isolated minus strict; intervals are deterministic 95\% bootstrap intervals. Holm correction covers the six AUPRC comparisons.}
\label{tab:rb09-effects}
\centering
\footnotesize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Dataset & Model & Strict & Isolated & $\Delta$ & 95\% CI & Rel. $\Delta$ (\%) & $d_z$ & Holm $p$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}""")
    return frame, path, ["results/tkde_rebuild/RB09_AUPRC_MAIN.csv"]


def ibm_baselines() -> tuple[pd.DataFrame, Path, list[str]]:
    source = pd.read_csv(FROZEN / "IBM_CELL_SUMMARY.csv")
    source = source[source.version.eq("V26")]
    rows: list[dict[str, Any]] = []
    for variant in ("hi-small", "hi-medium", "li-small", "li-medium"):
        for protocol in ("early_to_late_transfer", "late_window_holdout"):
            cell = source[source.variant.eq(variant) & source.protocol.eq(protocol)].set_index("config")
            values = {
                name: float(cell.loc[config, "auprc_mean"])
                for config, name in [
                    ("hist_gradient_boosting_edge_features", "histgb"),
                    ("logistic_regression_edge_features", "lr"),
                    ("graphsage_edge_minibatch_h32", "sage"),
                ]
            }
            ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            rows.append({
                "cell": f"{variant.upper()} / {'Early' if protocol.startswith('early') else 'Late'}",
                **values,
                "best": ordered[0][0],
                "best_second_delta": ordered[0][1] - ordered[1][1],
            })
    frame = pd.DataFrame(rows)
    names = {"histgb": "HistGB", "lr": "LR", "sage": "SAGE-h32"}
    body = []
    for r in frame.itertuples(index=False):
        vals = {"histgb": r.histgb, "lr": r.lr, "sage": r.sage}
        cells = [f"\\textbf{{{f4(value)}}}" if key == r.best else f4(value) for key, value in vals.items()]
        body.append(f"{esc(r.cell)} & {' & '.join(cells)} & {names[r.best]} & {r.best_second_delta:.4f}" + ROW_END)
    path = TABLES / "table05_ibm_baselines.tex"
    write(path, r"""\begin{table}[t]
\caption{IBM baseline AUPRC means over ten runs. Bold marks the best of the three recorded families; Fig.~\ref{fig:ibm-baselines} gives uncertainty.}
\label{tab:ibm-baselines}
\centering
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Cell & HistGB & LR & SAGE & Best & Gap \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}""")
    return frame, path, ["results/tkde_rebuild/IBM_CELL_SUMMARY.csv"]


def ibm_construction() -> tuple[pd.DataFrame, Path, list[str]]:
    cells = pd.read_csv(FROZEN / "IBM_CELL_SUMMARY.csv")
    effects = pd.read_csv(FROZEN / "IBM_MATCHED_ABLATION_EFFECTS.csv")
    graph = cells[(cells.version.isin(["V27", "V28"])) & ~cells.config.eq("account_account_sender_receiver")]
    rows: list[dict[str, Any]] = []
    for variant in ("hi-small", "hi-medium", "li-small", "li-medium"):
        for protocol in ("early_to_late_transfer", "late_window_holdout"):
            cell = graph[graph.variant.eq(variant) & graph.protocol.eq(protocol)]
            ref = cell[cell.config.eq("edge_aware_graphsage_h64")].iloc[0]
            candidates = cell[~cell.config.eq("edge_aware_graphsage_h64")].sort_values(["auprc_mean", "config"], ascending=[False, True])
            best = candidates.iloc[0]
            ap_test = effects[(effects.config.eq(best.config)) & effects["size"].eq(best["size"]) & effects.metric.eq("auprc")].iloc[0]
            rows.append({
                "cell": f"{variant.upper()} / {'Early' if protocol.startswith('early') else 'Late'}",
                "size": best["size"],
                "reference_auprc": ref.auprc_mean,
                "best_candidate": CONFIG[best.config],
                "auprc_delta": best.auprc_mean - ref.auprc_mean,
                "auroc_delta": best.auroc_mean - ref.auroc_mean,
                "f1_delta": best.f1_mean - ref.f1_mean,
                "auprc_holm_p_seed_block": ap_test.holm_p_within_size_metric,
                "feasibility": "Small measured" if best["size"] == "small" else "GINE Medium blocked",
            })
    frame = pd.DataFrame(rows)
    body = "\n".join(
        f"{esc(r.cell)} & {f4(r.reference_auprc)} & {r.best_candidate} & {r.auprc_delta:+.4f} & {r.auroc_delta:+.4f} & {r.f1_delta:+.4f} & {pvalue(r.auprc_holm_p_seed_block)} & {esc(r.feasibility)}" + ROW_END
        for r in frame.itertuples(index=False)
    )
    path = TABLES / "table06_ibm_construction.tex"
    write(path, r"""\begin{table*}[t]
\caption{Best measured non-reference IBM construction by AUPRC in each cell. Deltas compare that same construction with h64 on AUPRC, AUROC, and F1. The Holm value is the dependence-aware size-level AUPRC test over ten seed blocks, not a context-specific test.}
\label{tab:ibm-construction}
\centering
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\begin{tabularx}{\textwidth}{@{}lrrrrrrX@{}}
\toprule
Cell & Ref AUPRC & Best construction & $\Delta$AUPRC & $\Delta$AUROC & $\Delta$F1 & Holm $p$ & Feasibility \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}""")
    return frame, path, ["results/tkde_rebuild/IBM_CELL_SUMMARY.csv", "results/tkde_rebuild/IBM_MATCHED_ABLATION_EFFECTS.csv"]


def resource_boundaries() -> tuple[pd.DataFrame, Path, list[str]]:
    source = pd.read_csv(FROZEN / "RESOURCE_BOUNDARIES.csv")
    tags = {
        "SAFE_RESOURCE_BLOCKED": "GUARD-BLOCKED",
        "RESOURCE_BLOCKED_T4_CUDA_OOM": "T4-OOM",
        "BLOCKED_T4_OOM": "T4-OOM",
        "BLOCKED_WAITING_FOR_GPU": "UNMEASURED",
    }
    frame = source.assign(
        status_tag=source.status.map(tags).fillna("UNMEASURED"),
        benchmark_treatment="excluded from predictive ranks, matched means, and Pareto sets",
    )[["cell", "resource_envelope_or_reason", "status_tag", "benchmark_treatment"]]
    status_tex = {
        "GUARD-BLOCKED": r"\textsc{guard-\allowbreak blocked}",
        "T4-OOM": r"\textsc{t4-oom}",
        "UNMEASURED": r"\textsc{unmeasured}",
    }
    body = "\n".join(
        f"{esc(r.cell)} & {esc(r.resource_envelope_or_reason)} & {status_tex[r.status_tag]} & {esc(r.benchmark_treatment)}" + ROW_END
        for r in frame.itertuples(index=False)
    )
    path = TABLES / "table07_resource_boundaries.tex"
    write(path, r"""\begin{table}[t]
\caption{Unmeasured resource cells. Status is a feasibility outcome under the declared envelope, never a predictive score.}
\label{tab:resource-boundaries}
\centering
\footnotesize
\setlength{\tabcolsep}{2.2pt}
\begin{tabularx}{\columnwidth}{@{}p{0.23\columnwidth}p{0.18\columnwidth}p{0.24\columnwidth}X@{}}
\toprule
Cell & Envelope & Status & Benchmark treatment \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}""")
    return frame, path, ["results/tkde_rebuild/RESOURCE_BOUNDARIES.csv"]


def graphsafe() -> tuple[pd.DataFrame, Path, list[str]]:
    source = pd.read_csv(FROZEN / "GRAPHSAFE_BOUNDED_SUMMARY.csv")
    frame = source[source.method.isin(["best_val_branch", "simple_average", "graphsafe_conservative"])].copy()
    names = {"best_val_branch": "Best val", "simple_average": "Average", "graphsafe_conservative": "GraphSafe"}
    body = "\n".join(
        f"{esc(dataset_name(r.dataset))} & {names[r.method]} & {f3(r.f1_mean)} & {f3(r.recall_at_1pct_mean)} & {f3(r.cost_sensitive_risk_mean)}" + ROW_END
        for r in frame.itertuples(index=False)
    )
    path = TABLES / "table08_graphsafe_case.tex"
    write(path, r"""\begin{table}[t]
\caption{Bounded saved-score comparison over ten seed blocks. Lower illustrative cost risk is better; no GraphSafe contrast survives Holm correction over the declared 48-test family.}
\label{tab:graphsafe}
\centering
\footnotesize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}llrrr@{}}
\toprule
Data & Method & F1 & R@1\% & Risk \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}""")
    return frame, path, ["results/tkde_rebuild/GRAPHSAFE_BOUNDED_SUMMARY.csv"]


def build() -> list[dict[str, str]]:
    TABLES.mkdir(parents=True, exist_ok=True)
    builders = [
        ("T01", related_work, "Novelty positioning"),
        ("T02", dataset_tasks, "Dataset/task card"),
        ("T03", protocol_visibility, "Protocol/visibility contract"),
        ("T04", protocol_effects, "Paired protocol effects"),
        ("T05", ibm_baselines, "IBM baseline comparison"),
        ("T06", ibm_construction, "IBM matched construction effects"),
        ("T07", resource_boundaries, "Resource boundary"),
        ("T08", graphsafe, "Bounded GraphSafe case"),
    ]
    provenance: list[dict[str, str]] = []
    for table_id, builder, question in builders:
        frame, tex, sources = builder()
        data_path = write_data(tex.stem, frame)
        provenance.append({
            "table_id": table_id,
            "table_file": tex.relative_to(ROOT).as_posix(),
            "table_sha256": sha256(tex),
            "source_data_csv": data_path.relative_to(ROOT).as_posix(),
            "source_data_sha256": sha256(data_path),
            "upstream_sources": ";".join(sources),
            "upstream_sha256": ";".join(sha256(ROOT / source) for source in sources),
            "generation_script": "scripts/tkde_visual_rebuild/build_main_tables.py",
            "scientific_question": question,
            "seed_scope": "1-10 where empirical",
            "intended_use": "main",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "MAIN_TABLE_DATA_PROVENANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    return provenance


def main() -> int:
    provenance = build()
    print(f"generated_main_tables={len(provenance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
