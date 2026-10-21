#!/usr/bin/env python3
"""Build the portrait, human-readable FraudShiftBench supplement tables.

This is a representation-only generator.  It reads the frozen TKDE rebuild
surfaces, verifies their recorded hashes where available, and writes curated
LaTeX fragments.  Exhaustive seed/context rows remain machine-readable and are
indexed rather than typeset.  The script deliberately has no training or GPU
entry point.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
REBUILD = ROOT / "results/tkde_rebuild"
VISUAL = ROOT / "results/tkde_visual_rebuild"
TABLE_DIR = ROOT / "paper_tkde/supplement/tables"
FROZEN_HASHES = VISUAL / "FROZEN_SCIENTIFIC_INPUT_HASHES.csv"

RAW_TABLES = {
    "RB09_SEED_GRID": {
        "path": "results/runs_rb09v3/runs.csv",
        "rows": 180,
        "sha256": "d9f77bfd14ccaf157780858e35ecfee96af1e2fbf60f3559ad888d14c0ace2e9",
        "replacement": "Tables S14--S16: matched protocol effects by metric",
        "purpose": "complete node-protocol seed grid",
    },
    "V22_PAIRED_TESTS": {
        "path": "manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv",
        "rows": 198,
        "sha256": "513f1fb5c2dbae2f63255fd384c62fe86394a15746c31792a92a1f287b8ddadd",
        "replacement": "Table S19: correction-family aggregate",
        "purpose": "complete legacy V22 paired-test family",
    },
    "IBM_SEED_GRID": {
        "path": "results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv",
        "rows": 840,
        "sha256": "dae5d51178b9a746946c9fc05bf31e74840b6663d69031597ef2be45d5e766b6",
        "replacement": "Tables S20 and S21--S25: baseline and matched aggregates",
        "purpose": "complete IBM seed-level imported rows",
    },
    "IBM_CONTEXT_EFFECTS": {
        "path": "results/tkde_rebuild/IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv",
        "rows": 208,
        "sha256": "626f0e014a0b8d339fe344bb736d513ed63a6ec435be6104acad1f53f7bbe202",
        "replacement": "Tables S21--S23: ten-seed effects after fixed-context aggregation",
        "purpose": "complete context-specific IBM sensitivity rows",
    },
}

EXPECTED_CLAIM_STATUSES = Counter(
    {
        "SUPPORTED": 12,
        "DIAGNOSTIC_ONLY": 3,
        "REFUTED_IN_SCOPE": 2,
        "RESOURCE_BLOCKED": 2,
        "SUPPORTED_THEORETICALLY": 2,
        "SUPPORTED_WITH_RESOURCE_BOUNDARY": 1,
    }
)


@dataclass(frozen=True)
class Raw:
    text: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fieldnames(relative: str) -> list[str]:
    path = ROOT / relative
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def frozen_hash_map() -> dict[str, str]:
    if not FROZEN_HASHES.is_file():
        return {}
    rows = read_csv(str(FROZEN_HASHES.relative_to(ROOT)))
    out: dict[str, str] = {}
    for row in rows:
        if row.get("identical") == "True" and row.get("sha256_before"):
            out[row["path"]] = row["sha256_before"]
    return out


def verify_input(relative: str, frozen: Mapping[str, str]) -> None:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = frozen.get(relative)
    if expected and sha256(path) != expected:
        raise RuntimeError(f"Frozen scientific input changed: {relative}")


def latex_escape(value: object) -> str:
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
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_prose(value: str) -> str:
    """Protect comment markers in generator-authored LaTeX prose."""

    return re.sub(r"(?<!\\)%", r"\\%", value)


def cell(value: object) -> str:
    return value.text if isinstance(value, Raw) else latex_escape(value)


def path_cell(value: str) -> Raw:
    return Raw(r"\path{" + value + "}")


def tt(value: str) -> Raw:
    escaped = latex_escape(value)
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    escaped = escaped.replace("/", r"/\allowbreak{}")
    return Raw(r"\texttt{" + escaped + "}")


def hash_cell(value: str) -> Raw:
    chunks = [value[i : i + 8] for i in range(0, len(value), 8)]
    return Raw(r"\texttt{" + r"\allowbreak{}".join(chunks) + "}")


def as_float(value: object) -> float:
    if value in (None, "", "NA", "nan"):
        return math.nan
    return float(value)


def fmt(value: object, digits: int = 5) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "--"
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def fmt_signed(value: object, digits: int = 5) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "--"
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:+.{digits}f}"


def fmt_p(value: object) -> object:
    number = as_float(value)
    if math.isnan(number):
        return "--"
    if 0 < number < 0.0001:
        return Raw(r"$<0.0001$")
    return f"{number:.4f}"


def fmt_int(value: object) -> str:
    if value in (None, ""):
        return "--"
    return f"{int(float(value)):,}"


def fmt_pct(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "--"
    return f"{100 * number:.{digits}f}%"


def ci(low: object, high: object, digits: int = 5) -> str:
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def mean_ci(row: Mapping[str, str], metric: str) -> str:
    return (
        f"{fmt(row[f'{metric}_mean'])} "
        f"[{fmt(row[f'{metric}_ci95_low'])}, {fmt(row[f'{metric}_ci95_high'])}]"
    )


def pretty_dataset(value: str) -> str:
    return {"dgraphfin": "DGraphFin", "elliptic": "Elliptic"}.get(
        value.lower(), value
    )


def pretty_protocol(value: str) -> str:
    return {
        "strict_inductive": "Strict",
        "inductive_isolated": "Isolated",
        "transductive": "Transductive",
        "early_to_late_transfer": "Early-to-late",
        "late_window_holdout": "Late-window",
        "strict-inductive": "Strict",
        "isolated-inductive": "Isolated",
        "late-window holdout": "Late-window",
        "early-to-late transfer": "Early-to-late",
    }.get(value, value.replace("_", " ").title())


def pretty_model(value: str) -> str:
    return {
        "mlp": "MLP",
        "gcn": "GCN",
        "sage": "GraphSAGE",
        "graphsage_edge_minibatch_h32": "SAGE-edge h32",
        "hist_gradient_boosting_edge_features": "HistGB",
        "logistic_regression_edge_features": "LogReg",
        "edge_aware_graphsage_h64": "SAGE-edge h64",
        "edge_aware_graphsage_h64_no_edge_features": "NoEdge",
        "edge_aware_graphsage_h64_shuffled_edge_features": "ShuffledEdge",
        "edge_aware_graphsage_h64_degree_only": "DegreeOnly",
        "degree_capped_bipartite": "DegreeCap",
        "recent_window_only_graph": "RecentWindow",
        "gine_light_h64": "GINE h64",
        "account_account_sender_receiver": "Sender-receiver alias",
        "best_val_branch": "Best validation branch",
        "feature_only": "Feature branch",
        "graph_only": "Graph branch",
        "graphsafe_conservative": "GraphSafe conservative",
        "simple_average": "Simple average",
    }.get(value, value.replace("_", " "))


def write_fragment(name: str, text: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def tabularx(
    name: str,
    caption: str,
    label: str,
    colspec: str,
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
    note: str,
) -> Path:
    caption = latex_prose(caption)
    note = latex_prose(note)
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabularx}}{{\textwidth}}{{@{{}}{colspec}@{{}}}}",
        r"\toprule",
        " & ".join(r"\textbf{" + cell(h) + "}" for h in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(cell(v) for v in row) + r" \\" for row in rows)
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\textwidth}\footnotesize\emph{Scope and reading.} "
        + note
        + r"\end{minipage}",
        r"\end{table}",
    ]
    return write_fragment(name, "\n".join(lines))


def longtable(
    name: str,
    caption: str,
    label: str,
    colspec: str,
    headers: Sequence[object],
    rows: Iterable[Sequence[object]],
    note: str,
) -> Path:
    caption = latex_prose(caption)
    note = latex_prose(note)
    header = " & ".join(r"\textbf{" + cell(h) + "}" for h in headers) + r" \\"
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        rf"\begin{{longtable}}{{@{{}}{colspec}@{{}}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{{caption} (continued)}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{len(headers)}}}{{r}}{{\footnotesize Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    lines.extend(" & ".join(cell(v) for v in row) + r" \\" for row in rows)
    lines += [
        r"\end{longtable}",
        r"\noindent\footnotesize\emph{Scope and reading.} " + note + r"\par",
        r"\endgroup",
    ]
    return write_fragment(name, "\n".join(lines))


def build_navigation() -> Path:
    names = [
        "Scope and navigation",
        "Notation and formal definitions",
        "Support-relation properties and controlled validation",
        "Dataset cards",
        "Temporal windows and visibility matrices",
        "Graph-construction definitions",
        "Model cards and equations",
        "Optimization, selection, and thresholding",
        "Metrics and statistical procedure",
        "Elliptic/DGraphFin protocol results",
        "Robustness and negative controls",
        "IBM AML baseline results",
        "IBM graph construction and scale results",
        "Rank-versus-decision analyses",
        "Review-budget and calibration diagnostics",
        "GraphSafe-TTA case study",
        "Resource-boundary case studies",
        "Claim ledger and provenance map",
        "Reproduction and artifact index",
        "Extended limitations and ethics",
    ]
    purposes = [
        "Reading order, evidence boundary, and destination map.",
        "Typed tasks, contracts, evidence units, and claims.",
        "Five properties and the 14 controlled validator cases.",
        "Prediction units, labels, graph structure, counts, and limitations.",
        "Independent access coordinates, including IBM 50%/60% visibility.",
        "Reference graph and matched structural/feature interventions.",
        "Feature, node-GNN, edge-classifier, and GINE computations.",
        "Optimizer, schedule, validation selection, and decision thresholds.",
        "Metric estimands, seed blocks, intervals, tests, and correction families.",
        "Strict-to-isolated paired effects with uncertainty.",
        "V24 construct audit and V22 aggregate robustness evidence.",
        "Eight exact baseline contexts without cross-task pooling.",
        "Dependence-aware effects, scale feasibility, and runtime.",
        "Winner and rank disagreement under AUPRC, AUROC, and F1@0.5.",
        "Capacity-conditioned review and descriptive calibration.",
        "Bounded saved-prediction policy with positive and negative comparators.",
        "Five resource cases; no blocked cell receives a predictive value.",
        "Five thematic tables preserving all 22 typed claims.",
        "No-training rebuild, schemas, paths, checksums, and raw-row allocation.",
        "Construct, statistical, domain, compute, fairness, and intended-use limits.",
    ]
    return longtable(
        "table_v2_navigation.tex",
        "Reviewer path through the curated supplement.",
        "tab:s-navigation",
        r"r p{0.31\textwidth} p{0.60\textwidth}",
        ["Part", "Section", "Question answered"],
        ((i, name, purpose) for i, (name, purpose) in enumerate(zip(names, purposes), 1)),
        "The sequence moves from definitions to evidence and then to provenance. Exhaustive rows are not part of this reading path; Table~\ref{tab:s-raw-artifacts} locates them in the artifact.",
    )


def build_notation() -> Path:
    rows = [
        (Raw(r"$i$"), "Prediction unit", "Transaction node, account node, or transaction edge, as declared by the dataset card."),
        (Raw(r"$m_i$"), "Eligibility", Raw(r"$\mathbb{1}[y_i\neq0]$ for node tasks; unknown units never enter supervised masks.")),
        (Raw(r"$z_i$"), "Binary target", Raw(r"$\mathbb{1}[y_i=1]$ on eligible node units; IBM uses $z_i\in\{0,1\}$ directly.")),
        (Raw(r"$q_i$"), "Saved score", "Fraud score on the evaluation set under one exact contract."),
        (Raw(r"$\Pi$"), "Deployment contract", Raw(r"$(\mathcal{T},\mathcal{V},\mathcal{C},\mathcal{S},\mathcal{B},\mathcal{R})$.")),
        (Raw(r"$e$"), "Evidence unit", "Dataset, prediction unit, contract, construction, model, seed set, metrics, predictions, resources, and provenance."),
        (Raw(r"$c$"), "Typed claim", "Exact scope, quantifier, comparison, metric, direction, uncertainty rule, and deployment interpretation."),
        (Raw(r"$\mathcal{E}\models c$"), "Support", "Every required unit is admissible and complete, and the scoped predicate is true."),
        (Raw(r"$\Delta_s$"), "Paired effect", "Candidate-minus-reference or isolated-minus-strict difference for matched seed block s."),
        (Raw(r"$d_z$"), "Paired effect size", Raw(r"$\overline{\Delta}/s_{\Delta}$; undefined variance is not converted to evidence.")),
    ]
    return tabularx(
        "table_v2_notation.tex",
        "Notation used by the deployment-contract and statistical analyses.",
        "tab:s-notation",
        r">{\raggedright\arraybackslash}p{0.15\textwidth} >{\raggedright\arraybackslash}p{0.22\textwidth} Y",
        ["Symbol", "Role", "Definition"],
        rows,
        "The prediction unit is part of the evidence type. Consequently, node, account, and transaction-edge values are never pooled into a single leaderboard.",
    )


def build_support_properties() -> Path:
    rows = [
        ("P1", "Scope monotonicity", "Widening a claim adds requirements; it cannot erase requirements already named."),
        ("P2", "Evidence restriction", "Removing the only admissible witness for a required element breaks support."),
        ("P3", "Rank invariance", "Strictly increasing score calibration preserves AUROC/AUPRC ordering but may change a fixed-threshold decision."),
        ("P4", "Protocol non-substitutability", "A result under one claim-relevant visibility or selection coordinate cannot fill another without an invariance argument."),
        ("P5", "Resource separation", "A resource-blocked cell is measurable for feasibility but unordered by predictive performance."),
    ]
    return tabularx(
        "table_v2_support_properties.tex",
        "Properties enforced by the typed support relation.",
        "tab:s-support-properties",
        r"l >{\raggedright\arraybackslash}p{0.25\textwidth} Y",
        ["ID", "Property", "Operational consequence"],
        rows,
        "These are specification and ordering properties, not claims that the ontology is complete or that the validator establishes scientific truth.",
    )


def build_validation_cases(rows: list[dict[str, str]]) -> Path:
    if len(rows) != 14 or any(r["pass"] != "True" for r in rows):
        raise RuntimeError("Expected 14 passing framework validation cases")
    table_rows = []
    for r in rows:
        table_rows.append(
            (
                r["case_id"],
                r["base_claim"],
                r["mutation"],
                tt(r["expected_status"]),
                tt(r["observed_status"]),
            )
        )
    return longtable(
        "table_v2_validation_cases.tex",
        "Controlled support-relation validation cases (14 of 14 matched the expected status).",
        "tab:s-validation-cases",
        r"p{0.07\textwidth} p{0.18\textwidth} p{0.38\textwidth} p{0.15\textwidth} p{0.15\textwidth}",
        ["Case", "Base claim", "Controlled mutation", "Expected", "Observed"],
        table_rows,
        "The cases localize incomplete scope, missing seeds or predictions, integrity and construct failures, resource blocks, and directional refutation. They are project-authored conformance tests, not external validation.",
    )


def build_dataset_cards(data: list[dict[str, str]]) -> list[Path]:
    outputs: list[Path] = []
    for dataset, filename, label, limitation in [
        (
            "Elliptic",
            "table_v2_dataset_elliptic.tex",
            "tab:s-dataset-elliptic",
            "The 49 steps are dataset time bins; conclusions do not imply calendar-time or institution-level generalization.",
        ),
        (
            "DGraphFin",
            "table_v2_dataset_dgraphfin.tex",
            "tab:s-dataset-dgraphfin",
            "Time is a median incident-edge timestamp placed into equal-count buckets, not a calendar deployment schedule.",
        ),
    ]:
        r = next(row for row in data if row["dataset"] == dataset)
        card = [
            ("Origin", r["origin"]),
            ("Prediction unit", r["prediction_unit"]),
            ("Graph size", f"{fmt_int(r['nodes_or_accounts'])} nodes; {fmt_int(r['source_edges_or_transactions'])} source edges; {fmt_int(r['message_passing_arcs'])} message-passing arcs"),
            ("Features", f"{fmt_int(r['node_feature_dim'])} node; {fmt_int(r['edge_feature_dim'])} edge"),
            ("Temporal order", r["time_definition"]),
            ("Protocols/windows", r["protocol"]),
            ("Train/validation/test", f"{fmt_int(r['train_units'])} / {fmt_int(r['validation_units'])} / {fmt_int(r['test_units'])} eligible units"),
            ("Positive counts", f"{fmt_int(r['train_positives'])} / {fmt_int(r['validation_positives'])} / {fmt_int(r['test_positives'])}"),
            ("Test prevalence", fmt_pct(r["test_positive_rate"])),
            ("Label mapping", r["label_mapping"]),
            ("Source surface", path_cell(r["source"])),
            ("Known limitation", limitation),
        ]
        outputs.append(
            tabularx(
                filename,
                f"{dataset} dataset card.",
                label,
                r">{\raggedright\arraybackslash}p{0.23\textwidth} Y",
                ["Field", "Verified value"],
                card,
                "Counts and label mappings are recovered from the frozen dataset/task statistics. Unknown node labels remain graph context but never enter supervised masks.",
            )
        )

    ibm = [r for r in data if r["dataset"] == "IBM AML-Data"]
    variants = sorted({r["variant"] for r in ibm})
    card = [
        ("Origin", "Synthetic financial transactions from IBM AML-Data"),
        ("Prediction unit", "Directed transaction edge"),
        ("Measured variants", ", ".join(v.replace("-", " ").upper() for v in variants)),
        ("Graph range", f"{fmt_int(min(int(r['nodes_or_accounts']) for r in ibm))}--{fmt_int(max(int(r['nodes_or_accounts']) for r in ibm))} accounts; {fmt_int(min(int(r['source_edges_or_transactions']) for r in ibm))}--{fmt_int(max(int(r['source_edges_or_transactions']) for r in ibm))} transactions"),
        ("Features", "Eight transaction attributes and eight account-history attributes"),
        ("Temporal order", "Stable chronological transaction sort"),
        ("Labels", "is_laundering=1 positive; 0 negative; no unknown class in the materialized task"),
        ("Visibility", "Classifier labels use 50% or 60% by protocol; both protocols share the first-60% label-free account-history map"),
        ("Unmeasured scope", "HI/LI Large are guard-blocked; HI/LI Medium GINE h64 is T4-OOM"),
        ("Known limitation", "HI/LI are synthetic generator regimes, not observed bank populations or severity categories"),
    ]
    outputs.append(
        tabularx(
            "table_v2_dataset_ibm.tex",
            "IBM AML-Data dataset card.",
            "tab:s-dataset-ibm",
            r">{\raggedright\arraybackslash}p{0.23\textwidth} Y",
            ["Field", "Verified value"],
            card,
            "Predictive evidence exists only for the four Small/Medium variants. Large and Medium-GINE cells are retained as resource outcomes, not assigned performance.",
        )
    )
    return outputs


def build_protocol_tables(
    protocols: list[dict[str, str]], data: list[dict[str, str]]
) -> list[Path]:
    p_rows = [
        (
            pretty_protocol(r["protocol"]),
            r["temporal_masks"],
            r["graph_visibility"],
            r["label_availability"],
            r["selection_or_threshold"],
            r["instantiated_on"],
        )
        for r in protocols
    ]
    out = [
        tabularx(
            "table_v2_protocol_cards.tex",
            "Protocol cards keep time, visibility, selection, and decision coordinates separate.",
            "tab:s-protocol-cards",
            r">{\raggedright\arraybackslash}p{0.12\textwidth} >{\raggedright\arraybackslash}p{0.13\textwidth} Y Y Y >{\raggedright\arraybackslash}p{0.12\textwidth}",
            ["Protocol", "Masks", "Graph/covariate visibility", "Labels", "Selection/decision", "Datasets"],
            p_rows,
            "Strict and isolated protocols share masks but differ in graph access. IBM early-to-late is explicitly a 50% labeled-training / 60% label-free-history contract.",
        )
    ]

    visibility = [
        ("Strict", "Chronological train", "Validation F1", "Yes", "Held-out structure at evaluation", "Training-period only", "Validation-selected", "Recorded run envelope"),
        ("Isolated", "Same train", "Validation F1", "Yes", "Held-out subgraph; no train/held-out cross edges", "Training-period only", "Validation-selected", "Recorded run envelope"),
        ("Transductive", "Same train", "Validation F1", "Yes", "Full graph during training/evaluation; held-out labels hidden", "Full visible graph", "Validation-selected", "Recorded run envelope"),
        ("IBM late-window", "First 60% labels", "Next 20%", "Yes", "Account graph and transaction features", "First 60% label-free history", "Fixed 0.5 F1 threshold", "Small/Medium measured"),
        ("IBM early-to-late", "First 50% labels", "Next 20%", "Yes", "Account graph and transaction features", "Shared first 60%; includes 50--60% label-free covariates", "Fixed 0.5 F1 threshold", "Small/Medium measured"),
    ]
    out.append(
        longtable(
            "table_v2_visibility_matrix.tex",
            "Access matrix for the five evaluated deployment contracts.",
            "tab:s-visibility-matrix",
            r"p{0.11\textwidth} p{0.10\textwidth} p{0.09\textwidth} p{0.06\textwidth} p{0.18\textwidth} p{0.15\textwidth} p{0.11\textwidth} p{0.10\textwidth}",
            ["Contract", "Labeled train", "Validation", "Test features", "Graph nodes/edges", "Historical covariates", "Decision rule", "Resources"],
            visibility,
            "Test labels are hidden in every evaluation. Access to test features or graph structure is a declared contract coordinate and must not be confused with access to test labels.",
        )
    )

    ibm_rows = sorted(
        (r for r in data if r["dataset"] == "IBM AML-Data"),
        key=lambda r: (r["variant"], r["protocol"]),
    )
    out.append(
        longtable(
            "table_v2_ibm_windows.tex",
            "Exact IBM temporal-window counts and test prevalence.",
            "tab:s-ibm-windows",
            r"p{0.12\textwidth} p{0.14\textwidth} r r r r r",
            ["Variant", "Protocol", "Train", "Validation", "Test", "Test positives", "Test prevalence"],
            (
                (
                    r["variant"].replace("-", " ").upper(),
                    pretty_protocol(r["protocol"].split(":")[0]),
                    fmt_int(r["train_units"]),
                    fmt_int(r["validation_units"]),
                    fmt_int(r["test_units"]),
                    fmt_int(r["test_positives"]),
                    fmt_pct(r["test_positive_rate"]),
                )
                for r in ibm_rows
            ),
            "The table reports prediction-unit denominators. It does not pool HI/LI or Small/Medium, and the early-to-late history visibility remains the shared first-60% map described above.",
        )
    )
    return out


def build_model_and_construction_tables(
    inventory: list[dict[str, str]], training: list[dict[str, str]]
) -> list[Path]:
    rows_by_name = {r["method_or_construction"]: r for r in inventory}
    outputs: list[Path] = []
    constructions = [
        rows_by_name[name]
        for name in ["NoEdge", "ShuffledEdge", "DegreeOnly", "DegreeCap", "RecentWindow"]
    ]
    construction_rows = [
        (r["method_or_construction"], r["computational_form"], r["hypothesis"], r["configuration"])
        for r in constructions
    ]
    construction_rows.append(
        (
            "Sender-receiver alias",
            "Restates the already materialized sender-to-receiver transaction edges",
            "Contract identity check; no new transformation",
            "Diagnostic only; never counted as an independent method",
        )
    )
    outputs.append(
        tabularx(
            "table_v2_construction_cards.tex",
            "Matched IBM graph-construction and feature interventions.",
            "tab:s-construction-cards",
            r">{\raggedright\arraybackslash}p{0.16\textwidth} Y Y >{\raggedright\arraybackslash}p{0.20\textwidth}",
            ["Construction", "Operation", "Question", "Matched scope"],
            construction_rows,
            "Every intervention retains variant, protocol, seed, optimizer schedule, and h64 reference family unless the operation itself changes training structure. The sender-receiver row is an alias, not replication.",
        )
    )

    node = [rows_by_name[name] for name in ["MLP", "GCN", "GraphSAGE"]]
    outputs.append(
        tabularx(
            "table_v2_node_model_cards.tex",
            "Node-classifier cards for Elliptic and DGraphFin.",
            "tab:s-node-models",
            r">{\raggedright\arraybackslash}p{0.16\textwidth} Y Y >{\raggedright\arraybackslash}p{0.22\textwidth}",
            ["Model", "Computation", "Scientific role", "Configuration"],
            ((r["method_or_construction"], r["computational_form"], r["hypothesis"], r["configuration"]) for r in node),
            "The MLP is the graph-visibility negative control. The grid does not establish class-wide behavior for modern temporal or heterogeneous GNNs.",
        )
    )

    ibm_names = [
        "Logistic regression",
        "Histogram gradient boosting",
        "GraphSAGE-derived edge classifier h32",
        "Edge-aware GraphSAGE-derived edge classifier h64",
        "GINE h64",
    ]
    ibm_models = [rows_by_name[name] for name in ibm_names]
    outputs.append(
        tabularx(
            "table_v2_ibm_model_cards.tex",
            "IBM transaction-classifier cards.",
            "tab:s-ibm-models",
            r">{\raggedright\arraybackslash}p{0.20\textwidth} Y Y >{\raggedright\arraybackslash}p{0.20\textwidth}",
            ["Model", "Input/computation", "Scientific role", "Configuration"],
            ((r["method_or_construction"], r["computational_form"], r["hypothesis"], r["configuration"]) for r in ibm_models),
            "The h32/h64 systems are static one-hop GraphSAGE-derived edge classifiers. Medium GINE is resource-blocked and has no performance value.",
        )
    )

    outputs.append(
        longtable(
            "table_v2_training_cards.tex",
            "Optimization, validation selection, and decision settings.",
            "tab:s-training-cards",
            r"p{0.14\textwidth} p{0.10\textwidth} p{0.07\textwidth} p{0.07\textwidth} p{0.12\textwidth} p{0.09\textwidth} p{0.13\textwidth} p{0.13\textwidth}",
            ["Family", "Optimizer", "LR", "Weight decay", "Loss", "Iterations", "Selection", "Decision/batch"],
            (
                (
                    r["family"],
                    r["optimizer"],
                    r["learning_rate"],
                    r["weight_decay"],
                    r["loss"],
                    r["max_epochs_or_iterations"],
                    r["selection"],
                    f"{r['threshold_rule']}; {r['batch_or_forward']}",
                )
                for r in training
            ),
            "Node-grid checkpoints are selected by validation F1. IBM saved F1 uses 0.5. Deterministic library baselines may repeat exactly across nominal seeds; those repetitions are not reinterpreted as independent stochastic fits.",
        )
    )
    return outputs


def build_metrics_and_statistics() -> list[Path]:
    metrics = [
        ("AUPRC", "Rank", "Area under precision-recall curve", "Primary rare-event ranking metric; interpreted against within-cell prevalence."),
        ("AUROC", "Rank", "Probability a random positive outranks a random negative", "Can remain high when precision is poor under severe imbalance."),
        ("F1@0.5", "Decision", "Harmonic mean of precision and recall", "IBM operating point only; not threshold-invariant."),
        ("Precision/Recall@b", "Capacity", "Top K scores where K is the declared review fraction", "Reported at 0.5%, 1%, and 2%; queue-capacity conditioned."),
        ("Normalized AUPRC", "Diagnostic", "AUPRC divided by test prevalence", "Lift context only; not a prevalence-invariant universal metric."),
        ("Brier score", "Calibration", "Mean squared probability error", "Descriptive on saved predictions; lower is better."),
        ("ECE", "Calibration", "Ten-bin expected calibration error", "Binning dependent and descriptive."),
        ("Cost risk", "Decision", "FP + 5 FN divided by evaluation size", "Illustrative 1:5 scenario, not bank monetary loss."),
    ]
    out = [
        tabularx(
            "table_v2_metric_cards.tex",
            "Metric cards distinguish ranking, decision, capacity, and calibration questions.",
            "tab:s-metric-cards",
            r">{\raggedright\arraybackslash}p{0.16\textwidth} >{\raggedright\arraybackslash}p{0.12\textwidth} Y Y",
            ["Metric", "Type", "Estimand", "Interpretation boundary"],
            metrics,
            "No metric substitutes for another. In particular, monotone calibration cannot repair an AUPRC ranking, and fixed-threshold F1 cannot stand in for every review policy.",
        )
    ]
    stats = [
        ("Node visibility", "18 dataset-model-metric effects", "10 paired seeds", "10,000-resample percentile CI; paired dz; Wilcoxon", "Holm within metric"),
        ("IBM construction", "Candidate vs h64 by size and metric", "10 seed blocks after averaging four fixed contexts within seed", "10,000-resample percentile CI; paired dz; Wilcoxon", "Holm within size-by-metric"),
        ("IBM context sensitivity", "208 context-specific effects", "10 paired seeds per fixed context", "Descriptive CI, dz, and Wilcoxon", "Context table is sensitivity, not a second confirmatory family"),
        ("GraphSafe", "48 declared comparator-metric tests", "10 seed blocks after averaging six contexts within seed", "Percentile CI and Wilcoxon", "One Holm family of 48"),
        ("V22 robustness", "198 legacy paired tests", "10 paired seeds", "Bootstrap CI and paired test", "Original Benjamini-Hochberg values retained"),
    ]
    out.append(
        tabularx(
            "table_v2_statistical_families.tex",
            "Statistical families and inferential units.",
            "tab:s-statistical-families",
            r">{\raggedright\arraybackslash}p{0.16\textwidth} >{\raggedright\arraybackslash}p{0.18\textwidth} Y Y >{\raggedright\arraybackslash}p{0.18\textwidth}",
            ["Family", "Scope", "Independent unit", "Interval/test", "Correction"],
            stats,
            "Ten seeds quantify optimization variation on fixed datasets and splits. They do not estimate population variation across banks, institutions, or future periods.",
        )
    )
    return out


def build_protocol_effect_tables(effects: list[dict[str, str]]) -> list[Path]:
    if len(effects) != 18:
        raise RuntimeError(f"Expected 18 RB09 protocol effects, found {len(effects)}")
    out: list[Path] = []
    for metric in ["auprc", "auroc", "f1"]:
        rows = [r for r in effects if r["metric"] == metric]
        rows.sort(key=lambda r: (r["dataset"], r["model"]))
        out.append(
            tabularx(
                f"table_v2_protocol_effect_{metric}.tex",
                f"Strict-to-isolated paired {metric.upper()} effects.",
                f"tab:s-protocol-{metric}",
                r">{\raggedright\arraybackslash}p{0.13\textwidth} >{\raggedright\arraybackslash}p{0.13\textwidth} r r r >{\raggedright\arraybackslash}p{0.21\textwidth} r r",
                ["Dataset", "Model", "n", "Strict", "Isolated", Raw(r"$\Delta$ [95\% CI]"), Raw(r"$d_z$"), "Holm p"],
                (
                    (
                        pretty_dataset(r["dataset"]),
                        pretty_model(r["model"]),
                        r["n_pairs"],
                        fmt(r["strict_mean"]),
                        fmt(r["isolated_mean"]),
                        f"{fmt_signed(r['delta_isolated_minus_strict'])} {ci(r['delta_ci95_low'], r['delta_ci95_high'])}",
                        fmt(r["cohen_dz"], 3),
                        fmt_p(r["holm_p_within_metric"]),
                    )
                    for r in rows
                ),
                "All comparisons share seeds 1--10 and identical temporal masks. The intervention changes graph visibility; it does not identify a unique mechanism for the observed response.",
            )
        )
    return out


def build_robustness_tables(
    v24: list[dict[str, str]], lanes: list[dict[str, str]], tests: list[dict[str, str]]
) -> list[Path]:
    if len(v24) != 120 or not all(r["all_performance_metrics_identical"] == "True" for r in v24):
        raise RuntimeError("V24 duplicate audit no longer matches the frozen 120-cell finding")
    if len(tests) != 198:
        raise RuntimeError(f"Expected 198 V22 tests, found {len(tests)}")
    out = [
        tabularx(
            "table_v2_v24_construct_audit.tex",
            "Construct audit of the V24 temporal-stress labels.",
            "tab:s-v24-audit",
            r">{\raggedright\arraybackslash}p{0.18\textwidth} r r Y >{\raggedright\arraybackslash}p{0.24\textwidth}",
            ["Scientific base cells", "Metadata labels/cell", "Duplicate rows", "Audit finding", "Admissible use"],
            [
                (
                    "120 dataset/protocol/model/seed cells",
                    "3",
                    "240",
                    "The label was not passed to the benchmark harness; every performance metric repeats across labels.",
                    "One deduplicated representative per base cell as supplementary rerun evidence only.",
                )
            ],
            "Runtime can differ because the same harness was executed again. No temporal-stress contrast is admissible from this family.",
        )
    ]
    lane_names = {
        "RB28_DGRAPHFIN_LOSS_ROBUSTNESS_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin loss robustness",
        "RB28_ELLIPTIC_LOSS_ROBUSTNESS_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic loss robustness",
        "RB29_DGRAPHFIN_NEGATIVE_CONTROLS_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin negative controls",
        "RB29_ELLIPTIC_NEGATIVE_CONTROLS_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic negative controls",
        "RB30_DGRAPHFIN_EXTRA_ARCH_ISOLATED_FULL10_SEEDS_1_10_DUALGPU_V22": "DGraphFin fixed GAT h64/l2",
        "RB30_ELLIPTIC_EXTRA_ARCH_FULL10_SEEDS_1_10_DUALGPU_V22": "Elliptic GAT/GIN",
    }
    out.append(
        tabularx(
            "table_v2_v22_lanes.tex",
            "Canonical V22 robustness lanes and feasibility status.",
            "tab:s-v22-lanes-v2",
            r"Y >{\raggedright\arraybackslash}p{0.21\textwidth} r r r",
            ["Scientific lane", "Status", "Result rows", "Prediction files", "Prediction rows"],
            (
                (
                    lane_names.get(r["lane_id"], r["lane_id"]),
                    tt(r["status"]),
                    fmt_int(r["actual_json"]),
                    fmt_int(r["actual_prediction_csv"]),
                    fmt_int(r["prediction_rows"]),
                )
                for r in lanes
            ),
            "Five lanes are complete. DGraphFin GAT h64/l2 is T4-OOM; the memory-reduced diagnostic is a different configuration and cannot fill this cell.",
        )
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in tests:
        grouped[(r["family"], r["metric"])].append(r)
    summary = []
    for (family, metric), group in sorted(grouped.items()):
        positives = sum(as_float(r["mean_diff_left_minus_right"]) > 0 for r in group)
        negatives = sum(as_float(r["mean_diff_left_minus_right"]) < 0 for r in group)
        significant = sum(as_float(r["p_value_bh"]) < 0.05 for r in group)
        summary.append((family, metric.upper(), len(group), positives, negatives, significant))
    out.append(
        tabularx(
            "table_v2_v22_statistical_summary.tex",
            "Aggregate view of the 198 V22 paired tests.",
            "tab:s-v22-stats-summary",
            r"l l r r r r Y",
            ["Family", "Metric", "Tests", Raw(r"$\Delta>0$"), Raw(r"$\Delta<0$"), "BH p<0.05", "Machine-readable source"],
            ((a, b, c, d, e, f, path_cell("manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv")) for a, b, c, d, e, f in summary),
            "Signs are left-minus-right as declared in the source. This aggregate is a navigation aid; it does not create a new hypothesis family or mix the older BH values with the primary Holm families.",
        )
    )
    return out


def build_ibm_baseline(rows: list[dict[str, str]]) -> Path:
    configs = [
        "hist_gradient_boosting_edge_features",
        "logistic_regression_edge_features",
        "graphsage_edge_minibatch_h32",
    ]
    selected = [r for r in rows if r["version"] == "V26" and r["config"] in configs]
    if len(selected) != 24:
        raise RuntimeError(f"Expected 24 IBM baseline aggregate cells, found {len(selected)}")
    lookup = {(r["variant"], r["protocol"], r["config"]): r for r in selected}
    keys = sorted({(r["variant"], r["protocol"]) for r in selected})
    table_rows = []
    for variant, protocol in keys:
        table_rows.append(
            (
                variant.replace("-", " ").upper(),
                pretty_protocol(protocol),
                mean_ci(lookup[(variant, protocol, configs[0])], "auprc"),
                mean_ci(lookup[(variant, protocol, configs[1])], "auprc"),
                mean_ci(lookup[(variant, protocol, configs[2])], "auprc"),
            )
        )
    return tabularx(
        "table_v2_ibm_baselines.tex",
        "IBM baseline-grid AUPRC means and 95% seed intervals.",
        "tab:s-ibm-baselines",
        r">{\raggedright\arraybackslash}p{0.12\textwidth} >{\raggedright\arraybackslash}p{0.13\textwidth} Y Y Y",
        ["Variant", "Protocol", "HistGB", "LogReg", "SAGE-edge h32"],
        table_rows,
        "Each entry is mean [95% bootstrap interval] over ten saved rows. HistGB leads mean AUPRC in these eight exact contexts; the table does not support universal superiority over AML models.",
    )


def build_ibm_effects(rows: list[dict[str, str]]) -> list[Path]:
    if len(rows) != 52:
        raise RuntimeError(f"Expected 52 matched IBM effect rows, found {len(rows)}")
    out = []
    for metric in ["auprc", "auroc", "f1"]:
        selected = sorted(
            (r for r in rows if r["metric"] == metric),
            key=lambda r: (r["size"], pretty_model(r["config"])),
        )
        out.append(
            longtable(
                f"table_v2_ibm_effect_{metric}.tex",
                f"Dependence-aware matched IBM {metric.upper()} effects relative to SAGE-edge h64.",
                f"tab:s-ibm-effect-{metric}",
                r"p{0.08\textwidth} p{0.18\textwidth} r r p{0.24\textwidth} r r p{0.10\textwidth}",
                ["Size", "Candidate", "Reference", "Candidate", Raw(r"$\Delta$ [95\% CI]"), Raw(r"$d_z$"), "Holm p", "Context signs"],
                (
                    (
                        r["size"].title(),
                        pretty_model(r["config"]),
                        fmt(r["reference_mean"]),
                        fmt(r["candidate_mean"]),
                        f"{fmt_signed(r['mean_delta'])} {ci(r['delta_ci95_low'], r['delta_ci95_high'])}",
                        fmt(r["cohen_dz"], 3),
                        fmt_p(r["holm_p_within_size_metric"]),
                        f"+{r['n_contexts_positive']} / -{r['n_contexts_negative']} / 0:{r['n_contexts_zero']}",
                    )
                    for r in selected
                ),
                "Inference uses ten seed blocks after averaging four fixed HI/LI-by-protocol contexts within seed. The context-sign column summarizes heterogeneity; the complete 208 context rows remain in the artifact. The sender-receiver row is an identity diagnostic.",
            )
        )
    return out


def build_ibm_runtime(rows: list[dict[str, str]]) -> list[Path]:
    if len(rows) != 54:
        raise RuntimeError(f"Expected 54 IBM runtime/feasibility rows, found {len(rows)}")
    out = []
    for size in ["small", "medium"]:
        selected = sorted(
            (r for r in rows if r["size"] == size),
            key=lambda r: (r["variant"], r["protocol"], pretty_model(r["config"])),
        )
        out.append(
            longtable(
                f"table_v2_ibm_runtime_{size}.tex",
                f"IBM {size.title()} configuration-level AUPRC, runtime, and feasibility.",
                f"tab:s-ibm-runtime-{size}",
                r"p{0.11\textwidth} p{0.13\textwidth} p{0.19\textwidth} p{0.16\textwidth} r p{0.16\textwidth}",
                ["Variant", "Protocol", "Configuration", "Status", "AUPRC / seconds", "Within-cell treatment"],
                (
                    (
                        r["variant"].replace("-", " ").upper(),
                        pretty_protocol(r["protocol"]),
                        pretty_model(r["config"]),
                        tt(r["status"]),
                        "--" if not r["auprc_mean"] else f"{fmt(r['auprc_mean'])} / {fmt(r['runtime_seconds_mean'], 2)}",
                        "Pareto" if r["pareto_auprc_runtime_within_cell"] == "True" else ("Unmeasured" if not r["auprc_mean"] else "Measured, non-Pareto"),
                    )
                    for r in selected
                ),
                "Pareto treatment is computed only within the same variant and protocol. Runtime is runner elapsed time, not a hardware-normalized complexity or energy measure. Blocked rows remain nonnumeric.",
            )
        )
    return out


def build_rank_tables(rows: list[dict[str, str]]) -> list[Path]:
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 IBM rank-divergence contexts, found {len(rows)}")
    out = []
    for family, title in [("baseline_grid", "baseline grid"), ("graph_grid", "graph grid")]:
        selected = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: (r["variant"], r["protocol"]),
        )
        out.append(
            tabularx(
                f"table_v2_rank_{family}.tex",
                f"IBM {title}: AUPRC and F1@0.5 winners and rank agreement.",
                f"tab:s-rank-{family}",
                r">{\raggedright\arraybackslash}p{0.11\textwidth} >{\raggedright\arraybackslash}p{0.13\textwidth} r Y Y r l",
                ["Variant", "Protocol", "Feasible n", "AUPRC winner", "F1 winner", Raw(r"$\rho$"), "Disagree"],
                (
                    (
                        r["variant"].replace("-", " ").upper(),
                        pretty_protocol(r["protocol"]),
                        r["n_configurations"],
                        pretty_model(r["auprc_winner"]),
                        pretty_model(r["f1_winner"]),
                        fmt(r["spearman_auprc_vs_f1"], 3),
                        "Yes" if r["auprc_f1_winner_disagree"] == "True" else "No",
                    )
                    for r in selected
                ),
                "Spearman correlation is computed over the exact feasible set in each row. F1 uses the saved threshold of 0.5; disagreement is not asserted to be threshold-invariant.",
            )
        )
    return out


def build_review_and_graphsafe(
    budgets: list[dict[str, str]],
    calibration: list[dict[str, str]],
    summary: list[dict[str, str]],
    paired: list[dict[str, str]],
) -> list[Path]:
    out: list[Path] = []
    one_pct = sorted(
        (r for r in budgets if as_float(r["budget_pct"]) == 1.0),
        key=lambda r: (r["dataset"], r["method"]),
    )
    if len(one_pct) != 10:
        raise RuntimeError("Expected ten GraphSafe policy rows at the 1% review budget")
    out.append(
        tabularx(
            "table_v2_review_budget_1pct.tex",
            "Precision and recall at a 1% review budget.",
            "tab:s-review-budget-1pct",
            r">{\raggedright\arraybackslash}p{0.14\textwidth} Y r r r r",
            ["Dataset", "Policy", "n", "Precision", "SD", "Recall (SD)"],
            (
                (
                    pretty_dataset(r["dataset"]),
                    pretty_model(r["method"]),
                    r["n"],
                    fmt(r["precision_mean"], 4),
                    fmt(r["precision_std"], 4),
                    f"{fmt(r['recall_mean'], 4)} ({fmt(r['recall_std'], 4)})",
                )
                for r in one_pct
            ),
            "The six model/protocol contexts are averaged within seed before the ten-seed summary. Review-budget results are capacity-conditioned and do not replace AUPRC.",
        )
    )
    if len(calibration) != 10:
        raise RuntimeError("Expected ten GraphSafe calibration summary rows")
    out.append(
        tabularx(
            "table_v2_calibration.tex",
            "Saved-prediction calibration diagnostics.",
            "tab:s-calibration",
            r">{\raggedright\arraybackslash}p{0.14\textwidth} Y r r r r",
            ["Dataset", "Policy", "n", "ECE", "SD", "Brier (SD)"],
            (
                (
                    pretty_dataset(r["dataset"]),
                    pretty_model(r["method"]),
                    r["n"],
                    fmt(r["ece_mean"], 5),
                    fmt(r["ece_std"], 5),
                    f"{fmt(r['brier_mean'], 5)} ({fmt(r['brier_std'], 5)})",
                )
                for r in sorted(calibration, key=lambda r: (r["dataset"], r["method"]))
            ),
            "ECE uses ten bins and is descriptive. These fixed-dataset summaries do not establish calibrated probabilities in a deployment population.",
        )
    )
    if len(summary) != 10:
        raise RuntimeError("Expected ten GraphSafe bounded-summary rows")
    out.append(
        longtable(
            "table_v2_graphsafe_summary.tex",
            "GraphSafe and comparator outcomes after seed-block aggregation.",
            "tab:s-graphsafe-summary",
            r"p{0.12\textwidth} p{0.19\textwidth} r r r r r",
            ["Dataset", "Policy", "n", "F1", "AUPRC", "Recall@1%", "Cost risk"],
            (
                (
                    pretty_dataset(r["dataset"]),
                    pretty_model(r["method"]),
                    r["n"],
                    fmt(r["f1_mean"]),
                    fmt(r["auprc_mean"]),
                    fmt(r["recall_at_1pct_mean"]),
                    fmt(r["cost_sensitive_risk_mean"]),
                )
                for r in sorted(summary, key=lambda r: (r["dataset"], r["method"]))
            ),
            "Each row averages six fixed contexts within seed, then summarizes ten seeds. The 1:5 cost is illustrative. DGraphFin descriptive gains and Elliptic negative comparisons are both retained.",
        )
    )
    key = sorted(
        (r for r in paired if r["method"] == "graphsafe_conservative"),
        key=lambda r: (r["dataset"], r["metric"]),
    )
    if len(key) != 12:
        raise RuntimeError("Expected 12 GraphSafe-conservative comparisons to simple averaging")
    out.append(
        longtable(
            "table_v2_graphsafe_paired.tex",
            "GraphSafe conservative minus simple-average paired effects.",
            "tab:s-graphsafe-paired",
            r"p{0.13\textwidth} p{0.20\textwidth} p{0.09\textwidth} r p{0.25\textwidth} r",
            ["Dataset", "Metric", "Direction", Raw(r"$\Delta$"), "95% CI", "Holm p"],
            (
                (
                    pretty_dataset(r["dataset"]),
                    r["metric"].replace("_", " "),
                    r["direction"],
                    fmt_signed(r["mean_improvement"]),
                    ci(r["ci95_low"], r["ci95_high"]),
                    fmt_p(r["holm_p_all_graphsafe_vs_average"]),
                )
                for r in key
            ),
            "The correction family contains all 48 saved-prediction comparator tests, not only the 12 rows shown here. No adjusted p-value is below 0.05; therefore this case does not establish universal dominance or ranking repair.",
        )
    )
    return out


def build_resource_and_integrity(
    resources: list[dict[str, str]], false_promotions: list[dict[str, str]]
) -> list[Path]:
    if len(resources) != 6:
        raise RuntimeError("Expected six frozen resource-boundary rows")
    out = [
        longtable(
            "table_v2_resource_cases.tex",
            "Measured resource outcomes and benchmark treatment.",
            "tab:s-resource-cases",
            r"p{0.21\textwidth} p{0.21\textwidth} p{0.11\textwidth} p{0.15\textwidth} p{0.18\textwidth}",
            ["Evaluation cell", "Declared envelope / observed event", "Outputs", "Status", "Benchmark treatment"],
            (
                (
                    r["cell"],
                    r["resource_envelope_or_reason"],
                    f"{r['result_outputs']} results; {r['prediction_exports']} predictions",
                    tt(r["status"]),
                    r["interpretation"],
                )
                for r in resources
            ),
            "A blocked cell is unmeasured and excluded from predictive ranks, matched means, and Pareto sets. A reduced diagnostic cannot replace a fixed blocked configuration.",
        )
    ]
    if len(false_promotions) != 7:
        raise RuntimeError("Expected seven false-promotion audit rows")
    out.append(
        longtable(
            "table_v2_false_promotions.tex",
            "False-promotion patterns guarded by the artifact layer.",
            "tab:s-false-promotions-v2",
            r"p{0.08\textwidth} p{0.22\textwidth} r r p{0.20\textwidth} p{0.19\textwidth}",
            ["Audit", "Family", "Result risk", "Prediction risk", "Correct treatment", "Why count-only logic fails"],
            (
                (
                    r["audit_id"],
                    r["artifact_family"],
                    r["result_files_at_risk"],
                    r["prediction_files_at_risk"],
                    tt(r["correct_status"]),
                    r["reason"],
                )
                for r in false_promotions
            ),
            "The 310 result and 310 prediction files at risk are not additional evidence. Zero-output plans remain visible because directories and notebooks can otherwise be mistaken for completed runs.",
        )
    )
    return out


CLAIM_GROUPS = {
    "protocol": ["C01", "C02", "C03", "C04", "C19"],
    "ibm": ["C05", "C07", "C08", "C09", "C10", "C11", "C12", "C13"],
    "decision": ["C06", "C14", "C18"],
    "graphsafe": ["C15", "C16", "C17"],
    "integrity": ["C20", "C21", "C22"],
}


def build_claim_tables(claims: list[dict[str, str]]) -> list[Path]:
    if len(claims) != 22:
        raise RuntimeError(f"Expected 22 claims, found {len(claims)}")
    ids = {r["claim_id"] for r in claims}
    if ids != {f"C{i:02d}" for i in range(1, 23)}:
        raise RuntimeError("Claim ID set changed")
    statuses = Counter(r["support_status"] for r in claims)
    if statuses != EXPECTED_CLAIM_STATUSES:
        raise RuntimeError(f"Claim statuses changed: {statuses}")
    grouped_ids = [claim for group in CLAIM_GROUPS.values() for claim in group]
    if len(grouped_ids) != 22 or set(grouped_ids) != ids:
        raise RuntimeError("Thematic claim allocation must cover each claim exactly once")
    lookup = {r["claim_id"]: r for r in claims}
    titles = {
        "protocol": "Protocol and architecture claims",
        "ibm": "IBM AML baseline, construction, and scale claims",
        "decision": "Rank, decision, prevalence, and calibration claims",
        "graphsafe": "GraphSafe-TTA claims",
        "integrity": "Evidence-integrity and resource claims",
    }
    outputs = []
    for group, group_ids in CLAIM_GROUPS.items():
        selected = [lookup[cid] for cid in group_ids]
        outputs.append(
            tabularx(
                f"table_v2_claims_{group}.tex",
                titles[group] + ".",
                f"tab:s-claims-{group}",
                r"l >{\raggedright\arraybackslash}p{0.16\textwidth} Y Y Y",
                ["ID", "Status", "Exact scoped subject", "Permitted wording", "Prohibited generalization"],
                (
                    (
                        r["claim_id"],
                        tt(r["support_status"]),
                        r["scope"],
                        r["permitted_wording"],
                        r["prohibited_wording"],
                    )
                    for r in selected
                ),
                "Status and scope text are copied from the frozen 22-claim ledger. Exact matched evidence-unit identifiers remain in the machine-readable ledger rather than in these reviewer-facing cells.",
            )
        )
    return outputs


def build_provenance_map(families: list[dict[str, str]]) -> Path:
    keep = {
        ("core strict/isolated/transductive benchmark", "main-paper eligible"),
        ("loss/control/architecture extension", "supplement-only"),
        ("temporal-shift/perturbation stress", "supplement-only"),
        ("IBM AML baseline grid", "main-paper eligible"),
        ("IBM AML stronger graph study", "main-paper eligible"),
        ("IBM AML construction/feature ablation", "main-paper eligible"),
        ("validation-selected conservative GraphSafe policy", "main case-study eligible"),
        ("review-budget/worst-block/cost sensitivity", "supplement-only"),
    }
    selected = [r for r in families if (r["family"], r["eligibility"]) in keep]
    if len(selected) != 8:
        raise RuntimeError(f"Expected eight core provenance families, found {len(selected)}")
    return longtable(
        "table_v2_provenance_map.tex",
        "Reviewer-facing experiment-family to evidence-lock map.",
        "tab:s-provenance-map",
        r"p{0.27\textwidth} p{0.16\textwidth} r p{0.15\textwidth} p{0.31\textwidth}",
        ["Scientific family", "Eligibility", "Cells", "Datasets", "Canonical lock"],
        (
            (r["family"], r["eligibility"], r["cells"], r["datasets"], path_cell(r["locks"]))
            for r in selected
        ),
        "The complete 247-record evidence inventory and 6,796-record scalar map remain machine-readable. This table shows only the core paths a reviewer needs to follow the printed results.",
    )


def build_raw_artifact_index() -> tuple[Path, Path, Path]:
    rows: list[dict[str, str]] = []
    for object_id, spec in RAW_TABLES.items():
        path = ROOT / spec["path"]
        actual_rows = len(read_csv(spec["path"]))
        digest = sha256(path)
        if actual_rows != spec["rows"]:
            raise RuntimeError(
                f"{object_id}: expected {spec['rows']} rows, found {actual_rows}"
            )
        if digest != spec["sha256"]:
            raise RuntimeError(f"{object_id}: frozen SHA-256 mismatch")
        schema = fieldnames(spec["path"])
        rows.append(
            {
                "object_id": object_id,
                "purpose": spec["purpose"],
                "artifact_path": spec["path"],
                "data_rows": str(actual_rows),
                "columns": str(len(schema)),
                "sha256": digest,
                "schema": ";".join(schema),
                "rendered_replacement": spec["replacement"],
                "reproduction_command": "gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py",
                "allocation_reason": "Exhaustive row-level evidence is preserved for machines; the PDF reports readable cards, aggregates, uncertainty, and interpretation.",
            }
        )

    VISUAL.mkdir(parents=True, exist_ok=True)
    csv_path = VISUAL / "RAW_TABLE_ARTIFACT_INDEX.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    md_path = VISUAL / "RAW_TABLE_ARTIFACT_INDEX.md"
    md_lines = [
        "# Raw table artifact index",
        "",
        "The visual rebuild removes four exhaustive row dumps from the rendered supplement. The scientific rows are unchanged and remain the machine-readable source of truth. The generator verifies each frozen row count and SHA-256 before producing any curated table.",
        "",
        "| Object | Rows | Columns | Artifact path | SHA-256 | Rendered replacement |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['object_id']} | {r['data_rows']} | {r['columns']} | {r['artifact_path']} | {r['sha256']} | {r['rendered_replacement']} |"
        )
    md_lines += ["", "## Schemas", ""]
    for r in rows:
        md_lines += [
            f"### {r['object_id']}",
            "",
            f"- Purpose: {r['purpose']}",
            f"- Path: {r['artifact_path']}",
            f"- Data rows: {r['data_rows']}",
            f"- SHA-256: {r['sha256']}",
            f"- Columns in order: {r['schema'].replace(';', ', ')}",
            f"- Regenerate/verify curated representation: {r['reproduction_command']}",
            "",
        ]
    md_lines += [
        "## Allocation rule",
        "",
        "The PDF is the human review interface: definitions, dataset/model/protocol cards, aggregate effects, uncertainty, correction status, feasibility exclusions, interpretation, and limitations. The artifact is the exhaustive interface: one row per seed, context, or legacy test, with full paths and provenance. Moving these rows is a publication-design change only and does not change a value, test, claim status, or evidence scope.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_rows = []
    for r in rows:
        key_columns = ", ".join(r["schema"].split(";")[:6])
        tex_rows.append(
            (
                r["object_id"].replace("_", " "),
                r["data_rows"],
                path_cell(r["artifact_path"]),
                hash_cell(r["sha256"]),
                key_columns + "; ...",
            )
        )
    tex_path = longtable(
        "table_v2_raw_artifact_index.tex",
        "Machine-readable destination of the four exhaustive tables removed from the PDF.",
        "tab:s-raw-artifacts",
        r"p{0.16\textwidth} r p{0.28\textwidth} p{0.25\textwidth} p{0.22\textwidth}",
        ["Evidence family", "Rows", "Artifact path", "SHA-256", "Schema prefix"],
        tex_rows,
        r"Full ordered schemas, complete hashes, and the common regeneration command are recorded in \path{results/tkde_visual_rebuild/RAW_TABLE_ARTIFACT_INDEX.csv} and \path{results/tkde_visual_rebuild/RAW_TABLE_ARTIFACT_INDEX.md}. No row is deleted or converted to a PDF-only value.",
    )
    return csv_path, md_path, tex_path


def write_manifest(paths: Sequence[Path], source_paths: Sequence[str]) -> Path:
    manifest = VISUAL / "CURATED_SUPPLEMENT_TABLE_MANIFEST.csv"
    rows = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        forbidden = [
            token
            for token in (
                r"\tiny",
                r"\scriptsize",
                r"\resizebox",
                r"\begin{landscape}",
                r"\begin{sidewaystable}",
                r"\newpage",
                r"\clearpage",
            )
            if token in text
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden layout token(s) in {path}: {forbidden}")
        if r"\footnotesize" not in text:
            raise RuntimeError(f"Curated table lacks footnotesize declaration: {path}")
        rows.append(
            {
                "table_fragment": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "orientation": "portrait",
                "body_size": "footnotesize",
                "raw_row_dump": "False",
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> int:
    VISUAL.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    # The baseline table_s*.tex files are typeset seed/context dumps or their
    # landscape-era companions.  Their CSV sources remain frozen and indexed;
    # remove only the obsolete rendered fragments so a clean rebuild cannot
    # accidentally package both table systems.
    for legacy_fragment in TABLE_DIR.glob("table_s*.tex"):
        legacy_fragment.unlink()
    frozen = frozen_hash_map()
    sources = [
        "results/tkde_rebuild/CLAIM_EVIDENCE_LEDGER.csv",
        "results/tkde_rebuild/DATASET_TASK_STATISTICS.csv",
        "results/tkde_rebuild/PROTOCOL_DEFINITIONS.csv",
        "results/tkde_rebuild/MODEL_CONSTRUCTION_INVENTORY.csv",
        "results/tkde_rebuild/TRAINING_CONFIGURATION.csv",
        "results/tkde_rebuild/RB09_PROTOCOL_EFFECTS.csv",
        "results/tkde_rebuild/V24_DUPLICATE_STRESS_AUDIT.csv",
        "results/tkde_rebuild/RESOURCE_BOUNDARIES.csv",
        "results/tkde_rebuild/FRAMEWORK_VALIDATION_CASES.csv",
        "results/tkde_rebuild/FALSE_PROMOTION_AUDIT.csv",
        "results/tkde_rebuild/IBM_CELL_SUMMARY.csv",
        "results/tkde_rebuild/IBM_MATCHED_ABLATION_EFFECTS.csv",
        "results/tkde_rebuild/IBM_RUNTIME_FEASIBILITY.csv",
        "results/tkde_rebuild/IBM_RANK_DIVERGENCE.csv",
        "results/tkde_rebuild/GRAPHSAFE_BOUNDED_SUMMARY.csv",
        "results/tkde_rebuild/GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE.csv",
        "results/tkde_rebuild/GRAPHSAFE_CALIBRATION_SUMMARY.csv",
        "results/tkde_rebuild/REVIEW_BUDGET_CURVES.csv",
        "results/tkde_rebuild/table_data/EVIDENCE_FAMILY_SUMMARY.csv",
        "manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv",
        "manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv",
    ]
    for source in sources:
        verify_input(source, frozen)
    for spec in RAW_TABLES.values():
        verify_input(spec["path"], frozen)

    claims = read_csv("results/tkde_rebuild/CLAIM_EVIDENCE_LEDGER.csv")
    datasets = read_csv("results/tkde_rebuild/DATASET_TASK_STATISTICS.csv")
    protocols = read_csv("results/tkde_rebuild/PROTOCOL_DEFINITIONS.csv")
    inventory = read_csv("results/tkde_rebuild/MODEL_CONSTRUCTION_INVENTORY.csv")
    training = read_csv("results/tkde_rebuild/TRAINING_CONFIGURATION.csv")
    effects = read_csv("results/tkde_rebuild/RB09_PROTOCOL_EFFECTS.csv")
    v24 = read_csv("results/tkde_rebuild/V24_DUPLICATE_STRESS_AUDIT.csv")
    resources = read_csv("results/tkde_rebuild/RESOURCE_BOUNDARIES.csv")
    validation = read_csv("results/tkde_rebuild/FRAMEWORK_VALIDATION_CASES.csv")
    false_promotions = read_csv("results/tkde_rebuild/FALSE_PROMOTION_AUDIT.csv")
    ibm_cells = read_csv("results/tkde_rebuild/IBM_CELL_SUMMARY.csv")
    ibm_effects = read_csv("results/tkde_rebuild/IBM_MATCHED_ABLATION_EFFECTS.csv")
    ibm_runtime = read_csv("results/tkde_rebuild/IBM_RUNTIME_FEASIBILITY.csv")
    rank_rows = read_csv("results/tkde_rebuild/IBM_RANK_DIVERGENCE.csv")
    graphsafe = read_csv("results/tkde_rebuild/GRAPHSAFE_BOUNDED_SUMMARY.csv")
    graphsafe_paired = read_csv(
        "results/tkde_rebuild/GRAPHSAFE_PAIRED_VS_SIMPLE_AVERAGE.csv"
    )
    calibration = read_csv("results/tkde_rebuild/GRAPHSAFE_CALIBRATION_SUMMARY.csv")
    budgets = read_csv("results/tkde_rebuild/REVIEW_BUDGET_CURVES.csv")
    families = read_csv("results/tkde_rebuild/table_data/EVIDENCE_FAMILY_SUMMARY.csv")
    v22_lanes = read_csv("manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv")
    v22_tests = read_csv("manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv")

    outputs: list[Path] = []
    outputs += [build_navigation(), build_notation(), build_support_properties()]
    outputs += [build_validation_cases(validation)]
    outputs += build_dataset_cards(datasets)
    outputs += build_protocol_tables(protocols, datasets)
    outputs += build_model_and_construction_tables(inventory, training)
    outputs += build_metrics_and_statistics()
    outputs += build_protocol_effect_tables(effects)
    outputs += build_robustness_tables(v24, v22_lanes, v22_tests)
    outputs += [build_ibm_baseline(ibm_cells)]
    outputs += build_ibm_effects(ibm_effects)
    outputs += build_ibm_runtime(ibm_runtime)
    outputs += build_rank_tables(rank_rows)
    outputs += build_review_and_graphsafe(
        budgets, calibration, graphsafe, graphsafe_paired
    )
    outputs += build_resource_and_integrity(resources, false_promotions)
    outputs += build_claim_tables(claims)
    outputs += [build_provenance_map(families)]
    _, _, tex_index = build_raw_artifact_index()
    outputs.append(tex_index)
    manifest = write_manifest(outputs, sources)

    print(
        f"Wrote {len(outputs)} curated portrait table fragments; "
        f"verified 22 claim scopes and raw row counts 180/198/840/208; "
        f"manifest={manifest.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
