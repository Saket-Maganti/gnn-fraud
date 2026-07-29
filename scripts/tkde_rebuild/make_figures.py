#!/usr/bin/env python3
"""Generate the publication-design TKDE figures from frozen analysis CSVs.

This is a representation-only generator.  It reads the approved CPU-derived
analysis surfaces, writes exact-size PDF/PNG endpoints, exports every plotted
value and interval endpoint, and records hashes for both inputs and outputs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tkde_visual_rebuild.publication_style import (  # noqa: E402
    ACCENT_BLUE,
    ACCENT_TEAL,
    ACCENT_VERMILION,
    BASELINE_STYLES,
    BASE_FONT_PT,
    BLACK,
    CONFIG_LABELS,
    DARK_GRAY,
    ERROR_CAP_PT,
    FIGURE_SPECS,
    LEGEND_FONT_PT,
    LIGHT_GRAY,
    MID_GRAY,
    MIN_LINE_PT,
    MIN_MARKER_PT,
    MODEL_STYLES,
    PALE_GRAY,
    PANEL_FONT_PT,
    PANEL_GRAY,
    POLICY_STYLES,
    STATUS_SHORT_LABELS,
    TICK_FONT_PT,
    WHITE,
    panel_title,
    save_exact,
)


SOURCE = ROOT / "results" / "tkde_rebuild"
OUT = ROOT / "results" / "tkde_visual_rebuild"
DATA = OUT / "figure_data"
FIG = ROOT / "paper_tkde" / "figures"


def rel(path: Path | str) -> str:
    """Return a stable repository-relative path."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(filename: str, required: set[str]) -> pd.DataFrame:
    path = SOURCE / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{filename} missing required columns: {missing}")
    return frame


def export_source(frame: pd.DataFrame, stem: str) -> Path:
    """Write a stable, complete source CSV for one rendered figure."""

    path = DATA / f"{stem}.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def finish(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    pdf, png = save_exact(figure, stem, FIG)
    plt.close(figure)
    return pdf, png


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=MIN_LINE_PT,
            color=DARK_GRAY,
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig01_contract() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Compact six-coordinate band plus the formal support flow."""

    axes = load_csv(
        "DEPLOYMENT_CONTRACT_AXES.csv",
        {"symbol", "axis", "contract_contents", "why_it_matters"},
    )
    expected_symbols = ["T", "V", "C", "S", "B", "R"]
    if axes["symbol"].tolist() != expected_symbols:
        raise ValueError("deployment-coordinate order changed from T,V,C,S,B,R")

    short = {
        "T": ("Time", "windows / order"),
        "V": ("Visibility", "nodes / edges"),
        "C": ("Construction", "entities / features"),
        "S": ("Selection", "validation / policy"),
        "B": ("Decision", "threshold / budget"),
        "R": ("Resources", "compute / status"),
    }
    flow = [
        ("contract", "Deployment contract", r"$\Pi$"),
        ("evidence", "Evidence unit", r"$e$"),
        ("support", "Support test", r"$\mathcal{E}\models c$"),
        ("claim", "Typed claim", r"$c$"),
    ]
    statuses = [
        ("supported", "supported"),
        ("blocked", "blocked"),
        ("resource_blocked", "resource-blocked"),
        ("diagnostic", "diagnostic"),
        ("excluded", "excluded"),
        ("refuted", "refuted"),
    ]

    source_rows: list[dict[str, object]] = []
    for order, row in axes.reset_index(drop=True).iterrows():
        label, detail = short[row.symbol]
        source_rows.append(
            {
                "record_type": "contract_coordinate",
                "record_id": row.symbol,
                "label": label,
                "display_detail": detail,
                "order": order + 1,
                "contract_contents": row.contract_contents,
                "why_it_matters": row.why_it_matters,
            }
        )
    for order, (record_id, label, detail) in enumerate(flow, start=1):
        source_rows.append(
            {
                "record_type": "formal_flow",
                "record_id": record_id,
                "label": label,
                "display_detail": detail,
                "order": order,
                "contract_contents": "code-authored formal architecture",
                "why_it_matters": "maps a recorded contract to a scoped claim",
            }
        )
    for order, (record_id, label) in enumerate(statuses, start=1):
        source_rows.append(
            {
                "record_type": "status_semantics",
                "record_id": record_id,
                "label": label,
                "display_detail": "nonnumeric status",
                "order": order,
                "contract_contents": "compact legend label",
                "why_it_matters": "status is not a predictive score",
            }
        )
    source = pd.DataFrame(source_rows)

    spec = FIGURE_SPECS["fig01_deployment_contract"]
    figure, axis = plt.subplots(figsize=(spec.width, spec.height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    left, gap = 0.012, 0.008
    width = (0.976 - 5 * gap) / 6
    for index, row in axes.reset_index(drop=True).iterrows():
        x = left + index * (width + gap)
        axis.add_patch(
            Rectangle(
                (x, 0.61),
                width,
                0.31,
                facecolor=PANEL_GRAY if index % 2 == 0 else WHITE,
                edgecolor=DARK_GRAY,
                linewidth=0.8,
            )
        )
        label, detail = short[row.symbol]
        axis.text(
            x + width / 2,
            0.805,
            f"{row.symbol}  {label}",
            ha="center",
            va="center",
            fontsize=PANEL_FONT_PT,
            fontweight="bold",
            color=BLACK,
        )
        axis.text(
            x + width / 2,
            0.69,
            detail,
            ha="center",
            va="center",
            fontsize=TICK_FONT_PT,
            color=DARK_GRAY,
        )

    flow_x = [0.055, 0.31, 0.565, 0.82]
    flow_width = 0.13
    for index, (_, label, detail) in enumerate(flow):
        x = flow_x[index]
        axis.add_patch(
            Rectangle(
                (x, 0.29),
                flow_width,
                0.19,
                facecolor=WHITE,
                edgecolor=ACCENT_BLUE if index in {0, 3} else DARK_GRAY,
                linewidth=1.0 if index in {0, 3} else 0.8,
            )
        )
        display_label = "Deployment\ncontract" if index == 0 else label
        axis.text(
            x + flow_width / 2,
            0.407,
            display_label,
            ha="center",
            va="center",
            fontsize=TICK_FONT_PT if index == 0 else BASE_FONT_PT,
            fontweight="bold" if index in {0, 3} else "normal",
            linespacing=0.92,
        )
        axis.text(
            x + flow_width / 2,
            0.320 if index == 0 else 0.335,
            detail,
            ha="center",
            va="center",
            fontsize=BASE_FONT_PT,
        )
        if index < len(flow) - 1:
            _arrow(axis, (x + flow_width + 0.012, 0.385), (flow_x[index + 1] - 0.012, 0.385))

    marker_styles = ["o", "o", "s", "D", "x", "X"]
    marker_faces = [BLACK, WHITE, WHITE, LIGHT_GRAY, BLACK, BLACK]
    status_x = [0.03, 0.18, 0.315, 0.54, 0.70, 0.845]
    for (record_id, label), marker, face, x in zip(statuses, marker_styles, marker_faces, status_x):
        del record_id
        marker_kwargs = {"color": BLACK} if marker in {"x", "X"} else {"facecolor": face, "edgecolor": BLACK}
        axis.scatter([x], [0.10], marker=marker, s=23, linewidth=0.8, clip_on=False, **marker_kwargs)
        axis.text(x + 0.018, 0.10, label, ha="left", va="center", fontsize=TICK_FONT_PT)
    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.01)
    return source, finish(figure, "fig01_deployment_contract")


def fig02_protocol() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Paired protocol effects plus strict/isolated slopes at 8 pt."""

    data = load_csv(
        "RB09_AUPRC_MAIN.csv",
        {
            "dataset",
            "model",
            "n_pairs",
            "strict_mean",
            "isolated_mean",
            "delta_isolated_minus_strict",
            "delta_ci95_low",
            "delta_ci95_high",
        },
    )
    order = [
        ("elliptic", "gcn"),
        ("elliptic", "mlp"),
        ("elliptic", "sage"),
        ("dgraphfin", "gcn"),
        ("dgraphfin", "mlp"),
        ("dgraphfin", "sage"),
    ]
    ordered = pd.concat(
        [data[(data.dataset == dataset) & (data.model == model)] for dataset, model in order],
        ignore_index=True,
    )
    if len(ordered) != 6:
        raise ValueError("F02 requires the complete 2 dataset x 3 model grid")
    ordered["display_order"] = np.arange(1, len(ordered) + 1)
    ordered["model_label"] = ordered.model.map({"gcn": "GCN", "mlp": "MLP", "sage": "SAGE"})
    ordered["dataset_label"] = ordered.dataset.map({"elliptic": "Elliptic", "dgraphfin": "DGraphFin"})

    spec = FIGURE_SPECS["fig02_protocol_architecture_effects"]
    figure = plt.figure(figsize=(spec.width, spec.height))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.48, 1.0, 1.0], wspace=0.38)
    forest = figure.add_subplot(grid[0, 0])
    y = np.arange(len(ordered))[::-1]
    for yi, (_, row) in zip(y, ordered.iterrows()):
        style = MODEL_STYLES[row.model]
        forest.errorbar(
            row.delta_isolated_minus_strict,
            yi,
            xerr=[
                [row.delta_isolated_minus_strict - row.delta_ci95_low],
                [row.delta_ci95_high - row.delta_isolated_minus_strict],
            ],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markerfacecolor=style["color"] if row.model != "mlp" else WHITE,
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            markersize=MIN_MARKER_PT + 0.5,
            capsize=ERROR_CAP_PT,
            linewidth=1.0,
        )
    forest.axvline(0, color=DARK_GRAY, linewidth=0.8, linestyle="--")
    forest.set_yticks(
        y,
        [f"{row.dataset_label} / {row.model_label}" for _, row in ordered.iterrows()],
    )
    forest.set_xlabel("Paired AUPRC delta (isolated - strict)")
    panel_title(forest, "a", "Paired effect with 95% CI")
    forest.grid(axis="x")
    forest.xaxis.set_major_locator(MaxNLocator(nbins=5))
    forest.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for column, dataset in enumerate(["elliptic", "dgraphfin"], start=1):
        axis = figure.add_subplot(grid[0, column])
        subset = data[data.dataset.eq(dataset)].copy()
        for model in ["gcn", "mlp", "sage"]:
            row = subset[subset.model.eq(model)].iloc[0]
            style = MODEL_STYLES[model]
            values = [row.strict_mean, row.isolated_mean]
            axis.plot(
                [0, 1],
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                markersize=MIN_MARKER_PT + 0.3,
                markerfacecolor=style["color"] if model != "mlp" else WHITE,
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
            )
            axis.text(
                1.035,
                values[1],
                model.upper() if model != "sage" else "SAGE",
                va="center",
                fontsize=TICK_FONT_PT,
                color=style["color"],
            )
        axis.set_xticks([0, 1], ["Strict", "Isolated"])
        axis.set_xlim(-0.10, 1.40)
        axis.set_ylabel("Mean AUPRC")
        dataset_title = "Elliptic" if dataset == "elliptic" else "DGraphFin"
        panel_title(axis, "b" if dataset == "elliptic" else "c", dataset_title)
        axis.grid(axis="y")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if dataset == "elliptic" else "%.3f"))
    figure.subplots_adjust(left=0.17, right=0.982, top=0.92, bottom=0.19)
    return ordered, finish(figure, "fig02_protocol_architecture_effects")


def fig03_ibm() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """One-column IBM baseline-family AUPRC comparison."""

    data = load_csv(
        "IBM_CELL_SUMMARY.csv",
        {
            "version",
            "variant",
            "size",
            "regime",
            "protocol",
            "config",
            "n",
            "auprc_mean",
            "auprc_ci95_low",
            "auprc_ci95_high",
        },
    )
    baseline_configs = {
        "hist_gradient_boosting_edge_features",
        "logistic_regression_edge_features",
        "graphsage_edge_minibatch_h32",
    }
    baseline = data[data.version.eq("V26") & data.config.isin(baseline_configs)].copy()
    if len(baseline) != 24 or set(baseline.n.astype(int)) != {10}:
        raise ValueError("F03 requires 24 V26 baseline rows with ten seeds per cell")
    baseline["family"] = baseline.config.map(CONFIG_LABELS)
    baseline["protocol_label"] = baseline.protocol.map(
        {"early_to_late_transfer": "early", "late_window_holdout": "late"}
    )
    variant_order = ["hi-small", "li-small", "hi-medium", "li-medium"]
    protocol_order = ["early_to_late_transfer", "late_window_holdout"]
    contexts = [(variant, protocol) for variant in variant_order for protocol in protocol_order]
    y_positions = [8.0, 7.0, 6.0, 5.0, 3.5, 2.5, 1.5, 0.5]
    context_y = {context: y for context, y in zip(contexts, y_positions)}
    context_labels = {
        (variant, protocol): f"{variant.split('-')[0].upper()}-{variant.split('-')[1][0].upper()} / {'early' if protocol.startswith('early') else 'late'}"
        for variant, protocol in contexts
    }
    family_offsets = {"HistGB": 0.20, "LogReg": 0.0, "SAGE-h32": -0.20}
    baseline["plot_y"] = baseline.apply(
        lambda row: context_y[(row.variant, row.protocol)] + family_offsets[row.family], axis=1
    )
    baseline["context_label"] = baseline.apply(
        lambda row: context_labels[(row.variant, row.protocol)], axis=1
    )
    baseline = baseline.sort_values(
        ["size", "regime", "protocol", "family"],
        key=lambda col: col.map(
            {
                "small": 0,
                "medium": 1,
                "hi": 0,
                "li": 1,
                "early_to_late_transfer": 0,
                "late_window_holdout": 1,
                "HistGB": 0,
                "LogReg": 1,
                "SAGE-h32": 2,
            }
        ),
    )

    spec = FIGURE_SPECS["fig03_ibm_metric_scale_construction"]
    figure, axis = plt.subplots(figsize=(spec.width, spec.height))
    for _, row in baseline.iterrows():
        style = BASELINE_STYLES[row.family]
        axis.errorbar(
            row.auprc_mean,
            row.plot_y,
            xerr=[
                [row.auprc_mean - row.auprc_ci95_low],
                [row.auprc_ci95_high - row.auprc_mean],
            ],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markerfacecolor=style["facecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            markersize=MIN_MARKER_PT,
            capsize=ERROR_CAP_PT,
            linewidth=0.9,
        )
    axis.axhline(4.25, color=LIGHT_GRAY, linewidth=0.7)
    axis.text(0.995, 4.43, "Medium", transform=axis.get_yaxis_transform(), ha="right", va="bottom", fontsize=TICK_FONT_PT, color=DARK_GRAY)
    axis.text(0.995, 8.42, "Small", transform=axis.get_yaxis_transform(), ha="right", va="bottom", fontsize=TICK_FONT_PT, color=DARK_GRAY)
    axis.set_yticks(
        y_positions,
        [context_labels[context] for context in contexts],
    )
    axis.set_ylim(0.05, 8.75)
    axis.set_xlim(left=0, right=max(0.185, baseline.auprc_ci95_high.max() * 1.05))
    axis.set_xlabel("AUPRC (mean with 95% bootstrap CI)")
    axis.grid(axis="x")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_title("IBM AML baseline families", loc="left", fontsize=PANEL_FONT_PT, pad=5)
    handles = [
        Line2D(
            [0],
            [0],
            marker=BASELINE_STYLES[label]["marker"],
            color=BASELINE_STYLES[label]["color"],
            markerfacecolor=BASELINE_STYLES[label]["facecolor"],
            markeredgecolor=BASELINE_STYLES[label]["color"],
            linestyle="none",
            label="LogReg (det.)" if label == "LogReg" else label,
        )
        for label in ["HistGB", "LogReg", "SAGE-h32"]
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.965), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.285, right=0.975, top=0.86, bottom=0.13)
    return baseline.reset_index(drop=True), finish(figure, "fig03_ibm_metric_scale_construction")


def _rank_context_label(variant: str, protocol: str) -> str:
    regime, size = variant.split("-", maxsplit=1)
    return f"{regime.upper()}-{size[0].upper()} / {'early' if protocol.startswith('early') else 'late'}"


def fig04_rank() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """One-column vertical summary of all cells plus one explicit reversal."""

    cells = load_csv(
        "IBM_RANK_DIVERGENCE.csv",
        {
            "family",
            "variant",
            "protocol",
            "n_configurations",
            "auprc_winner",
            "f1_winner",
            "auprc_f1_winner_disagree",
            "spearman_auprc_vs_f1",
            "spearman_auprc_vs_f1_p",
        },
    )
    ranks = load_csv(
        "IBM_METRIC_RANKS.csv",
        {"family", "variant", "protocol", "config", "auprc", "f1", "auprc_rank", "f1_rank"},
    )
    if len(cells) != 16:
        raise ValueError("F04 requires all 16 baseline/graph rank-divergence cells")

    variant_order = ["hi-small", "hi-medium", "li-small", "li-medium"]
    protocol_order = ["early_to_late_transfer", "late_window_holdout"]
    contexts = [(variant, protocol) for variant in variant_order for protocol in protocol_order]
    context_to_y = {context: len(contexts) - 1 - i for i, context in enumerate(contexts)}
    cells = cells.copy()
    cells["context_label"] = cells.apply(
        lambda row: _rank_context_label(row.variant, row.protocol), axis=1
    )
    cells["plot_y"] = cells.apply(
        lambda row: context_to_y[(row.variant, row.protocol)]
        + (0.14 if row.family == "baseline_grid" else -0.14),
        axis=1,
    )

    candidates = cells[
        cells.family.eq("graph_grid") & cells.auprc_f1_winner_disagree.astype(bool)
    ].sort_values(["spearman_auprc_vs_f1", "variant", "protocol"])
    if candidates.empty:
        raise ValueError("F04 requires at least one observed AUPRC/F1 winner reversal")
    example_cell = candidates.iloc[0]
    example = ranks[
        ranks.family.eq(example_cell.family)
        & ranks.variant.eq(example_cell.variant)
        & ranks.protocol.eq(example_cell.protocol)
    ].copy()
    if len(example) != int(example_cell.n_configurations):
        raise ValueError("F04 example rank rows do not match n_configurations")
    example["label"] = example.config.map(CONFIG_LABELS)
    if example.label.isna().any():
        raise ValueError("F04 encountered an unknown configuration label")

    source_rows: list[dict[str, object]] = []
    for _, row in cells.sort_values(["variant", "protocol", "family"]).iterrows():
        source_rows.append(
            {
                "record_type": "cell_correlation",
                "family": row.family,
                "variant": row.variant,
                "protocol": row.protocol,
                "context_label": row.context_label,
                "config": "",
                "config_label": "",
                "metric": "spearman_auprc_vs_f1",
                "value": row.spearman_auprc_vs_f1,
                "rank": np.nan,
                "p_value": row.spearman_auprc_vs_f1_p,
                "n_configurations": row.n_configurations,
                "auprc_winner": row.auprc_winner,
                "f1_winner": row.f1_winner,
                "winner_disagrees": bool(row.auprc_f1_winner_disagree),
                "selected_example": bool(
                    row.family == example_cell.family
                    and row.variant == example_cell.variant
                    and row.protocol == example_cell.protocol
                ),
            }
        )
    for _, row in example.sort_values("auprc_rank").iterrows():
        for metric in ["auprc", "f1"]:
            source_rows.append(
                {
                    "record_type": "example_rank",
                    "family": example_cell.family,
                    "variant": example_cell.variant,
                    "protocol": example_cell.protocol,
                    "context_label": _rank_context_label(example_cell.variant, example_cell.protocol),
                    "config": row.config,
                    "config_label": row.label,
                    "metric": metric,
                    "value": row[metric],
                    "rank": row[f"{metric}_rank"],
                    "p_value": np.nan,
                    "n_configurations": example_cell.n_configurations,
                    "auprc_winner": example_cell.auprc_winner,
                    "f1_winner": example_cell.f1_winner,
                    "winner_disagrees": bool(example_cell.auprc_f1_winner_disagree),
                    "selected_example": True,
                }
            )
    source = pd.DataFrame(source_rows)

    spec = FIGURE_SPECS["fig04_rank_decision_divergence"]
    figure = plt.figure(figsize=(spec.width, spec.height))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.52, 1.0], hspace=0.52)
    agreement = figure.add_subplot(grid[0, 0])
    for _, row in cells.iterrows():
        marker = "o" if row.family == "baseline_grid" else "^"
        fill = BLACK if bool(row.auprc_f1_winner_disagree) else WHITE
        agreement.scatter(
            row.spearman_auprc_vs_f1,
            row.plot_y,
            marker=marker,
            s=31,
            facecolor=fill,
            edgecolor=BLACK,
            linewidth=0.8,
            zorder=3,
        )
    agreement.axvline(0, color=DARK_GRAY, linestyle="--", linewidth=0.8)
    agreement.set_xlim(-1.08, 1.08)
    agreement.set_ylim(-0.55, 7.55)
    agreement.set_yticks(
        list(context_to_y.values()),
        [_rank_context_label(variant, protocol) for variant, protocol in contexts],
    )
    agreement.set_xlabel(r"Spearman $\rho$: AUPRC rank vs F1 rank")
    panel_title(agreement, "a", "All matched cells")
    agreement.grid(axis="x")
    agreement.xaxis.set_major_locator(MaxNLocator(nbins=5))
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor=WHITE, markeredgecolor=BLACK, linestyle="none", label="baseline"),
            Line2D([0], [0], marker="^", markerfacecolor=WHITE, markeredgecolor=BLACK, linestyle="none", label="graph"),
            Line2D([0], [0], marker="o", markerfacecolor=BLACK, markeredgecolor=BLACK, linestyle="none", label="filled: winner differs"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.55,
    )

    slope = figure.add_subplot(grid[1, 0])
    short_rank_labels = {
        "Ref": "Ref",
        "No edge": "NoEdge",
        "Shuffle": "Shuffle",
        "Degree": "Degree",
        "Degree cap": "DegCap",
        "Recent": "Recent",
        "GINE": "GINE",
    }
    auprc_winner = CONFIG_LABELS.get(example_cell.auprc_winner, example_cell.auprc_winner)
    f1_winner = CONFIG_LABELS.get(example_cell.f1_winner, example_cell.f1_winner)
    for _, row in example.sort_values("auprc_rank").iterrows():
        is_auprc_winner = row.label == auprc_winner
        is_f1_winner = row.label == f1_winner
        if is_auprc_winner:
            color, linewidth = ACCENT_BLUE, 1.5
        elif is_f1_winner:
            color, linewidth = ACCENT_VERMILION, 1.5
        else:
            color, linewidth = MID_GRAY, 0.85
        slope.plot(
            [0, 1],
            [row.auprc_rank, row.f1_rank],
            color=color,
            linewidth=linewidth,
            marker="o",
            markersize=MIN_MARKER_PT,
            markerfacecolor=WHITE if not (is_auprc_winner or is_f1_winner) else color,
            markeredgecolor=color,
            markeredgewidth=0.8,
        )
        label = short_rank_labels[row.label]
        weight = "bold" if is_auprc_winner or is_f1_winner else "normal"
        slope.text(-0.08, row.auprc_rank, label, ha="right", va="center", fontsize=TICK_FONT_PT, fontweight=weight, color=color)
        slope.text(1.08, row.f1_rank, label, ha="left", va="center", fontsize=TICK_FONT_PT, fontweight=weight, color=color)
    n_configurations = int(example_cell.n_configurations)
    slope.set_xlim(-0.68, 1.62)
    slope.set_ylim(n_configurations + 0.45, 0.55)
    slope.set_xticks([0, 1], ["AUPRC", "F1 @ 0.5"])
    slope.set_yticks(range(1, n_configurations + 1))
    slope.set_ylabel("Rank (1 best)")
    panel_title(
        slope,
        "b",
        f"{_rank_context_label(example_cell.variant, example_cell.protocol)} graph grid; "
        + rf"$\rho={example_cell.spearman_auprc_vs_f1:.2f}$",
    )
    slope.grid(axis="y")
    figure.subplots_adjust(left=0.305, right=0.84, top=0.89, bottom=0.10)
    return source, finish(figure, "fig04_rank_decision_divergence")


def fig05_ablation() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Two-panel, metric-separated matched construction effects."""

    effects = load_csv(
        "IBM_MATCHED_ABLATION_EFFECTS.csv",
        {
            "config",
            "size",
            "metric",
            "n_pairs",
            "mean_delta",
            "delta_ci95_low",
            "delta_ci95_high",
            "holm_p_within_size_metric",
            "source_paths",
        },
    )
    runtime = load_csv(
        "IBM_RUNTIME_FEASIBILITY.csv",
        {"variant", "size", "protocol", "config", "status", "n", "source_paths"},
    )
    metrics = ["auprc", "auroc", "f1"]
    data = effects[
        effects.metric.isin(metrics)
        & ~effects.config.eq("account_account_sender_receiver")
    ].copy()
    data["label"] = data.config.map(CONFIG_LABELS)
    data["status"] = "MEASURED_FULL10"
    data["significant_holm"] = data.holm_p_within_size_metric < 0.05
    if data.label.isna().any():
        raise ValueError("F05 encountered an unknown construction label")

    blocked = runtime[
        runtime["size"].eq("medium")
        & runtime.config.eq("gine_light_h64")
        & runtime.status.eq("RESOURCE_BLOCKED_T4_CUDA_OOM")
    ]
    if len(blocked) != 2 or not blocked.n.eq(0).all():
        raise ValueError("F05 requires both Medium GINE variants to be nonnumeric T4-OOM rows")
    blocked_source = ";".join(sorted(set(blocked.source_paths.astype(str))))
    blocked_rows = []
    for metric in metrics:
        blocked_rows.append(
            {
                "config": "gine_light_h64",
                "size": "medium",
                "metric": metric,
                "n_pairs": 0,
                "n_raw_context_seed_pairs": 0,
                "n_variant_protocol_cells": 0,
                "reference_mean": np.nan,
                "candidate_mean": np.nan,
                "mean_delta": np.nan,
                "delta_ci95_low": np.nan,
                "delta_ci95_high": np.nan,
                "relative_delta_pct": np.nan,
                "cohen_dz": np.nan,
                "wilcoxon_p": np.nan,
                "context_delta_min": np.nan,
                "context_delta_max": np.nan,
                "n_contexts_positive": 0,
                "n_contexts_negative": 0,
                "n_contexts_zero": 0,
                "all_nonzero_contexts_same_sign": False,
                "source_paths": blocked_source,
                "holm_p_within_size_metric": np.nan,
                "label": "GINE",
                "status": "RESOURCE_BLOCKED_T4_CUDA_OOM",
                "significant_holm": False,
            }
        )
    source = pd.concat([data, pd.DataFrame(blocked_rows)], ignore_index=True, sort=False)
    numeric_columns = ["mean_delta", "delta_ci95_low", "delta_ci95_high"]
    if source.loc[source.status.ne("MEASURED_FULL10"), numeric_columns].notna().any().any():
        raise ValueError("resource-blocked F05 cells must remain nonnumeric")

    label_order = ["No edge", "Shuffle", "Degree", "Degree cap", "Recent", "GINE"]
    y_positions = {label: len(label_order) - 1 - i for i, label in enumerate(label_order)}
    metric_titles = {"auprc": "AUPRC", "auroc": "AUROC", "f1": "F1 @ 0.5"}
    spec = FIGURE_SPECS["fig05_matched_ablation_effects"]
    figure, axes = plt.subplots(2, 3, figsize=(spec.width, spec.height), squeeze=False)
    for row_index, size in enumerate(["small", "medium"]):
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            subset = source[
                source["size"].eq(size)
                & source.metric.eq(metric)
                & source.status.eq("MEASURED_FULL10")
            ]
            for _, row in subset.iterrows():
                yi = y_positions[row.label]
                significant = bool(row.significant_holm)
                axis.errorbar(
                    row.mean_delta,
                    yi,
                    xerr=[
                        [row.mean_delta - row.delta_ci95_low],
                        [row.delta_ci95_high - row.mean_delta],
                    ],
                    fmt="o",
                    color=BLACK,
                    ecolor=BLACK,
                    markerfacecolor=BLACK if significant else WHITE,
                    markeredgecolor=BLACK,
                    markeredgewidth=0.9,
                    markersize=MIN_MARKER_PT,
                    capsize=ERROR_CAP_PT,
                    linewidth=0.9,
                )
            axis.axvline(0, color=DARK_GRAY, linestyle="--", linewidth=0.8)
            endpoints = subset[["delta_ci95_low", "delta_ci95_high"]].to_numpy(dtype=float)
            limit = max(0.005, float(np.nanmax(np.abs(endpoints))) * 1.13)
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-0.55, len(label_order) - 0.45)
            axis.set_yticks(
                [y_positions[label] for label in label_order],
                label_order if column_index == 0 else [""] * len(label_order),
            )
            axis.set_title(metric_titles[metric], loc="center", fontsize=PANEL_FONT_PT, pad=4)
            axis.set_xlabel("Matched delta vs h64 reference")
            axis.grid(axis="x")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            if size == "medium":
                axis.text(
                    0.98,
                    y_positions["GINE"],
                    "T4 OOM",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=TICK_FONT_PT,
                    color=DARK_GRAY,
                    bbox={"facecolor": WHITE, "edgecolor": MID_GRAY, "linewidth": 0.7, "pad": 1.2},
                )
    figure.text(0.012, 0.91, "(a) Small - six feasible constructions", fontsize=PANEL_FONT_PT, fontweight="bold", ha="left")
    figure.text(0.012, 0.47, "(b) Medium - five measured; GINE is resource-blocked", fontsize=PANEL_FONT_PT, fontweight="bold", ha="left")
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor=BLACK, markeredgecolor=BLACK, linestyle="none", label="Holm p < .05"),
            Line2D([0], [0], marker="o", markerfacecolor=WHITE, markeredgecolor=BLACK, linestyle="none", label="not Holm-significant"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        ncol=2,
    )
    figure.subplots_adjust(left=0.12, right=0.975, top=0.84, bottom=0.09, hspace=0.92, wspace=0.28)
    source["plot_order"] = source.label.map(y_positions)
    return source.sort_values(["size", "metric", "plot_order"], ascending=[True, True, False]).reset_index(drop=True), finish(figure, "fig05_matched_ablation_effects")


def fig06_runtime() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Grayscale-first runtime/Pareto view with blocked cells off-scale."""

    data = load_csv(
        "IBM_RUNTIME_FEASIBILITY.csv",
        {
            "variant",
            "size",
            "regime",
            "protocol",
            "config",
            "status",
            "n",
            "auprc_mean",
            "auprc_std",
            "runtime_seconds_mean",
            "runtime_seconds_min",
            "runtime_seconds_max",
            "pareto_auprc_runtime_within_cell",
        },
    ).copy()
    blocked_mask = data.status.ne("MEASURED_FULL10")
    numeric = [
        "auprc_mean",
        "auprc_std",
        "runtime_seconds_mean",
        "runtime_seconds_min",
        "runtime_seconds_max",
    ]
    if data.loc[blocked_mask, numeric].notna().any().any():
        raise ValueError("resource-blocked F06 rows must not contain numeric performance/runtime")
    if not data.loc[blocked_mask, "n"].eq(0).all():
        raise ValueError("resource-blocked F06 rows must have n=0")

    data["config_label"] = data.config.map(CONFIG_LABELS)
    if data.config_label.isna().any():
        raise ValueError("F06 encountered an unknown configuration label")
    measured = ~blocked_mask
    data["runtime_interval_low"] = np.where(measured, data.runtime_seconds_min, np.nan)
    data["runtime_interval_high"] = np.where(measured, data.runtime_seconds_max, np.nan)
    data["auprc_sd_low"] = np.where(measured, np.maximum(0.0, data.auprc_mean - data.auprc_std), np.nan)
    data["auprc_sd_high"] = np.where(measured, np.minimum(1.0, data.auprc_mean + data.auprc_std), np.nan)
    data["protocol_label"] = data.protocol.map(
        {
            "early_to_late_transfer": "early-to-late",
            "late_window_holdout": "late holdout",
            "both planned protocols": "both planned protocols",
        }
    )

    spec = FIGURE_SPECS["fig06_runtime_resource_pareto"]
    figure, axes = plt.subplots(2, 2, figsize=(spec.width, spec.height), squeeze=False)
    variant_order = ["hi-small", "hi-medium", "li-small", "li-medium"]
    label_offsets = {
        ("hi-small", "Degree cap"): (5, 7),
        ("hi-small", "Recent"): (5, -14),
        ("hi-small", "GINE"): (-5, 7),
        ("hi-medium", "Ref"): (5, 8),
        ("hi-medium", "Degree"): (5, -10),
        ("hi-medium", "Recent"): (5, -3),
        ("li-small", "Degree cap"): (8, 8),
        ("li-small", "Ref"): (8, 17),
        ("li-small", "Recent"): (7, -15),
        ("li-small", "GINE"): (-5, 7),
        ("li-medium", "Ref"): (5, 8),
        ("li-medium", "Recent"): (5, -11),
    }
    for axis, variant, panel in zip(axes.flat, variant_order, list("abcd")):
        subset = data[data.variant.eq(variant) & data.status.eq("MEASURED_FULL10")].copy()
        for _, row in subset.iterrows():
            pareto = bool(row.pareto_auprc_runtime_within_cell)
            marker = "o" if row.protocol == "early_to_late_transfer" else "s"
            if row.config_label == "Ref":
                edge = ACCENT_BLUE
                face = ACCENT_BLUE if pareto else WHITE
            elif row.config_label == "GINE":
                edge = ACCENT_VERMILION
                face = ACCENT_VERMILION if pareto else WHITE
            elif pareto:
                edge = BLACK
                face = BLACK
            else:
                edge = LIGHT_GRAY
                face = WHITE
            axis.errorbar(
                row.runtime_seconds_mean,
                row.auprc_mean,
                xerr=[
                    [row.runtime_seconds_mean - row.runtime_interval_low],
                    [row.runtime_interval_high - row.runtime_seconds_mean],
                ],
                yerr=[
                    [row.auprc_mean - row.auprc_sd_low],
                    [row.auprc_sd_high - row.auprc_mean],
                ],
                fmt=marker,
                color=edge,
                ecolor=edge,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=0.9,
                markersize=MIN_MARKER_PT,
                capsize=ERROR_CAP_PT,
                linewidth=0.75,
                alpha=1.0 if pareto or row.config_label in {"Ref", "GINE"} else 0.78,
                zorder=3 if pareto else 2,
            )
        pareto_labels = sorted(
            subset.loc[subset.pareto_auprc_runtime_within_cell.astype(bool), "config_label"].unique()
        )
        for label in pareto_labels:
            points = subset[
                subset.config_label.eq(label)
                & subset.pareto_auprc_runtime_within_cell.astype(bool)
            ]
            point = points.sort_values("auprc_mean").iloc[-1]
            dx, dy = label_offsets.get((variant, label), (5, 4))
            axis.annotate(
                label,
                (point.runtime_seconds_mean, point.auprc_mean),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                va="bottom" if dy >= 0 else "top",
                fontsize=TICK_FONT_PT,
                color=BLACK,
                arrowprops={"arrowstyle": "-", "color": MID_GRAY, "linewidth": 0.45},
            )
        blocked = data[data.variant.eq(variant) & data.status.ne("MEASURED_FULL10")]
        if not blocked.empty:
            axis.text(
                0.99,
                1.025,
                "GINE h64: T4 OOM (unmeasured)",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=TICK_FONT_PT,
                color=DARK_GRAY,
                bbox={"facecolor": WHITE, "edgecolor": MID_GRAY, "linewidth": 0.7, "pad": 1.5},
                clip_on=False,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Runtime per seed/protocol (s, log scale)")
        axis.set_ylabel("Mean AUPRC")
        panel_title(axis, panel, variant.replace("-", " / ").upper())
        axis.grid(which="major")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor=WHITE, markeredgecolor=BLACK, linestyle="none", label="early-to-late"),
            Line2D([0], [0], marker="s", markerfacecolor=WHITE, markeredgecolor=BLACK, linestyle="none", label="late holdout"),
            Line2D([0], [0], marker="o", markerfacecolor=BLACK, markeredgecolor=BLACK, linestyle="none", label="within-cell Pareto"),
            Line2D([0], [0], marker="o", markerfacecolor=WHITE, markeredgecolor=LIGHT_GRAY, linestyle="none", label="measured, dominated"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.105, hspace=0.43, wspace=0.27)
    return data.sort_values(["variant", "status", "protocol", "config"]).reset_index(drop=True), finish(figure, "fig06_runtime_resource_pareto")


def fig07_budget() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Supplement review-budget curves with exported 95% band endpoints."""

    data = load_csv(
        "REVIEW_BUDGET_CURVES.csv",
        {
            "dataset",
            "method",
            "budget_pct",
            "n",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
        },
    )
    methods = ["feature_only", "best_val_branch", "simple_average", "graphsafe_conservative"]
    source = data[data.method.isin(methods)].copy()
    if len(source) != 24:
        raise ValueError("F07 requires 2 datasets x 4 methods x 3 budgets")
    for metric in ["precision", "recall"]:
        half_width = 1.96 * source[f"{metric}_std"] / np.sqrt(source.n)
        source[f"{metric}_ci95_low"] = np.maximum(0.0, source[f"{metric}_mean"] - half_width)
        source[f"{metric}_ci95_high"] = np.minimum(1.0, source[f"{metric}_mean"] + half_width)
    source["method_label"] = source.method.map(
        {
            "feature_only": "Feature only",
            "best_val_branch": "Best val branch",
            "simple_average": "Simple average",
            "graphsafe_conservative": "GraphSafe conservative",
        }
    )

    spec = FIGURE_SPECS["fig07_review_budget_analysis"]
    figure, axes = plt.subplots(2, 2, figsize=(spec.width, spec.height), sharex=True, squeeze=False)
    for row_index, dataset in enumerate(["elliptic", "dgraphfin"]):
        for column_index, (metric, ylabel) in enumerate(
            [("precision", "Precision at budget"), ("recall", "Recall at budget")]
        ):
            axis = axes[row_index, column_index]
            for method in methods:
                group = source[
                    source.dataset.eq(dataset) & source.method.eq(method)
                ].sort_values("budget_pct")
                style = POLICY_STYLES[method]
                x = group.budget_pct.to_numpy(dtype=float)
                mean = group[f"{metric}_mean"].to_numpy(dtype=float)
                low = group[f"{metric}_ci95_low"].to_numpy(dtype=float)
                high = group[f"{metric}_ci95_high"].to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                    linewidth=1.25,
                    markersize=MIN_MARKER_PT,
                    markerfacecolor=WHITE if method in {"feature_only", "best_val_branch"} else style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=0.8,
                    label=group.method_label.iloc[0],
                )
                axis.fill_between(x, low, high, color=style["color"], alpha=0.08, linewidth=0)
            axis.set_ylabel(ylabel)
            panel_title(
                axis,
                chr(97 + row_index * 2 + column_index),
                f"{dataset.title()} / {metric}",
            )
            axis.grid()
            axis.set_xticks([0.5, 1.0, 2.0])
            axis.set_xlabel("Cases reviewed (% of test set)")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4)
    figure.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.11, hspace=0.44, wspace=0.25)
    return source.sort_values(["dataset", "method", "budget_pct"]).reset_index(drop=True), finish(figure, "fig07_review_budget_analysis")


def fig08_validation() -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """All-case expected-versus-observed status matrix."""

    data = load_csv(
        "FRAMEWORK_VALIDATION_CASES.csv",
        {"case_id", "mutation", "expected_status", "observed_status", "pass"},
    ).copy()
    if len(data) != 14 or data.case_id.nunique() != 14:
        raise ValueError("F08 must display all 14 unique controlled validation cases")
    status_order = list(STATUS_SHORT_LABELS)
    unknown = sorted(
        set(data.expected_status).union(data.observed_status).difference(status_order)
    )
    if unknown:
        raise ValueError(f"F08 encountered unregistered statuses: {unknown}")
    status_x = {status: index for index, status in enumerate(status_order)}
    data["expected_status_index"] = data.expected_status.map(status_x)
    data["observed_status_index"] = data.observed_status.map(status_x)
    data["expected_status_label"] = data.expected_status.map(STATUS_SHORT_LABELS)
    data["observed_status_label"] = data.observed_status.map(STATUS_SHORT_LABELS)
    data = data.sort_values("case_id", key=lambda col: col.str.extract(r"(\d+)")[0].astype(int))
    data["plot_y"] = np.arange(len(data))[::-1]

    spec = FIGURE_SPECS["fig08_claim_support_validation"]
    figure, axis = plt.subplots(figsize=(spec.width, spec.height))
    axis.scatter(
        data.expected_status_index,
        data.plot_y,
        marker="s",
        s=43,
        facecolor=WHITE,
        edgecolor=BLACK,
        linewidth=0.9,
        label="expected",
        zorder=2,
    )
    axis.scatter(
        data.observed_status_index,
        data.plot_y,
        marker="o",
        s=17,
        facecolor=BLACK,
        edgecolor=BLACK,
        linewidth=0.5,
        label="observed",
        zorder=3,
    )
    axis.set_xlim(-0.55, len(status_order) - 0.45)
    axis.set_ylim(-0.6, len(data) - 0.4)
    axis.set_xticks(
        range(len(status_order)),
        [STATUS_SHORT_LABELS[status] for status in status_order],
        rotation=48,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticks(data.plot_y, data.case_id)
    matches = int((data.expected_status == data.observed_status).sum())
    axis.set_title(f"Controlled validator suite: {matches}/{len(data)} status matches", loc="left", fontsize=PANEL_FONT_PT, pad=5)
    axis.grid(axis="both")
    axis.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="both", length=0)
    figure.subplots_adjust(left=0.14, right=0.985, top=0.91, bottom=0.28)
    return data.reset_index(drop=True), finish(figure, "fig08_claim_support_validation")


@dataclass(frozen=True)
class FigureJob:
    figure_id: str
    stem: str
    builder: Callable[[], tuple[pd.DataFrame, tuple[Path, Path]]]
    upstream: tuple[str, ...]
    filters_and_pairing: str
    seed_set: str
    statistical_summary: str
    blocked_missing_treatment: str


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = [
        FigureJob(
            "F01",
            "fig01_deployment_contract",
            fig01_contract,
            ("results/tkde_rebuild/DEPLOYMENT_CONTRACT_AXES.csv",),
            "All six frozen deployment coordinates in T,V,C,S,B,R order; formal flow and status key are code-authored from the accepted ontology.",
            "not applicable",
            "formal architecture; no empirical aggregation",
            "status labels are categorical and never plotted on a performance scale",
        ),
        FigureJob(
            "F02",
            "fig02_protocol_architecture_effects",
            fig02_protocol,
            ("results/tkde_rebuild/RB09_AUPRC_MAIN.csv",),
            "Complete 2-dataset x 3-model strict/isolated grid; paired seeds 1-10.",
            "1-10 paired",
            "means and paired deltas with deterministic 95% bootstrap intervals",
            "complete matched grid; no missing or blocked rows",
        ),
        FigureJob(
            "F03",
            "fig03_ibm_metric_scale_construction",
            fig03_ibm,
            ("results/tkde_rebuild/IBM_CELL_SUMMARY.csv",),
            "V26 baseline families only; HI/LI x Small/Medium x two protocols; 24 full10 rows.",
            "1-10 per cell",
            "AUPRC means and deterministic 95% bootstrap intervals; LogReg intervals are degenerate because the baseline is deterministic",
            "IBM Large is outside empirical scope; no blocked cell is shown as performance",
        ),
        FigureJob(
            "F04",
            "fig04_rank_decision_divergence",
            fig04_rank,
            (
                "results/tkde_rebuild/IBM_RANK_DIVERGENCE.csv",
                "results/tkde_rebuild/IBM_METRIC_RANKS.csv",
            ),
            "All 16 exact baseline/graph feasibility cells; representative reversal selected deterministically as the lowest-rho graph cell with different AUPRC/F1 winners.",
            "cell means over seeds 1-10",
            "descriptive Spearman rank correlation and exact within-cell ranks",
            "each rank comparison uses its own feasible set; Medium GINE is absent rather than ranked",
        ),
        FigureJob(
            "F05",
            "fig05_matched_ablation_effects",
            fig05_ablation,
            (
                "results/tkde_rebuild/IBM_MATCHED_ABLATION_EFFECTS.csv",
                "results/tkde_rebuild/IBM_RUNTIME_FEASIBILITY.csv",
            ),
            "Alias excluded; fixed HI/LI x two-protocol contexts averaged within seed; Small and Medium shown separately for AUPRC, AUROC, and F1.",
            "10 matched seed blocks per measured construction/metric",
            "paired mean deltas with deterministic 95% bootstrap intervals and Holm-corrected Wilcoxon status",
            "Medium GINE is exported and labeled RESOURCE_BLOCKED_T4_CUDA_OOM with all effect fields nonnumeric",
        ),
        FigureJob(
            "F06",
            "fig06_runtime_resource_pareto",
            fig06_runtime,
            ("results/tkde_rebuild/IBM_RUNTIME_FEASIBILITY.csv",),
            "Four variant panels; every measured point uses its configuration-specific runtime/performance; Pareto flags remain within variant/protocol cells.",
            "1-10 per measured cell",
            "mean AUPRC with +/-1 SD clipped to [0,1]; mean runtime with observed min-max",
            "two Medium GINE rows remain n=0 and nonnumeric; T4 OOM is shown in an off-scale status box",
        ),
        FigureJob(
            "F07",
            "fig07_review_budget_analysis",
            fig07_budget,
            ("results/tkde_rebuild/REVIEW_BUDGET_CURVES.csv",),
            "Four saved-output comparators at 0.5%, 1%, and 2% review budgets on Elliptic and DGraphFin.",
            "10 seed blocks; six contexts averaged within seed",
            "mean precision/recall with exported normal-approximation 95% intervals (1.96 x SD/sqrt(n))",
            "bounded case study only; no universal dominance encoding",
        ),
        FigureJob(
            "F08",
            "fig08_claim_support_validation",
            fig08_validation,
            ("results/tkde_rebuild/FRAMEWORK_VALIDATION_CASES.csv",),
            "All 14 controlled validation cases in case-ID order; expected and observed statuses plotted independently.",
            "case-specific; not a seed analysis",
            "14-case conformance count; no uncertainty claim",
            "resource-blocked/excluded/refuted remain categorical statuses, not numeric scores",
        ),
    ]

    provenance: list[dict[str, object]] = []
    for job in jobs:
        frame, (pdf, png) = job.builder()
        source_path = export_source(frame, job.stem)
        upstream_paths = [ROOT / path for path in job.upstream]
        for path in upstream_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        spec = FIGURE_SPECS[job.stem]
        provenance.append(
            {
                "figure_id": job.figure_id,
                "figure_file": rel(pdf),
                "png_preview": rel(png),
                "source_data_csv": rel(source_path),
                "upstream_evidence": ";".join(job.upstream),
                "upstream_sha256": ";".join(sha256(path) for path in upstream_paths),
                "generation_script": "scripts/tkde_rebuild/make_figures.py",
                "style_authority": "scripts/tkde_visual_rebuild/publication_style.py",
                "filters_and_pairing": job.filters_and_pairing,
                "seed_set": job.seed_set,
                "statistical_summary": job.statistical_summary,
                "blocked_missing_treatment": job.blocked_missing_treatment,
                "intended_use": spec.intended_use,
                "physical_width_in": f"{spec.width:.2f}",
                "physical_height_in": f"{spec.height:.2f}",
                "minimum_text_size_pt": f"{min(TICK_FONT_PT, LEGEND_FONT_PT):.1f}",
                "source_rows": len(frame),
                "sha256_source_data": sha256(source_path),
                "sha256_pdf": sha256(pdf),
                "sha256_png": sha256(png),
            }
        )
    provenance_path = OUT / "FIGURE_DATA_PROVENANCE.csv"
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    print(f"generated {len(provenance)} figures and source CSVs")
    print(f"provenance: {rel(provenance_path)}")


if __name__ == "__main__":
    main()
