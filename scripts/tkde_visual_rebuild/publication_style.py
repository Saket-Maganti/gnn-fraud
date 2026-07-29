#!/usr/bin/env python3
"""Shared publication style for the TKDE visual-rebuild figures.

The values in this module are physical-output requirements, not plotting
suggestions.  Figure generators save at the declared IEEE column widths with
``bbox_inches=None`` so that a later rebuild cannot silently rescale text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl


ONE_COLUMN_IN = 3.45
TWO_COLUMN_IN = 7.16

BASE_FONT_PT = 8.0
TICK_FONT_PT = 7.5
TITLE_FONT_PT = 8.2
LEGEND_FONT_PT = 7.5
PANEL_FONT_PT = 8.2
MIN_LINE_PT = 0.9
MIN_MARKER_PT = 4.5
ERROR_CAP_PT = 2.0

WHITE = "#FFFFFF"
BLACK = "#161616"
DARK_GRAY = "#4A4A4A"
MID_GRAY = "#777777"
LIGHT_GRAY = "#B8B8B8"
PALE_GRAY = "#E6E6E6"
PANEL_GRAY = "#F4F4F4"
ACCENT_BLUE = "#0072B2"
ACCENT_VERMILION = "#D55E00"
ACCENT_TEAL = "#009E73"


@dataclass(frozen=True)
class FigureSpec:
    """Exact physical geometry and intended publication placement."""

    width: float
    height: float
    intended_use: str


FIGURE_SPECS = {
    "fig01_deployment_contract": FigureSpec(TWO_COLUMN_IN, 1.92, "main/full-width"),
    "fig02_protocol_architecture_effects": FigureSpec(TWO_COLUMN_IN, 2.82, "main/full-width"),
    "fig03_ibm_metric_scale_construction": FigureSpec(ONE_COLUMN_IN, 3.72, "main/one-column"),
    "fig04_rank_decision_divergence": FigureSpec(ONE_COLUMN_IN, 4.28, "main/one-column"),
    "fig05_matched_ablation_effects": FigureSpec(TWO_COLUMN_IN, 4.42, "main/full-width"),
    "fig06_runtime_resource_pareto": FigureSpec(TWO_COLUMN_IN, 4.62, "main/full-width"),
    "fig07_review_budget_analysis": FigureSpec(TWO_COLUMN_IN, 4.18, "supplement/full-width"),
    "fig08_claim_support_validation": FigureSpec(ONE_COLUMN_IN, 3.42, "main/one-column"),
}


MODEL_STYLES = {
    "gcn": {"color": ACCENT_BLUE, "marker": "o", "linestyle": "-"},
    "mlp": {"color": MID_GRAY, "marker": "s", "linestyle": "--"},
    "sage": {"color": ACCENT_VERMILION, "marker": "^", "linestyle": "-"},
}

BASELINE_STYLES = {
    "HistGB": {"color": BLACK, "marker": "o", "facecolor": BLACK},
    "LogReg": {"color": DARK_GRAY, "marker": "D", "facecolor": WHITE},
    "SAGE-h32": {"color": MID_GRAY, "marker": "^", "facecolor": WHITE},
}

POLICY_STYLES = {
    "feature_only": {"color": MID_GRAY, "marker": "s", "linestyle": ":"},
    "best_val_branch": {"color": BLACK, "marker": "^", "linestyle": "--"},
    "simple_average": {"color": ACCENT_BLUE, "marker": "o", "linestyle": "-"},
    "graphsafe_conservative": {
        "color": ACCENT_VERMILION,
        "marker": "D",
        "linestyle": "-.",
    },
}

CONFIG_LABELS = {
    "edge_aware_graphsage_h64": "Ref",
    "edge_aware_graphsage_h64_no_edge_features": "No edge",
    "edge_aware_graphsage_h64_shuffled_edge_features": "Shuffle",
    "edge_aware_graphsage_h64_degree_only": "Degree",
    "degree_capped_bipartite": "Degree cap",
    "recent_window_only_graph": "Recent",
    "gine_light_h64": "GINE",
    "hist_gradient_boosting_edge_features": "HistGB",
    "logistic_regression_edge_features": "LogReg",
    "graphsage_edge_minibatch_h32": "SAGE-h32",
}

STATUS_SHORT_LABELS = {
    "SUPPORTED": "Supported",
    "BLOCKED_INCOMPLETE_SCOPE": "Scope\nblocked",
    "BLOCKED_INCOMPLETE_SEEDS": "Seeds\nblocked",
    "BLOCKED_MISSING_PREDICTIONS": "Pred.\nblocked",
    "RESOURCE_BLOCKED": "Resource\nblocked",
    "EXCLUDED_INTEGRITY": "Integrity\nexcluded",
    "EXCLUDED_CONSTRUCT_INVALID": "Construct\nexcluded",
    "REFUTED_IN_SCOPE": "Refuted",
}


def configure_matplotlib() -> None:
    """Install the deterministic, grayscale-first Matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": BASE_FONT_PT,
            "axes.labelsize": BASE_FONT_PT,
            "axes.titlesize": TITLE_FONT_PT,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.7,
            "axes.edgecolor": DARK_GRAY,
            "axes.labelcolor": BLACK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": TICK_FONT_PT,
            "ytick.labelsize": TICK_FONT_PT,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": LEGEND_FONT_PT,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.45,
            "lines.linewidth": MIN_LINE_PT,
            "lines.markersize": MIN_MARKER_PT,
            "patch.linewidth": 0.7,
            "grid.color": PALE_GRAY,
            "grid.linewidth": 0.55,
            "grid.alpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
        }
    )


def panel_title(axis, label: str, title: str) -> None:
    """Apply a uniform upper-left panel title."""

    axis.set_title(f"({label}) {title}", loc="left", fontsize=PANEL_FONT_PT, pad=4.0)


def save_exact(figure, stem: str, output_dir: Path) -> tuple[Path, Path]:
    """Save deterministic PDF and PNG endpoints at the registered dimensions."""

    if stem not in FIGURE_SPECS:
        raise KeyError(f"unregistered figure stem: {stem}")
    spec = FIGURE_SPECS[stem]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(spec.width, spec.height, forward=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    common = {"bbox_inches": None, "facecolor": WHITE, "edgecolor": WHITE}
    figure.savefig(
        pdf_path,
        format="pdf",
        metadata={
            "Title": stem,
            "Author": "FraudShiftBench",
            "Subject": "TKDE publication figure",
            "Keywords": "FraudShiftBench TKDE",
            "Creator": "scripts/tkde_rebuild/make_figures.py",
            "Producer": "Matplotlib PDF backend",
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Software": "FraudShiftBench TKDE visual rebuild"},
        **common,
    )
    return pdf_path, png_path


configure_matplotlib()

