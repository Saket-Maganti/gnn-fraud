#!/usr/bin/env python3
"""Generate schematic assets and result-gated TeX placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper_iclr/figures"
GENERATED = ROOT / "paper_iclr/generated"


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.2, 2.5))
    axis.axis("off")
    labels = ["Six-axis contract", "Feasible experts", "CoReRouter", "Score / abstain"]
    x = np.linspace(0.08, 0.92, len(labels))
    for index, (position, label) in enumerate(zip(x, labels, strict=True)):
        axis.text(
            position,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "fc": "#e8f1f8", "ec": "#355c7d"},
        )
        if index:
            axis.annotate("", (position - 0.11, 0.5), (x[index - 1] + 0.11, 0.5),
                          arrowprops={"arrowstyle": "->", "color": "#355c7d"})
    save(fig, "figure_1_system_schematic")

    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    axis.set_title("Toy rank crossing (not an empirical result)")
    axis.plot([0, 1], [0.2, 0.8], marker="o", label="expert A risk")
    axis.plot([0, 1], [0.8, 0.2], marker="o", label="expert B risk")
    axis.set_xticks([0, 1], ["contract 1", "contract 2"])
    axis.set_ylabel("illustrative risk")
    axis.legend(frameon=False)
    save(fig, "figure_2_fixed_mixture_counterexample")

    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    matrix = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(3), ["feature", "graph", "temporal"])
    axis.set_yticks(range(4), ["CPU", "missing graph", "event stream", "T4"])
    axis.set_title("Illustrative expert feasibility mask")
    save(fig, "figure_3_resource_mask")

    placeholder = (
        "% Generated pre-run placeholder. Do not replace manually.\n"
        "\\begin{tabular}{ll}\\toprule\n"
        "Asset & Status \\\\\\midrule\n"
        "Empirical results & \\texttt{TABLE\\_PENDING\\_RUNS} \\\\\n"
        "\\bottomrule\\end{tabular}\n"
    )
    for name in ("main_results", "ablation_results", "resource_results", "calibration_results"):
        (GENERATED / f"{name}.tex").write_text(placeholder, encoding="utf-8")
    report = {
        "schema": "coregraph_paper_assets_v1",
        "schematic_figures": 3,
        "empirical_tables": "TABLE_PENDING_RUNS",
        "invented_results": False,
    }
    (ROOT / "results/coregraph_build/PAPER_ASSET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
