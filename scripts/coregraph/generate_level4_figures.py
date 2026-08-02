#!/usr/bin/env python3
"""Generate non-empirical Level-4 figures and empty result templates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_iclr" / "figures"
BLUE = "#2457A6"
TEAL = "#168B8C"
ORANGE = "#D7792D"
GREY = "#5B6573"
LIGHT = "#F3F6FA"


def _canvas(width: float = 8.4, height: float = 3.4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box(ax, xy, width, height, text, *, color=BLUE, size=9, face=LIGHT):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=color,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=size, color="#172033")


def _arrow(ax, start, end, *, color=GREY, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, linewidth=1.2, color=color))


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"Creator": "Anonymous CoReGraph build"})
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", metadata={"Software": "Anonymous CoReGraph build"})
    plt.close(fig)


def problem_overview():
    fig, ax = _canvas()
    _box(ax, (0.03, 0.58), 0.21, 0.25, "Labelled source\ncontracts", color=TEAL)
    _box(ax, (0.03, 0.16), 0.21, 0.25, "Unseen target\ncomposition", color=ORANGE)
    _box(ax, (0.38, 0.37), 0.24, 0.30, "Contract-aware\nfeasible router", color=BLUE)
    _box(ax, (0.76, 0.58), 0.20, 0.25, "Expert\ndistribution", color=TEAL)
    _box(ax, (0.76, 0.16), 0.20, 0.25, "Abstain /\nfallback", color=ORANGE)
    for start, end in (((0.24, 0.70), (0.38, 0.57)), ((0.24, 0.29), (0.38, 0.47)), ((0.62, 0.55), (0.76, 0.70)), ((0.62, 0.45), (0.76, 0.29))):
        _arrow(ax, start, end)
    ax.text(0.5, 0.05, "Target labels are unavailable during fitting and deployment", ha="center", fontsize=9, color=GREY)
    _save(fig, "01_problem_overview")


def factorisation():
    fig, ax = _canvas(height=2.8)
    labels = ("time", "visibility", "construction", "selection", "budget", "resource")
    colors = (TEAL, BLUE, ORANGE, TEAL, BLUE, ORANGE)
    for index, (label, color) in enumerate(zip(labels, colors)):
        left = 0.02 + index * 0.16
        _box(ax, (left, 0.54), 0.14, 0.23, label, color=color, size=8)
        _arrow(ax, (left + 0.07, 0.54), (0.50, 0.28), color=color)
    _box(ax, (0.37, 0.08), 0.26, 0.20, "Factorised embedding\n+ bounded interactions", color=BLUE)
    ax.text(0.5, 0.90, "Each axis carries value, state, confidence, and missing/unseen semantics", ha="center", fontsize=9)
    _save(fig, "02_contract_factorisation")


def architecture():
    fig, ax = _canvas(height=3.7)
    _box(ax, (0.02, 0.62), 0.18, 0.20, "Contract\nencoder", color=BLUE)
    _box(ax, (0.02, 0.20), 0.18, 0.20, "Expert-aware\ndiagnostics", color=TEAL)
    _box(ax, (0.29, 0.47), 0.18, 0.22, "Contract\nprior", color=BLUE)
    _box(ax, (0.29, 0.13), 0.18, 0.22, "Bounded instance\ncorrection", color=TEAL)
    _box(ax, (0.56, 0.37), 0.18, 0.24, "Resource mask\nthen normalize", color=ORANGE)
    _box(ax, (0.82, 0.59), 0.16, 0.19, "Routed\nscore", color=BLUE)
    _box(ax, (0.82, 0.20), 0.16, 0.19, "Controlled\nabstention", color=ORANGE)
    for start, end in (((0.20, 0.72), (0.29, 0.60)), ((0.20, 0.30), (0.29, 0.24)), ((0.47, 0.58), (0.56, 0.51)), ((0.47, 0.24), (0.56, 0.45)), ((0.74, 0.51), (0.82, 0.69)), ((0.74, 0.43), (0.82, 0.29))):
        _arrow(ax, start, end)
    ax.text(0.50, 0.91, "CoReGraph: hierarchical routing over heterogeneous pretrained experts", ha="center", fontsize=10, weight="bold")
    _save(fig, "03_coregraph_architecture")


def flow():
    fig, ax = _canvas(height=3.1)
    labels = ("Source contracts", "Fit encoder / diagnostics", "Freeze router + threshold", "Target contract", "Mask / route / abstain")
    for index, label in enumerate(labels):
        left = 0.015 + index * 0.197
        _box(ax, (left, 0.40), 0.17, 0.24, label, color=BLUE if index < 3 else TEAL, size=8)
        if index < len(labels) - 1:
            _arrow(ax, (left + 0.17, 0.52), (left + 0.197, 0.52))
    ax.text(0.30, 0.77, "TRAINING (source labels allowed)", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(0.80, 0.77, "INFERENCE (target labels forbidden)", ha="center", fontsize=9, color=TEAL, weight="bold")
    ax.plot([0.60, 0.60], [0.22, 0.83], linestyle="--", color=GREY, linewidth=1)
    _save(fig, "04_training_inference_flow")


def scenarios():
    fig, ax = _canvas(height=3.2)
    _box(ax, (0.03, 0.65), 0.17, 0.20, "Protocol A\n3 experts", color=TEAL)
    _box(ax, (0.03, 0.38), 0.17, 0.20, "Protocol B\n3 experts", color=TEAL)
    _box(ax, (0.03, 0.11), 0.17, 0.20, "Protocol C\n3 experts", color=ORANGE)
    _box(ax, (0.36, 0.34), 0.24, 0.31, "One dataset + seed\n6 source bindings\n3 target bindings", color=BLUE)
    _box(ax, (0.75, 0.57), 0.21, 0.22, "Source fit\ntrain + validation", color=TEAL)
    _box(ax, (0.75, 0.17), 0.21, 0.22, "Target score\nknown-label test only", color=ORANGE)
    for y in (0.75, 0.48, 0.21):
        _arrow(ax, (0.20, y), (0.36, 0.50))
    _arrow(ax, (0.60, 0.55), (0.75, 0.68))
    _arrow(ax, (0.60, 0.44), (0.75, 0.28))
    ax.text(0.50, 0.92, "Role-neutral artifacts; roles exist only inside a scenario", ha="center", fontsize=10, weight="bold")
    _save(fig, "05_scenario_construction")


def theory():
    fig, ax = _canvas(height=3.2)
    _box(ax, (0.03, 0.58), 0.22, 0.23, "Contract 1\nExpert A preferred", color=TEAL)
    _box(ax, (0.03, 0.18), 0.22, 0.23, "Contract 2\nExpert B preferred", color=ORANGE)
    _box(ax, (0.38, 0.38), 0.22, 0.25, "Fixed mixture\npositive worst-contract\nregret", color=GREY)
    _box(ax, (0.74, 0.58), 0.23, 0.23, "Factorised router\nuses contract axes", color=BLUE)
    _box(ax, (0.74, 0.18), 0.23, 0.23, "Failure case\narbitrary interaction", color=ORANGE)
    for start, end in (((0.25, 0.69), (0.38, 0.55)), ((0.25, 0.29), (0.38, 0.45)), ((0.60, 0.54), (0.74, 0.69)), ((0.60, 0.45), (0.74, 0.29))):
        _arrow(ax, start, end)
    _save(fig, "06_theory_intuition")


def masking():
    fig, ax = _canvas(height=3.0)
    _box(ax, (0.03, 0.56), 0.18, 0.22, "Router logits", color=BLUE)
    _box(ax, (0.29, 0.56), 0.18, 0.22, "Availability +\nmemory + latency", color=ORANGE)
    _box(ax, (0.55, 0.56), 0.18, 0.22, "Feasible softmax\nunavailable = zero", color=TEAL)
    _box(ax, (0.80, 0.67), 0.17, 0.18, "Unit feasible\nmass", color=BLUE)
    _box(ax, (0.80, 0.27), 0.17, 0.18, "Empty set:\nabstain", color=ORANGE)
    _arrow(ax, (0.21, 0.67), (0.29, 0.67))
    _arrow(ax, (0.47, 0.67), (0.55, 0.67))
    _arrow(ax, (0.73, 0.70), (0.80, 0.76))
    _arrow(ax, (0.73, 0.61), (0.80, 0.36))
    ax.text(0.50, 0.15, "Constraints are enforced before selection; masks are recorded in manifests", ha="center", fontsize=9)
    _save(fig, "07_resource_feasible_masking")


def taxonomy():
    fig, ax = _canvas(height=4.0)
    groups = {
        "Graph": ("edge deletion", "degree shift", "homophily", "heterophily", "truncation"),
        "Feature / time": ("feature corruption", "concept drift", "delayed labels"),
        "Contract": ("noisy metadata", "missing axis", "correlated shift"),
        "Operations": ("unavailable expert", "budget contraction", "dynamic resource", "all experts degrade"),
    }
    positions = ((0.03, 0.55), (0.52, 0.55), (0.03, 0.12), (0.52, 0.12))
    colors = (BLUE, TEAL, ORANGE, GREY)
    for (title, items), position, color in zip(groups.items(), positions, colors):
        _box(ax, position, 0.44, 0.31, title + "\n" + " · ".join(items), color=color, size=8)
    ax.text(0.5, 0.94, "Fifteen controlled mechanisms isolate routing assumptions", ha="center", fontsize=10, weight="bold")
    _save(fig, "08_synthetic_mechanism_taxonomy")


def result_template(name: str, title: str, x_label: str, y_label: str):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#D7DDE6", linewidth=0.6)
    ax.text(0.5, 0.52, "PENDING VALIDATED RUNS", transform=ax.transAxes, ha="center", va="center", fontsize=13, color=GREY, weight="bold")
    ax.text(0.5, 0.39, "No curve or value is populated", transform=ax.transAxes, ha="center", fontsize=9, color=GREY)
    _save(fig, name)


def main() -> int:
    problem_overview()
    factorisation()
    architecture()
    flow()
    scenarios()
    theory()
    masking()
    taxonomy()
    templates = (
        ("template_regret_comparison", "Contract regret comparison", "Method", "Regret"),
        ("template_worst_contract", "Worst-contract profile", "Contract", "Regret"),
        ("template_budget_frontier", "Budget-performance frontier", "Review budget", "Utility"),
        ("template_routing_heatmap", "Routing policy", "Expert", "Contract"),
        ("template_counterfactual", "Counterfactual contract response", "Intervention", "Routing mass"),
        ("template_failure_taxonomy", "Failure incidence", "Failure label", "Count"),
        ("template_latent_contract", "Latent contract discovery", "Latent factor 1", "Latent factor 2"),
    )
    for template in templates:
        result_template(*template)
    print(f"generated {8 + len(templates)} Level-4 figure families in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
