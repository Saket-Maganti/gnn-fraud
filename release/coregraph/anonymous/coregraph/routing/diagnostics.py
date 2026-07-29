"""Leakage metadata and computations for router diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from coregraph.data.graph_views import GraphView


class DiagnosticLevel(str, Enum):
    EXAMPLE = "example"
    CONTRACT = "contract"


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    level: DiagnosticLevel
    graph_view: str
    labels_required: bool
    target_access_required: bool
    leakage_restriction: str


DIAGNOSTIC_REGISTRY: dict[str, DiagnosticSpec] = {
    "expert_logits": DiagnosticSpec("expert_logits", DiagnosticLevel.EXAMPLE, "none", False, False, "scores from admissible expert inference only"),
    "confidence": DiagnosticSpec("confidence", DiagnosticLevel.EXAMPLE, "none", False, False, "no calibration on target labels"),
    "entropy": DiagnosticSpec("entropy", DiagnosticLevel.EXAMPLE, "none", False, False, "probabilities only"),
    "score_disagreement": DiagnosticSpec("score_disagreement", DiagnosticLevel.EXAMPLE, "none", False, False, "aligned prediction IDs required"),
    "rank_disagreement": DiagnosticSpec("rank_disagreement", DiagnosticLevel.EXAMPLE, "batch", False, False, "batch composition must be declared"),
    "degree": DiagnosticSpec("degree", DiagnosticLevel.EXAMPLE, "current_contract", False, False, "admissible visible edges only"),
    "local_degree_shift": DiagnosticSpec("local_degree_shift", DiagnosticLevel.EXAMPLE, "source_and_target", False, True, "TTA or declared label-free target covariates only"),
    "feature_missingness": DiagnosticSpec("feature_missingness", DiagnosticLevel.EXAMPLE, "none", False, False, "covariates only"),
    "edge_attribute_missingness": DiagnosticSpec("edge_attribute_missingness", DiagnosticLevel.CONTRACT, "current_contract", False, False, "visible attributes only"),
    "temporal_instability": DiagnosticSpec("temporal_instability", DiagnosticLevel.EXAMPLE, "current_contract", False, False, "past edge timestamps only"),
    "graph_density": DiagnosticSpec("graph_density", DiagnosticLevel.CONTRACT, "current_contract", False, False, "visible graph only"),
    "component_isolation": DiagnosticSpec("component_isolation", DiagnosticLevel.EXAMPLE, "current_contract", False, False, "visible graph only"),
    "score_distribution_drift": DiagnosticSpec("score_distribution_drift", DiagnosticLevel.CONTRACT, "source_and_target", False, True, "unlabelled scores only"),
    "predicted_prevalence": DiagnosticSpec("predicted_prevalence", DiagnosticLevel.CONTRACT, "none", False, True, "must not use target labels"),
    "expert_resource_cost": DiagnosticSpec("expert_resource_cost", DiagnosticLevel.CONTRACT, "none", False, False, "declared estimate or measurement"),
    "expert_availability": DiagnosticSpec("expert_availability", DiagnosticLevel.CONTRACT, "none", False, False, "resource/task contract only"),
    "observed_target_error": DiagnosticSpec(
        "observed_target_error",
        DiagnosticLevel.EXAMPLE,
        "none",
        True,
        True,
        "offline evaluation only; prohibited as a label-free router input",
    ),
    "target_label_prevalence": DiagnosticSpec(
        "target_label_prevalence",
        DiagnosticLevel.CONTRACT,
        "none",
        True,
        True,
        "offline evaluation only; prohibited as a label-free router input",
    ),
}


def score_diagnostics(expert_scores: np.ndarray) -> dict[str, np.ndarray]:
    scores = np.asarray(expert_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("expert_scores must have shape [examples, experts]")
    clipped = np.clip(scores, 1e-8, 1 - 1e-8)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    return {
        "confidence": np.abs(scores - 0.5) * 2,
        "entropy": entropy,
        "score_disagreement": np.std(scores, axis=1),
        "predicted_prevalence": np.asarray([scores.mean()]),
    }


def graph_diagnostics(view: GraphView) -> dict[str, np.ndarray | float]:
    n = len(view.visible_node_ids)
    degree = np.zeros(n, dtype=float)
    index = {int(node): position for position, node in enumerate(view.visible_node_ids)}
    for endpoint in view.edge_index.reshape(-1):
        degree[index[int(endpoint)]] += 1
    density = 0.0 if n < 2 else float(view.edge_count / (n * (n - 1)))
    return {
        "degree": degree,
        "component_isolation": (degree == 0).astype(float),
        "graph_density": density,
        "edge_attribute_missingness": float(view.edge_attributes is None),
    }


def validate_target_diagnostics(names: Sequence[str], *, target_access_allowed: bool) -> None:
    unknown = sorted(set(names) - set(DIAGNOSTIC_REGISTRY))
    if unknown:
        raise ValueError(f"unknown diagnostics: {unknown}")
    for name in names:
        spec = DIAGNOSTIC_REGISTRY[name]
        if spec.labels_required:
            raise ValueError(f"target diagnostic {name} requires labels")
        if spec.target_access_required and not target_access_allowed:
            raise ValueError(f"target diagnostic {name} requires declared target access")
