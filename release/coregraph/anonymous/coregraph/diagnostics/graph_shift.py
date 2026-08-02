"""Label-free graph topology and perturbation diagnostics."""

from __future__ import annotations

import numpy as np

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="graph_shift",
    required_inputs=("visible_degrees", "visible_edge_count", "node_count"),
    target_labels_required=False,
    source_fitting="SOURCE_DEGREE_HISTOGRAM_ONLY",
    missing_semantics=MissingSemantics.ZERO_WITH_MASK,
    computational_cost="O(nodes + visible_edges)",
)


def graph_shift_diagnostics(
    degrees: np.ndarray,
    *,
    edge_count: int,
    source_degree_histogram: np.ndarray | None = None,
    perturbation_scores: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(degrees, dtype=float).reshape(-1)
    if np.any(values < 0) or edge_count < 0:
        raise ValueError("degrees and edge count must be non-negative")
    n = len(values)
    density = 0.0 if n < 2 else float(edge_count / (n * (n - 1)))
    output = {
        "graph_density": density,
        "isolated_node_fraction": float(np.mean(values == 0)) if n else 1.0,
        "mean_degree": float(values.mean()) if n else 0.0,
        "neighbourhood_truncation": float(np.mean(values == values.max())) if n else 1.0,
        "edge_perturbation_sensitivity": (
            float(np.mean(np.abs(perturbation_scores)))
            if perturbation_scores is not None
            else float("nan")
        ),
    }
    if source_degree_histogram is None:
        output["degree_distribution_shift"] = float("nan")
    else:
        source = np.asarray(source_degree_histogram, dtype=float)
        if source.ndim != 1 or source.sum() <= 0:
            raise ValueError("source degree histogram must be a positive vector")
        bins = len(source)
        observed = np.bincount(np.clip(values.astype(int), 0, bins - 1), minlength=bins)
        p = observed / max(observed.sum(), 1)
        q = source / source.sum()
        output["degree_distribution_shift"] = float(0.5 * np.abs(p - q).sum())
    return output
