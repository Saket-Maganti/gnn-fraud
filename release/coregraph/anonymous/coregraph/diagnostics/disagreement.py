"""Expert disagreement and ensemble-variance diagnostics."""

from __future__ import annotations

import numpy as np

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="expert_disagreement",
    required_inputs=("aligned_expert_scores",),
    target_labels_required=False,
    source_fitting="NONE",
    missing_semantics=MissingSemantics.NAN_WITH_MASK,
    computational_cost="O(instances * experts^2)",
)


def disagreement_diagnostics(scores: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("disagreement requires at least two aligned experts")
    differences = np.abs(values[:, :, None] - values[:, None, :])
    upper = np.triu_indices(values.shape[1], k=1)
    return {
        "pairwise_disagreement": differences[:, upper[0], upper[1]],
        "ensemble_variance": np.var(values, axis=1),
        "mean_absolute_disagreement": differences[:, upper[0], upper[1]].mean(axis=1),
    }
