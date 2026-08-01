"""Per-expert confidence, entropy, margin, and logit-norm diagnostics."""

from __future__ import annotations

import numpy as np

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="confidence_bundle",
    required_inputs=("expert_probabilities",),
    target_labels_required=False,
    source_fitting="NONE",
    missing_semantics=MissingSemantics.NAN_WITH_MASK,
    computational_cost="O(instances * experts)",
)


def confidence_diagnostics(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must have shape [instances,experts] in [0,1]")
    clipped = np.clip(values, 1e-8, 1 - 1e-8)
    logits = np.log(clipped / (1 - clipped))
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    return {
        "max_probability": np.maximum(values, 1 - values),
        "entropy": entropy,
        "margin": np.abs(values - 0.5) * 2,
        "logit_norm": np.abs(logits),
    }
