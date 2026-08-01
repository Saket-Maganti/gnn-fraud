"""Calibration proxy fitted exclusively on source validation labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="source_calibration_proxy",
    required_inputs=("source_validation_scores", "source_validation_labels"),
    target_labels_required=False,
    source_fitting="SOURCE_VALIDATION_LABELS_ONLY",
    missing_semantics=MissingSemantics.NAN_WITH_MASK,
    computational_cost="O(source_validation + target_instances)",
)


@dataclass(frozen=True)
class SourceCalibrationProxy:
    edges: np.ndarray
    empirical_rate: np.ndarray

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        bins: int = 10,
    ) -> "SourceCalibrationProxy":
        scores = np.asarray(probabilities, dtype=float).reshape(-1)
        targets = np.asarray(labels, dtype=int).reshape(-1)
        if scores.shape != targets.shape or bins < 2:
            raise ValueError("source calibration inputs must align and bins >= 2")
        if np.any((scores < 0) | (scores > 1)) or not set(np.unique(targets)).issubset({0, 1}):
            raise ValueError("calibration inputs require probabilities and binary labels")
        edges = np.linspace(0.0, 1.0, bins + 1)
        index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
        global_rate = float(targets.mean())
        rates = np.asarray(
            [targets[index == position].mean() if np.any(index == position) else global_rate for position in range(bins)]
        )
        return cls(edges, rates)

    def score(self, target_probabilities: np.ndarray) -> np.ndarray:
        scores = np.asarray(target_probabilities, dtype=float)
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("target calibration proxy accepts probabilities only")
        index = np.clip(np.digitize(scores, self.edges[1:-1]), 0, len(self.empirical_rate) - 1)
        return np.abs(scores - self.empirical_rate[index])
