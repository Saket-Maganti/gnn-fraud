"""Source-fitted feature and representation drift scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coregraph.diagnostics.specification import DiagnosticDeclaration, MissingSemantics


DECLARATION = DiagnosticDeclaration(
    name="feature_shift",
    required_inputs=("label_free_features",),
    target_labels_required=False,
    source_fitting="SOURCE_TRAIN_FEATURE_MOMENTS_ONLY",
    missing_semantics=MissingSemantics.NAN_WITH_MASK,
    computational_cost="O(instances * features)",
)


@dataclass(frozen=True)
class SourceFeatureReference:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, source_features: np.ndarray) -> "SourceFeatureReference":
        values = np.asarray(source_features, dtype=float)
        if values.ndim != 2 or not len(values):
            raise ValueError("source features must be a non-empty matrix")
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean, scale)

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("feature matrix does not match source reference")
        return np.nanmean(np.abs((values - self.mean) / self.scale), axis=1)
