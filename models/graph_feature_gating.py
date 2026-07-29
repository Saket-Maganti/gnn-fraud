"""
models/graph_feature_gating.py

Graph-vs-feature gating: *learn when to trust the graph and when to trust raw
features.*

This is **method direction 4** in the AAAI reframing. The professor asked for
"a graph-vs-feature gating mechanism where the model learns when to rely on raw
features and when to rely on graph representations." The premise from the
empirical side of the project is that a feature-only model (e.g. random forest
on the 165-d Elliptic features) can beat the best GNN under strict temporal
evaluation — but not uniformly: on some nodes / time windows the graph branch is
better. A gate that routes each node to the more reliable branch should dominate
either branch alone.

The gate consumes the **positive-class scores** of a graph model and a
feature model (both in ``[0, 1]``), plus optional per-node side information
(graph-harm score, degree, temporal instability), and produces a blended score.
It is fit on **validation** scores + labels only; ``predict`` accepts **no**
labels and raises :class:`GatingLeakageError` if any are smuggled in.

Modes
-----
* ``"feature_only"`` / ``"graph_only"`` — degenerate baselines (no fit needed).
* ``"static"`` — fixed 0.5 average of the two branches.
* ``"global_weight"`` — one scalar ``alpha`` chosen on validation F1:
  ``alpha * graph + (1 - alpha) * feature``. Learns *which branch is better on
  average* without per-node routing.
* ``"logistic"`` — a per-node logistic gate trained to predict *which branch is
  correct* from gating features (branch confidences, disagreement, harm,
  degree, instability). ``out = g * graph + (1 - g) * feature`` with
  ``g = P(trust graph | gating features)``. This is the AAAI method variant.
* ``"harm_aware"`` — rule gate driven by the harm score:
  ``w_graph = 1 - harm`` (clipped), no labels needed beyond the harm estimate.

Safety
------
``fit`` falls back to the ``static`` average when validation is too small or
single-class (``fallback_used`` flag set). ``predict`` is deterministic and
score-only. Nothing trains a GNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

try:  # sklearn is already a project dependency; degrade gracefully if absent.
    from sklearn.linear_model import LogisticRegression
    _HAVE_SKLEARN = True
except Exception:  # noqa: BLE001
    _HAVE_SKLEARN = False


VALID_MODES = ("feature_only", "graph_only", "static", "global_weight", "logistic", "harm_aware")


class GatingLeakageError(ValueError):
    """Raised when ``predict`` is handed labels (it must be score-only)."""


_FORBIDDEN_LABEL_KEYS = frozenset(
    {"y", "labels", "label", "test_labels", "test_y", "y_test", "targets", "target", "gold"}
)


def _reject_labels(kwargs: Mapping[str, object]) -> None:
    leaked = sorted(set(kwargs) & _FORBIDDEN_LABEL_KEYS)
    if leaked:
        raise GatingLeakageError(
            f"gate.predict must not receive labels; refusing {leaked}. "
            "Routing uses scores + side-information only."
        )
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")


def _binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp == 0 and (fp + fn) == 0:
        return 0.0
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _gating_features(
    graph_scores: np.ndarray,
    feature_scores: np.ndarray,
    metadata: Optional[Mapping[str, object]],
    n: int,
) -> np.ndarray:
    """Per-node gating feature matrix. Side info broadcast/aligned to length N."""
    g = graph_scores
    f = feature_scores
    cols = [
        g,                          # graph score
        f,                          # feature score
        np.abs(g - 0.5) * 2.0,      # graph confidence
        np.abs(f - 0.5) * 2.0,      # feature confidence
        np.abs(g - f),              # branch disagreement
    ]
    if metadata:
        for key in ("harm", "harm_score", "degree", "instability", "temporal_instability"):
            if key in metadata:
                val = np.asarray(metadata[key], dtype=float)
                if val.ndim == 0:
                    val = np.full(n, float(val))
                cols.append(val.reshape(-1)[:n])
    return np.column_stack(cols)


@dataclass
class GraphFeatureGate:
    """Validation-fit, score-only graph-vs-feature router.

    Construct with a ``mode`` (see module docstring), call :meth:`fit` on
    validation scores+labels, then :meth:`predict` on test scores.
    """

    mode: str = "logistic"
    min_validation: int = 12
    seed: int = 0

    # learned state (populated by fit)
    alpha: float = 0.5
    fallback_used: bool = False
    _clf: object = field(default=None, repr=False)
    _fitted: bool = False
    mean_gate: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown mode {self.mode!r}; choose from {VALID_MODES}")

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        validation_graph_scores: np.ndarray,
        validation_feature_scores: np.ndarray,
        validation_labels: np.ndarray,
        *,
        validation_metadata: Optional[Mapping[str, object]] = None,
    ) -> "GraphFeatureGate":
        g = np.asarray(validation_graph_scores, dtype=float).reshape(-1)
        f = np.asarray(validation_feature_scores, dtype=float).reshape(-1)
        y = (np.asarray(validation_labels).reshape(-1) == 1).astype(int)
        if g.shape != f.shape or g.shape[0] != y.shape[0]:
            raise ValueError("validation score/label length mismatch")

        too_small = y.shape[0] < self.min_validation
        single_class = len(np.unique(y)) < 2
        if self.mode in ("feature_only", "graph_only", "static", "harm_aware"):
            self._fitted = True
            return self  # no labels needed for these modes
        if too_small or single_class:
            self.fallback_used = True
            self.mode = "static"
            self._fitted = True
            return self

        if self.mode == "global_weight":
            self.alpha = self._fit_global_weight(g, f, y)
        elif self.mode == "logistic":
            self._fit_logistic(g, f, y, validation_metadata)
        self._fitted = True
        return self

    def _fit_global_weight(self, g: np.ndarray, f: np.ndarray, y: np.ndarray) -> float:
        best_a, best_f1 = 0.5, -1.0
        for a in np.linspace(0.0, 1.0, 21):
            blended = a * g + (1.0 - a) * f
            f1 = _binary_f1(y, (blended >= 0.5).astype(int))
            if f1 > best_f1:
                best_f1, best_a = f1, float(a)
        return best_a

    def _fit_logistic(self, g, f, y, metadata) -> None:
        # Target: did the graph branch get this node "more right" than features?
        graph_err = np.abs(g - y)
        feat_err = np.abs(f - y)
        trust_graph = (graph_err < feat_err).astype(int)
        # weight by how decisive the difference is; drop ties
        margin = np.abs(graph_err - feat_err)
        keep = margin > 1e-6
        if keep.sum() < self.min_validation or len(np.unique(trust_graph[keep])) < 2 or not _HAVE_SKLEARN:
            # fall back to a global weight if routing target is degenerate
            self.alpha = self._fit_global_weight(g, f, y)
            self.mode = "global_weight"
            self.fallback_used = True
            return
        Z = _gating_features(g, f, metadata, n=g.shape[0])[keep]
        clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        clf.fit(Z, trust_graph[keep], sample_weight=margin[keep])
        self._clf = clf

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(
        self,
        graph_scores: np.ndarray,
        feature_scores: np.ndarray,
        *,
        metadata: Optional[Mapping[str, object]] = None,
        **forbidden: object,
    ) -> np.ndarray:
        """Return blended positive-class scores. Score-only (no labels)."""
        _reject_labels(forbidden)
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        g = np.asarray(graph_scores, dtype=float).reshape(-1)
        f = np.asarray(feature_scores, dtype=float).reshape(-1)
        if g.shape != f.shape:
            raise ValueError("graph/feature score length mismatch")

        if self.mode == "feature_only":
            self.mean_gate = 0.0
            return f
        if self.mode == "graph_only":
            self.mean_gate = 1.0
            return g
        if self.mode == "static":
            self.mean_gate = 0.5
            return 0.5 * g + 0.5 * f
        if self.mode == "global_weight":
            self.mean_gate = self.alpha
            return self.alpha * g + (1.0 - self.alpha) * f
        if self.mode == "harm_aware":
            gate = self._harm_aware_gate(metadata, g.shape[0])
            self.mean_gate = float(np.mean(gate))
            return gate * g + (1.0 - gate) * f
        if self.mode == "logistic":
            if self._clf is None:
                self.mean_gate = 0.5
                return 0.5 * g + 0.5 * f
            Z = _gating_features(g, f, metadata, n=g.shape[0])
            gate = self._clf.predict_proba(Z)[:, 1]
            self.mean_gate = float(np.mean(gate))
            return gate * g + (1.0 - gate) * f
        raise ValueError(f"unknown mode {self.mode!r}")

    def _harm_aware_gate(self, metadata: Optional[Mapping[str, object]], n: int) -> np.ndarray:
        if not metadata or not any(k in metadata for k in ("harm", "harm_score")):
            return np.full(n, 0.5)
        harm = metadata.get("harm", metadata.get("harm_score"))
        harm_arr = np.asarray(harm, dtype=float)
        if harm_arr.ndim == 0:
            harm_arr = np.full(n, float(harm_arr))
        return np.clip(1.0 - harm_arr.reshape(-1)[:n], 0.0, 1.0)

    # ── convenience ────────────────────────────────────────────────────────────

    def predict_label(
        self,
        graph_scores: np.ndarray,
        feature_scores: np.ndarray,
        *,
        threshold: float = 0.5,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> np.ndarray:
        return (self.predict(graph_scores, feature_scores, metadata=metadata) >= threshold).astype(int)
