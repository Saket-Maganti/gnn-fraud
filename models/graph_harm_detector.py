"""
models/graph_harm_detector.py

Graph-harm diagnostics: *when does message passing become a liability?*

This is **method direction 3** in the AAAI reframing. The professor asked for
"a method for detecting when graph structure becomes harmful — neighbourhood
label/feature drift, homophily collapse, or temporal instability — before
deciding whether to trust message passing." A graph helps a GNN when a node's
neighbourhood is informative about its label (homophily, smooth features). It
*hurts* when neighbours carry the opposite label or unrelated features, because
message passing then averages in noise. Under temporal shift the neighbourhood
can drift from helpful to harmful between train and deploy time.

This module computes leakage-safe diagnostics from the graph, node features,
node time-buckets, and **train/validation labels only** (never test labels),
and folds them into a single ``harm_score`` in ``[0, 1]`` — higher means the
structure is more likely to hurt message passing, so a deployment system should
down-weight the graph branch (see :mod:`models.graph_feature_gating`).

Diagnostics
-----------
* **edge homophily** — fraction of labelled–labelled edges joining same-class
  nodes (Zhu et al. 2020). Homophily collapse → harm.
* **neighbour label mixing** — mean fraction of a labelled node's labelled
  neighbours carrying a *different* class. High mixing → harm.
* **feature neighbour drift** — mean edge feature distance relative to the
  distance between random node pairs. ``~0`` when neighbours look alike
  (smooth, message passing helps), ``~1`` when neighbours are as dissimilar as
  random pairs (message passing injects noise).
* **temporal neighbourhood instability** — mean absolute time-bucket gap along
  edges relative to the timeline span. Edges that bridge distant times indicate
  the neighbourhood a node aggregates was built in a different regime → harm
  under shift.
* **degree dispersion** — coefficient of variation of node degree (reported,
  not scored: heavy-tailed degree alone is not harm, but it contextualises the
  other signals).
* **ego feature variance** — mean within-neighbourhood feature variance
  (reported, contextual).
* **graph-vs-feature validation gap** — *optional*: if the caller supplies the
  validation F1 of a feature-only model and a graph model, a positive gap
  (feature beats graph on val) is direct evidence the graph hurts. Folded into
  the score only when provided.

All scored components live in ``[0, 1]`` and ``harm_score`` is their mean over
the components that could be computed. ``GraphHarmReport.warning`` is raised
when the score crosses ``harm_threshold``. Nothing here trains a model or reads
test labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphHarmReport:
    harm_score: float
    components: Dict[str, float]
    diagnostics: Dict[str, float]
    warning: bool
    harm_threshold: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "harm_score": round(float(self.harm_score), 6),
            "warning": bool(self.warning),
            "harm_threshold": float(self.harm_threshold),
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
            "diagnostics": {k: round(float(v), 6) for k, v in self.diagnostics.items()},
            "notes": list(self.notes),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _undirected_degree(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    deg = np.zeros(n_nodes, dtype=float)
    np.add.at(deg, edge_index[0], 1.0)
    np.add.at(deg, edge_index[1], 1.0)
    return deg


def _neighbour_mean_features(edge_index: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mean of each node's neighbour features (undirected), zeros if isolated."""
    n, d = x.shape
    acc = np.zeros((n, d), dtype=float)
    deg = np.zeros(n, dtype=float)
    src, dst = edge_index[0], edge_index[1]
    np.add.at(acc, src, x[dst])
    np.add.at(acc, dst, x[src])
    np.add.at(deg, src, 1.0)
    np.add.at(deg, dst, 1.0)
    safe = np.maximum(deg, 1.0)[:, None]
    return acc / safe


def _edge_homophily(edge_index: np.ndarray, y: np.ndarray, labeled: np.ndarray) -> Optional[float]:
    src, dst = edge_index[0], edge_index[1]
    both = labeled[src] & labeled[dst]
    if not both.any():
        return None
    same = (y[src] == y[dst]) & both
    return float(same.sum()) / float(both.sum())


def _neighbour_label_mixing(edge_index: np.ndarray, y: np.ndarray, labeled: np.ndarray, n_nodes: int) -> Optional[float]:
    """Mean over labelled nodes of the fraction of labelled neighbours that disagree."""
    src, dst = edge_index[0], edge_index[1]
    both = labeled[src] & labeled[dst]
    if not both.any():
        return None
    es, ed = src[both], dst[both]
    diff = (y[es] != y[ed]).astype(float)
    diff_count = np.zeros(n_nodes, dtype=float)
    deg_count = np.zeros(n_nodes, dtype=float)
    # undirected: count each labelled-labelled edge for both endpoints
    np.add.at(diff_count, es, diff)
    np.add.at(diff_count, ed, diff)
    np.add.at(deg_count, es, 1.0)
    np.add.at(deg_count, ed, 1.0)
    has_nbr = deg_count > 0
    if not has_nbr.any():
        return None
    frac = diff_count[has_nbr] / deg_count[has_nbr]
    return float(frac.mean())


def _feature_neighbour_drift(edge_index: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> Optional[float]:
    """Edge feature distance relative to random-pair distance, squashed to [0, 1]."""
    src, dst = edge_index[0], edge_index[1]
    if src.size == 0:
        return None
    d_edge = np.linalg.norm(x[src] - x[dst], axis=1).mean()
    n = x.shape[0]
    m = min(max(src.size, 256), 10000)
    a = rng.integers(0, n, size=m)
    b = rng.integers(0, n, size=m)
    mask = a != b
    if not mask.any():
        return None
    d_rand = np.linalg.norm(x[a[mask]] - x[b[mask]], axis=1).mean()
    if d_rand <= 1e-12:
        return 0.0
    ratio = d_edge / d_rand
    return float(np.clip(ratio, 0.0, 2.0) / 2.0)


def _temporal_instability(edge_index: np.ndarray, time_steps: np.ndarray) -> Optional[float]:
    src, dst = edge_index[0], edge_index[1]
    if src.size == 0:
        return None
    span = float(time_steps.max() - time_steps.min())
    if span <= 0:
        return 0.0
    gaps = np.abs(time_steps[src] - time_steps[dst]).astype(float)
    return float(np.clip(gaps.mean() / span, 0.0, 1.0))


def _degree_cv(deg: np.ndarray) -> float:
    m = deg.mean()
    if m <= 1e-12:
        return 0.0
    return float(deg.std() / m)


def _ego_feature_variance(edge_index: np.ndarray, x: np.ndarray) -> float:
    nbr_mean = _neighbour_mean_features(edge_index, x)
    return float(((x - nbr_mean) ** 2).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class GraphHarmDetector:
    """Compute leakage-safe graph-harm diagnostics and a single harm score.

    Parameters
    ----------
    harm_threshold:
        ``harm_score`` at or above this raises ``GraphHarmReport.warning``.
    weights:
        Optional per-component weight overrides. Components absent from the
        graph (e.g. temporal instability with no time provided) are dropped and
        the remaining weights renormalised.
    seed:
        Seed for the random-pair sample used by the feature-drift estimator —
        makes the score deterministic.
    """

    DEFAULT_WEIGHTS = {
        "heterophily": 1.0,
        "label_mixing": 1.0,
        "feature_drift": 1.0,
        "temporal_instability": 1.0,
        "val_gap": 1.0,
    }

    def __init__(
        self,
        harm_threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ) -> None:
        self.harm_threshold = float(harm_threshold)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.seed = int(seed)

    def analyze(
        self,
        edge_index: np.ndarray,
        features: np.ndarray,
        *,
        time_steps: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        labeled_mask: Optional[np.ndarray] = None,
        feature_val_f1: Optional[float] = None,
        graph_val_f1: Optional[float] = None,
    ) -> GraphHarmReport:
        """Return a :class:`GraphHarmReport`. Uses train/val labels only.

        ``labeled_mask`` flags the nodes whose ``labels`` may be read (the
        train+val nodes). If ``labels`` is given without a mask, every node with
        a non-negative, non-zero label is treated as labelled (the Elliptic
        convention: ``0`` = unknown). Test labels must simply not be included in
        ``labeled_mask`` — there is no argument through which they could enter
        the harm score. ``feature_val_f1`` / ``graph_val_f1`` are *validation*
        metrics; supplying both folds the graph-vs-feature gap into the score.
        """
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must have shape (N, F)")
        n_nodes = x.shape[0]
        rng = np.random.default_rng(self.seed)

        components: Dict[str, float] = {}
        diagnostics: Dict[str, float] = {}
        notes: List[str] = []

        # Resolve labelled set (train/val only).
        labeled = None
        y = None
        if labels is not None:
            y = np.asarray(labels)
            if labeled_mask is not None:
                labeled = np.asarray(labeled_mask, dtype=bool)
            else:
                labeled = (y != 0) & (y >= 0)
                notes.append("labeled_mask not given; inferred from non-zero labels (Elliptic convention)")

        # 1. homophily / heterophily
        if y is not None and labeled is not None:
            hom = _edge_homophily(edge_index, y, labeled)
            if hom is not None:
                diagnostics["edge_homophily"] = hom
                components["heterophily"] = float(np.clip(1.0 - hom, 0.0, 1.0))
            else:
                notes.append("no labelled-labelled edges; homophily undefined")
            mix = _neighbour_label_mixing(edge_index, y, labeled, n_nodes)
            if mix is not None:
                diagnostics["neighbour_label_mixing"] = mix
                components["label_mixing"] = float(np.clip(mix, 0.0, 1.0))
        else:
            notes.append("no labels supplied; label-based harm signals skipped")

        # 2. feature drift
        drift = _feature_neighbour_drift(edge_index, x, rng)
        if drift is not None:
            components["feature_drift"] = drift
            diagnostics["feature_neighbour_drift"] = drift

        # 3. temporal instability
        if time_steps is not None:
            ts = np.asarray(time_steps)
            inst = _temporal_instability(edge_index, ts)
            if inst is not None:
                components["temporal_instability"] = inst
                diagnostics["temporal_instability"] = inst
        else:
            notes.append("no time_steps supplied; temporal instability skipped")

        # 4. graph-vs-feature validation gap (optional, direct evidence)
        if feature_val_f1 is not None and graph_val_f1 is not None:
            gap = float(feature_val_f1) - float(graph_val_f1)
            diagnostics["graph_vs_feature_val_gap"] = gap
            # squash gap into [0,1]: 0 when graph >= feature, ->1 as feature wins big
            components["val_gap"] = float(np.clip(gap, 0.0, 0.5) / 0.5)

        # contextual-only diagnostics (not scored)
        deg = _undirected_degree(edge_index, n_nodes)
        diagnostics["degree_cv"] = _degree_cv(deg)
        diagnostics["mean_degree"] = float(deg.mean())
        diagnostics["ego_feature_variance"] = _ego_feature_variance(edge_index, x)
        diagnostics["isolated_fraction"] = float((deg == 0).mean())

        if not components:
            raise ValueError(
                "no harm components could be computed; supply labels and/or features"
            )

        # weighted mean over available components
        wsum = sum(self.weights.get(k, 1.0) for k in components)
        harm = sum(self.weights.get(k, 1.0) * v for k, v in components.items()) / wsum
        harm = float(np.clip(harm, 0.0, 1.0))

        return GraphHarmReport(
            harm_score=harm,
            components=components,
            diagnostics=diagnostics,
            warning=harm >= self.harm_threshold,
            harm_threshold=self.harm_threshold,
            notes=notes,
        )


def analyze_by_window(
    edge_index: np.ndarray,
    features: np.ndarray,
    time_steps: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    labeled_mask: Optional[np.ndarray] = None,
    detector: Optional[GraphHarmDetector] = None,
) -> Dict[int, GraphHarmReport]:
    """Run the harm detector independently on each time bucket's induced subgraph.

    For each bucket ``t`` we keep nodes with ``time_steps == t`` and the edges
    fully inside that bucket, then score the subgraph. Surfaces *when* the graph
    turns harmful over deployment time. Buckets with no internal edges are
    skipped.
    """
    det = detector or GraphHarmDetector()
    ts = np.asarray(time_steps)
    out: Dict[int, GraphHarmReport] = {}
    for t in np.unique(ts):
        node_mask = ts == t
        idx = np.where(node_mask)[0]
        remap = -np.ones(node_mask.shape[0], dtype=int)
        remap[idx] = np.arange(idx.size)
        src, dst = edge_index[0], edge_index[1]
        keep = node_mask[src] & node_mask[dst]
        if not keep.any():
            continue
        sub_edges = np.stack([remap[src[keep]], remap[dst[keep]]])
        sub_x = features[idx]
        sub_labels = None if labels is None else np.asarray(labels)[idx]
        sub_mask = None if labeled_mask is None else np.asarray(labeled_mask)[idx]
        try:
            out[int(t)] = det.analyze(
                sub_edges, sub_x, labels=sub_labels, labeled_mask=sub_mask
            )
        except ValueError:
            continue
    return out
