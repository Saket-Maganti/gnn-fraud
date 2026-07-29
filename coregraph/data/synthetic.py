"""Controlled ContractShift generator for mechanism and theory checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

import numpy as np


class SyntheticRegime(str, Enum):
    GRAPH_BEST = "graph_best"
    FEATURE_BEST = "feature_best"
    ORDERING_CROSSES = "ordering_crosses"
    FIXED_MIXTURE_REGRET = "fixed_mixture_regret"
    FACTORISED_GENERALISATION = "factorised_generalisation"
    INTERACTION_BREAKS_FACTORISATION = "interaction_breaks_factorisation"
    RESOURCE_MASK = "resource_mask"
    BUDGET_CHANGES_EXPERT = "budget_changes_expert"
    NOISY_CONTRACT_METADATA = "noisy_contract_metadata"
    ALL_EXPERTS_UNAVAILABLE = "all_experts_unavailable"
    CALIBRATION_MISMATCH = "calibration_mismatch"


@dataclass(frozen=True)
class SyntheticControls:
    num_nodes: int = 200
    time_steps: int = 4
    class_prior: float = 0.1
    feature_signal: float = 1.0
    graph_signal: float = 1.0
    homophily: float = 0.8
    heterophily: float = 0.2
    degree_mean: float = 4.0
    hub_concentration: float = 0.0
    edge_deletion: float = 0.0
    directed: bool = False
    temporal_drift: float = 0.0
    covariate_shift: float = 0.0
    concept_shift: float = 0.0
    missing_feature_fraction: float = 0.0
    missing_graph: bool = False
    feature_expert_available: bool = True
    graph_expert_available: bool = True
    contract_metadata_noise: float = 0.0
    review_budget: float = 0.01
    label_delay: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_nodes < 20 or self.time_steps < 1:
            raise ValueError("synthetic graph requires >=20 nodes and >=1 time step")
        for name in (
            "class_prior",
            "homophily",
            "heterophily",
            "hub_concentration",
            "edge_deletion",
            "missing_feature_fraction",
            "review_budget",
            "contract_metadata_noise",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True)
class ContractShiftSample:
    features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    edge_index: np.ndarray
    expert_scores: Mapping[str, np.ndarray]
    expert_available: Mapping[str, bool]
    controls: SyntheticControls
    regime: SyntheticRegime
    mechanism_report: Mapping[str, object]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def _generate_edges(
    labels: np.ndarray,
    controls: SyntheticControls,
    rng: np.random.Generator,
) -> np.ndarray:
    if controls.missing_graph:
        return np.empty((2, 0), dtype=int)
    n = len(labels)
    edge_count = max(1, int(n * controls.degree_mean / 2))
    sources = np.empty(edge_count, dtype=int)
    targets = np.empty(edge_count, dtype=int)
    for index in range(edge_count):
        if controls.hub_concentration and rng.random() < controls.hub_concentration:
            source = 0
        else:
            source = int(rng.integers(n))
        same = rng.random() < controls.homophily
        candidates = np.where(labels == labels[source] if same else labels != labels[source])[0]
        target = int(rng.choice(candidates)) if len(candidates) else int(rng.integers(n))
        sources[index], targets[index] = source, target
    keep = rng.random(edge_count) >= controls.edge_deletion
    edges = np.stack([sources[keep], targets[keep]])
    if not controls.directed:
        edges = np.concatenate([edges, edges[::-1]], axis=1)
    return edges


def _neighbour_signal(edge_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    total = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)
    if edge_index.size:
        np.add.at(total, edge_index[1], labels[edge_index[0]])
        np.add.at(count, edge_index[1], 1)
    return np.divide(total, count, out=np.full(n, labels.mean()), where=count > 0)


def generate_contract_shift(
    regime: SyntheticRegime | str,
    controls: Optional[SyntheticControls] = None,
) -> ContractShiftSample:
    regime = SyntheticRegime(regime)
    base = controls or SyntheticControls()
    overrides: dict[str, object] = {}
    if regime is SyntheticRegime.GRAPH_BEST:
        overrides = {"feature_signal": 0.2, "graph_signal": 3.0, "homophily": 0.95}
    elif regime is SyntheticRegime.FEATURE_BEST:
        overrides = {"feature_signal": 3.0, "graph_signal": 0.1, "homophily": 0.5}
    elif regime in {
        SyntheticRegime.ORDERING_CROSSES,
        SyntheticRegime.FIXED_MIXTURE_REGRET,
    }:
        overrides = {"feature_signal": 2.0, "graph_signal": 2.0, "temporal_drift": 1.0}
    elif regime is SyntheticRegime.RESOURCE_MASK:
        overrides = {"feature_signal": 0.4, "graph_signal": 3.0, "graph_expert_available": False}
    elif regime is SyntheticRegime.BUDGET_CHANGES_EXPERT:
        overrides = {"feature_signal": 1.5, "graph_signal": 1.5, "review_budget": 0.05}
    elif regime is SyntheticRegime.NOISY_CONTRACT_METADATA:
        overrides = {"contract_metadata_noise": 0.25}
    elif regime is SyntheticRegime.ALL_EXPERTS_UNAVAILABLE:
        overrides = {
            "feature_expert_available": False,
            "graph_expert_available": False,
            "missing_graph": True,
        }
    elif regime is SyntheticRegime.CALIBRATION_MISMATCH:
        overrides = {"feature_signal": 2.0, "graph_signal": 2.0}
    parameters = {**base.__dict__, **overrides}
    effective = SyntheticControls(**parameters)
    rng = np.random.default_rng(effective.seed)
    timestamps = np.repeat(
        np.arange(effective.time_steps),
        int(np.ceil(effective.num_nodes / effective.time_steps)),
    )[: effective.num_nodes]
    prior = np.clip(
        effective.class_prior
        + effective.concept_shift * (timestamps / max(effective.time_steps - 1, 1) - 0.5),
        0.01,
        0.99,
    )
    labels = (rng.random(effective.num_nodes) < prior).astype(int)
    latent = (2 * labels - 1).astype(float)
    drift = effective.covariate_shift * timestamps / max(effective.time_steps - 1, 1)
    features = np.column_stack(
        [
            effective.feature_signal * latent + drift + rng.normal(0, 1, effective.num_nodes),
            rng.normal(0, 1, effective.num_nodes),
            timestamps / max(effective.time_steps - 1, 1),
        ]
    )
    missing = rng.random(features.shape) < effective.missing_feature_fraction
    features[missing] = np.nan
    edges = _generate_edges(labels, effective, rng)
    feature_scores = _sigmoid(
        effective.feature_signal * latent
        + rng.normal(0, 1, effective.num_nodes)
        + effective.temporal_drift * (timestamps >= effective.time_steps // 2) * latent
    )
    neighbour = _neighbour_signal(edges, labels)
    graph_margin = (2 * neighbour - 1) * effective.graph_signal
    graph_scores = _sigmoid(graph_margin + rng.normal(0, 0.7, effective.num_nodes))

    if regime in {
        SyntheticRegime.ORDERING_CROSSES,
        SyntheticRegime.FIXED_MIXTURE_REGRET,
    }:
        late = timestamps >= effective.time_steps // 2
        # Enforce an interpretable rank crossing: graph is near-oracle early,
        # feature is near-oracle late.
        graph_scores = np.where(
            late,
            _sigmoid(-2.5 * latent + rng.normal(0, 0.3, len(labels))),
            _sigmoid(2.5 * latent + rng.normal(0, 0.3, len(labels))),
        )
        feature_scores = np.where(
            late,
            _sigmoid(2.5 * latent + rng.normal(0, 0.3, len(labels))),
            _sigmoid(-2.5 * latent + rng.normal(0, 0.3, len(labels))),
        )
    if regime is SyntheticRegime.BUDGET_CHANGES_EXPERT:
        positives = np.where(labels == 1)[0]
        graph_scores = rng.uniform(0.0, 0.5, len(labels))
        feature_scores = rng.uniform(0.0, 0.6, len(labels))
        # Graph concentrates a few positives at the very top; features spread
        # signal across more positives, inducing budget-dependent preference.
        graph_scores[positives[: max(1, len(positives) // 4)]] = 0.99
        feature_scores[positives] = np.linspace(0.95, 0.65, len(positives))
        negatives = np.where(labels == 0)[0]
        feature_scores[negatives[: max(1, len(negatives) // 100)]] = 0.98
    if regime is SyntheticRegime.CALIBRATION_MISMATCH:
        calibrated = np.where(labels == 1, 0.8, 0.2)
        feature_scores = calibrated
        graph_scores = np.sqrt(calibrated)

    additive_axis_effect = (
        regime is not SyntheticRegime.INTERACTION_BREAKS_FACTORISATION
    )
    interaction_residual = (
        0.0 if additive_axis_effect else 1.0
    )
    return ContractShiftSample(
        features=features,
        labels=labels,
        timestamps=timestamps,
        edge_index=edges,
        expert_scores={"feature": feature_scores, "graph": graph_scores},
        expert_available={
            "feature": effective.feature_expert_available,
            "graph": effective.graph_expert_available and not effective.missing_graph,
        },
        controls=effective,
        regime=regime,
        mechanism_report={
            "illustrative_only": True,
            "fraud_realism_claim": "PROHIBITED",
            "known_ordering": regime.value,
            "additive_axis_effect": additive_axis_effect,
            "interaction_residual": interaction_residual,
            "review_budget": effective.review_budget,
            "label_delay": effective.label_delay,
            "contract_metadata_noise": effective.contract_metadata_noise,
        },
    )


def oracle_expert_ranking(
    sample: ContractShiftSample,
    *,
    mask: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    keep = np.ones(len(sample.labels), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    risks = []
    for expert, scores in sample.expert_scores.items():
        if sample.expert_available[expert]:
            risk = float(np.mean((scores[keep] - sample.labels[keep]) ** 2))
            risks.append((expert, risk))
    return sorted(risks, key=lambda row: (row[1], row[0]))
