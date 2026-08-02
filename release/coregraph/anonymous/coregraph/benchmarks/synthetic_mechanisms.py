"""Deterministic controlled mechanisms for compositional router checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Mechanism(str, Enum):
    FEATURE_CORRUPTION = "feature_corruption"
    EDGE_DELETION = "edge_deletion"
    DEGREE_SHIFT = "degree_shift"
    HOMOPHILY_REDUCTION = "homophily_reduction"
    HETEROPHILY_REVERSAL = "heterophily_reversal"
    TEMPORAL_CONCEPT_DRIFT = "temporal_concept_drift"
    DELAYED_LABELS = "delayed_labels"
    NEIGHBOURHOOD_TRUNCATION = "neighbourhood_truncation"
    EXPERT_UNAVAILABILITY = "expert_unavailability"
    REVIEW_BUDGET_CONTRACTION = "review_budget_contraction"
    CORRELATED_GRAPH_FEATURE_SHIFT = "correlated_graph_feature_shift"
    NOISY_CONTRACT_METADATA = "noisy_contract_metadata"
    MISSING_CONTRACT_AXIS = "missing_contract_axis"
    DYNAMIC_RESOURCE_CHANGE = "dynamic_resource_change"
    ALL_EXPERT_DEGRADATION = "all_expert_degradation"


@dataclass(frozen=True)
class MechanismSpec:
    mechanism: Mechanism
    generator: str
    source_contract: str
    held_out_contract: str
    expected_best_expert: str
    expected_router_direction: str
    ground_truth: str
    pilot_size: int
    full_size: int
    metrics: tuple[str, ...]
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class MechanismFixture:
    features: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray
    expert_scores: np.ndarray
    feasible_mask: np.ndarray
    contract_metadata: np.ndarray
    review_budget: float
    mechanism: Mechanism
    seed: int
    deterministic_sha256: str


def mechanism_registry() -> tuple[MechanismSpec, ...]:
    common_metrics = ("brier_contract_regret", "routing_stability", "selective_risk")
    direction = {
        Mechanism.FEATURE_CORRUPTION: ("graph", "reduce feature-expert mass"),
        Mechanism.EDGE_DELETION: ("feature", "reduce graph-expert mass"),
        Mechanism.DEGREE_SHIFT: ("context_dependent", "respond to degree-shift diagnostic"),
        Mechanism.HOMOPHILY_REDUCTION: ("feature", "reduce homophily-dependent expert mass"),
        Mechanism.HETEROPHILY_REVERSAL: ("feature", "avoid unadapted homophily expert"),
        Mechanism.TEMPORAL_CONCEPT_DRIFT: ("recent", "increase recent-window expert mass"),
        Mechanism.DELAYED_LABELS: ("label_free", "avoid label-dependent adaptation"),
        Mechanism.NEIGHBOURHOOD_TRUNCATION: ("feature", "reduce high-receptive-field expert mass"),
        Mechanism.EXPERT_UNAVAILABILITY: ("feasible_only", "assign unavailable expert zero mass"),
        Mechanism.REVIEW_BUDGET_CONTRACTION: ("budget_specific", "prefer top-ranked economical policy"),
        Mechanism.CORRELATED_GRAPH_FEATURE_SHIFT: ("context_dependent", "increase abstention if all diagnostics worsen"),
        Mechanism.NOISY_CONTRACT_METADATA: ("diagnostic_aware", "downweight uncertain metadata"),
        Mechanism.MISSING_CONTRACT_AXIS: ("diagnostic_aware", "use missing token and uncertainty"),
        Mechanism.DYNAMIC_RESOURCE_CHANGE: ("feasible_only", "reroute between batches"),
        Mechanism.ALL_EXPERT_DEGRADATION: ("abstain", "increase abstention"),
    }
    return tuple(
        MechanismSpec(
            mechanism=mechanism,
            generator=f"deterministic_v1:{mechanism.value}",
            source_contract="base_graph_feature_contract",
            held_out_contract=f"base_plus_{mechanism.value}",
            expected_best_expert=direction[mechanism][0],
            expected_router_direction=direction[mechanism][1],
            ground_truth=mechanism.value,
            pilot_size=512,
            full_size=10_000,
            metrics=common_metrics,
            failure_modes=("wrong_expert_selected", "all_experts_poor", "routing_instability"),
        )
        for mechanism in Mechanism
    )


def _hash_fixture(arrays: tuple[np.ndarray, ...], metadata: str) -> str:
    digest = hashlib.sha256(metadata.encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def generate_mechanism_fixture(
    mechanism: Mechanism | str,
    *,
    size: int = 64,
    seed: int = 0,
) -> MechanismFixture:
    mechanism = Mechanism(mechanism)
    if size < 20:
        raise ValueError("mechanism fixture size must be at least 20")
    rng = np.random.default_rng(seed)
    labels = (rng.random(size) < 0.2).astype(int)
    latent = labels * 2 - 1
    features = np.column_stack((latent + rng.normal(0, 0.7, size), rng.normal(size=size)))
    sources = np.arange(size)
    targets = (sources + 1) % size
    edge_index = np.stack((sources, targets))
    feature_scores = 1 / (1 + np.exp(-(2 * latent + rng.normal(0, 0.5, size))))
    graph_scores = 1 / (1 + np.exp(-(2 * latent + rng.normal(0, 0.5, size))))
    recent_scores = 1 / (1 + np.exp(-(1.5 * latent + rng.normal(0, 0.6, size))))
    feasible = np.ones((size, 3), dtype=bool)
    metadata = np.ones((size, 6), dtype=float)
    budget = 0.02
    late = np.arange(size) >= size // 2

    if mechanism is Mechanism.FEATURE_CORRUPTION:
        features += rng.normal(0, 5, features.shape)
        feature_scores = rng.random(size)
    elif mechanism is Mechanism.EDGE_DELETION:
        edge_index = edge_index[:, : max(1, size // 10)]
        graph_scores = rng.random(size)
    elif mechanism is Mechanism.DEGREE_SHIFT:
        edge_index = np.stack((np.zeros(size, dtype=int), np.arange(size)))
        graph_scores = np.clip(graph_scores + rng.normal(0, 0.2, size), 0, 1)
    elif mechanism is Mechanism.HOMOPHILY_REDUCTION:
        targets = rng.permutation(targets)
        edge_index = np.stack((sources, targets))
        graph_scores = rng.random(size)
    elif mechanism is Mechanism.HETEROPHILY_REVERSAL:
        graph_scores = 1 - graph_scores
    elif mechanism is Mechanism.TEMPORAL_CONCEPT_DRIFT:
        feature_scores[late] = 1 - feature_scores[late]
        recent_scores[late] = np.clip(labels[late] * 0.8 + 0.1, 0, 1)
    elif mechanism is Mechanism.DELAYED_LABELS:
        metadata[late, 3] = 0
    elif mechanism is Mechanism.NEIGHBOURHOOD_TRUNCATION:
        edge_index = edge_index[:, ::4]
        graph_scores = np.clip(graph_scores + rng.normal(0, 0.3, size), 0, 1)
    elif mechanism is Mechanism.EXPERT_UNAVAILABILITY:
        feasible[:, 1] = False
    elif mechanism is Mechanism.REVIEW_BUDGET_CONTRACTION:
        budget = 0.005
    elif mechanism is Mechanism.CORRELATED_GRAPH_FEATURE_SHIFT:
        features += rng.normal(0, 3, features.shape)
        feature_scores = rng.random(size)
        graph_scores = rng.random(size)
    elif mechanism is Mechanism.NOISY_CONTRACT_METADATA:
        metadata += rng.normal(0, 0.5, metadata.shape)
    elif mechanism is Mechanism.MISSING_CONTRACT_AXIS:
        metadata[:, 2] = np.nan
    elif mechanism is Mechanism.DYNAMIC_RESOURCE_CHANGE:
        feasible[late, 1] = False
        feasible[~late, 0] = False
    elif mechanism is Mechanism.ALL_EXPERT_DEGRADATION:
        feature_scores = rng.random(size)
        graph_scores = rng.random(size)
        recent_scores = rng.random(size)

    scores = np.column_stack((feature_scores, graph_scores, recent_scores))
    checksum = _hash_fixture(
        (features, labels, edge_index, scores, feasible, metadata),
        f"{mechanism.value}:{seed}:{budget}",
    )
    return MechanismFixture(
        features,
        labels,
        edge_index,
        scores,
        feasible,
        metadata,
        budget,
        mechanism,
        seed,
        checksum,
    )
