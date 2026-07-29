from __future__ import annotations

import numpy as np

from coregraph.data.synthetic import (
    SyntheticControls,
    SyntheticRegime,
    generate_contract_shift,
    oracle_expert_ranking,
)


def _risk(scores: np.ndarray, labels: np.ndarray, mask=None) -> float:
    keep = np.ones(len(labels), dtype=bool) if mask is None else mask
    return float(np.mean((scores[keep] - labels[keep]) ** 2))


def _recall(scores: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(labels) * fraction)))
    selected = np.argsort(-scores, kind="stable")[:k]
    return float(labels[selected].sum() / max(labels.sum(), 1))


def test_synthetic_expert_wins_and_crossing_are_qualitative() -> None:
    controls = SyntheticControls(num_nodes=128, seed=21)
    graph = generate_contract_shift(SyntheticRegime.GRAPH_BEST, controls)
    feature = generate_contract_shift(SyntheticRegime.FEATURE_BEST, controls)
    assert oracle_expert_ranking(graph)[0][0] == "graph"
    assert oracle_expert_ranking(feature)[0][0] == "feature"

    crossing = generate_contract_shift(SyntheticRegime.ORDERING_CROSSES, controls)
    early = crossing.timestamps < crossing.controls.time_steps // 2
    late = ~early
    assert _risk(
        crossing.expert_scores["graph"],
        crossing.labels,
        early,
    ) < _risk(crossing.expert_scores["feature"], crossing.labels, early)
    assert _risk(
        crossing.expert_scores["feature"],
        crossing.labels,
        late,
    ) < _risk(crossing.expert_scores["graph"], crossing.labels, late)


def test_unseen_combination_and_interaction_failure_are_explicit() -> None:
    controls = SyntheticControls(num_nodes=128, seed=22)
    unseen = generate_contract_shift(
        SyntheticRegime.FACTORISED_GENERALISATION,
        controls,
    )
    interaction = generate_contract_shift(
        SyntheticRegime.INTERACTION_BREAKS_FACTORISATION,
        controls,
    )
    assert unseen.mechanism_report["additive_axis_effect"] is True
    assert unseen.mechanism_report["interaction_residual"] == 0
    assert interaction.mechanism_report["additive_axis_effect"] is False
    assert interaction.mechanism_report["interaction_residual"] == 1


def test_resource_mask_removes_best_expert_and_budget_changes_preference() -> None:
    controls = SyntheticControls(num_nodes=256, seed=23)
    resource = generate_contract_shift(SyntheticRegime.RESOURCE_MASK, controls)
    assert _risk(
        resource.expert_scores["graph"],
        resource.labels,
    ) < _risk(resource.expert_scores["feature"], resource.labels)
    assert resource.expert_available["graph"] is False
    assert oracle_expert_ranking(resource)[0][0] == "feature"

    budget = generate_contract_shift(
        SyntheticRegime.BUDGET_CHANGES_EXPERT,
        controls,
    )
    graph_small = _recall(
        budget.expert_scores["graph"],
        budget.labels,
        0.005,
    )
    feature_small = _recall(
        budget.expert_scores["feature"],
        budget.labels,
        0.005,
    )
    graph_large = _recall(
        budget.expert_scores["graph"],
        budget.labels,
        0.2,
    )
    feature_large = _recall(
        budget.expert_scores["feature"],
        budget.labels,
        0.2,
    )
    assert graph_small > feature_small
    assert feature_large > graph_large


def test_noisy_metadata_all_unavailable_and_calibration_mismatch() -> None:
    controls = SyntheticControls(num_nodes=128, seed=24)
    noisy = generate_contract_shift(
        SyntheticRegime.NOISY_CONTRACT_METADATA,
        controls,
    )
    assert noisy.mechanism_report["contract_metadata_noise"] == 0.25
    assert all(np.isfinite(scores).all() for scores in noisy.expert_scores.values())

    unavailable = generate_contract_shift(
        SyntheticRegime.ALL_EXPERTS_UNAVAILABLE,
        controls,
    )
    assert not any(unavailable.expert_available.values())

    calibration = generate_contract_shift(
        SyntheticRegime.CALIBRATION_MISMATCH,
        controls,
    )
    feature_risk = _risk(
        calibration.expert_scores["feature"],
        calibration.labels,
    )
    graph_risk = _risk(
        calibration.expert_scores["graph"],
        calibration.labels,
    )
    assert feature_risk < graph_risk
    assert np.array_equal(
        np.argsort(calibration.expert_scores["feature"]),
        np.argsort(calibration.expert_scores["graph"]),
    )
