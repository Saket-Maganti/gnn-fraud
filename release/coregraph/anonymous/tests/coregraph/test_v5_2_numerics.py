from __future__ import annotations

import numpy as np
import pytest

from coregraph.evaluation.regret import (
    feasible_row_oracle_brier_with_abstention,
    v5_matched_action_brier_metrics,
)
from coregraph.experiments.v5_numerics import (
    HULL_PROJECTION_TOLERANCE,
    HullDiagnostics,
    SimplexDiagnostics,
    _validate_normalized_feasible_weights,
    normalize_feasible_weights_float64,
    routed_scores_in_feasible_hull_float64,
)


def test_reproduces_v5_1_float32_invariant_failure_and_repairs_it() -> None:
    raw = np.asarray([[0.33333337, 0.33333337, 0.33333337]], dtype=np.float32)
    experts = np.asarray([[0.9, 0.9, 0.9]], dtype=np.float32)
    labels = np.asarray([1.0])
    old_score = np.sum(raw * experts, axis=1)
    old_regret = (old_score - labels) ** 2 - np.min(
        (experts - labels[:, None]) ** 2, axis=1
    )
    assert raw.sum(axis=1)[0] == np.float32(1.0000001)
    assert old_regret[0] < -1e-8

    weights, forced, simplex = normalize_feasible_weights_float64(
        raw, np.ones_like(raw, dtype=bool)
    )
    scores, hull = routed_scores_in_feasible_hull_float64(
        weights, experts, np.ones_like(raw, dtype=bool)
    )
    metrics = v5_matched_action_brier_metrics(
        labels=labels,
        method_scores=scores,
        method_abstains=forced,
        expert_scores=experts,
        availability=np.ones_like(raw, dtype=bool),
        abstention_cost=0.2,
    )
    assert weights.dtype == scores.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(weights.sum(axis=1), np.ones(1))
    assert simplex.max_abs_weight_sum_error_after_residual == 0.0
    assert hull.max_projection_delta <= HULL_PROJECTION_TOLERANCE
    assert metrics["minimum_raw_row_regret"] >= -1e-12
    assert metrics["rows_with_raw_regret_below_tolerance"] == 0


@pytest.mark.parametrize(
    "raw,availability,expected,forced",
    [
        ([0.50000006, 0.5, 0.0], [True, True, True], [0.50000003, 0.49999997, 0.0], False),
        ([0.49999997, 0.5, 0.0], [True, True, True], [0.499999985, 0.500000015, 0.0], False),
        ([1e-20, 1.0, 1e20], [True, True, True], [1e-40, 1e-20, 1.0], False),
        ([0.9, 0.1, 0.0], [False, True, True], [0.0, 1.0, 0.0], False),
        ([0.2, 0.3, 0.5], [False, True, False], [0.0, 1.0, 0.0], False),
        ([0.2, 0.3, 0.5], [False, False, False], [0.0, 0.0, 0.0], True),
        ([-1e-14, 0.4, 0.6], [True, True, True], [0.0, 0.4, 0.6], False),
        ([0.0, 0.0, 0.0], [True, True, True], [0.0, 0.0, 0.0], True),
    ],
)
def test_adversarial_feasible_simplex_cases(
    raw: list[float],
    availability: list[bool],
    expected: list[float],
    forced: bool,
) -> None:
    weights, abstain, diagnostics = normalize_feasible_weights_float64(
        np.asarray([raw]), np.asarray([availability])
    )
    np.testing.assert_allclose(weights[0], expected, rtol=0, atol=2e-15)
    assert bool(abstain[0]) is forced
    assert diagnostics.rows_with_unavailable_nonzero_weight == 0
    if not forced:
        assert weights[0].sum() == 1.0


@pytest.mark.parametrize("bad", [-1e-6, np.nan, np.inf, -np.inf])
def test_invalid_raw_weights_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        normalize_feasible_weights_float64(
            np.asarray([[bad, 1.0, 0.0]]), np.ones((1, 3), dtype=bool)
        )


def test_simplex_input_validation_and_unavailable_diagnostic() -> None:
    with pytest.raises(ValueError, match="aligned"):
        normalize_feasible_weights_float64(np.ones(3), np.ones((1, 3), dtype=bool))
    with pytest.raises(ValueError, match="negative weight tolerance"):
        normalize_feasible_weights_float64(
            np.ones((1, 3)), np.ones((1, 3), dtype=bool), negative_tolerance=-1
        )
    with pytest.raises(ValueError, match="simplex tolerance"):
        normalize_feasible_weights_float64(
            np.ones((1, 3)), np.ones((1, 3), dtype=bool), simplex_tolerance=0
        )
    _, _, diagnostics = normalize_feasible_weights_float64(
        np.asarray([[0.8, 0.2, 0.0]]), np.asarray([[False, True, True]])
    )
    assert diagnostics.input_rows_with_unavailable_nonzero_weight == 1
    assert diagnostics.to_dict()["rows_with_unavailable_nonzero_weight"] == 0
    assert HullDiagnostics(0.0, 0, 0.0).to_dict()["rows_projected_to_hull"] == 0


def test_post_normalization_invariants_fail_closed() -> None:
    feasible = np.ones((1, 3), dtype=bool)
    active = np.asarray([True])
    with pytest.raises(ValueError, match="maximum_sum_error"):
        _validate_normalized_feasible_weights(
            np.asarray([[0.6, 0.6, 0.0]]), feasible, active, 1e-12
        )
    with pytest.raises(ValueError, match="outside"):
        _validate_normalized_feasible_weights(
            np.asarray([[1.1, -0.1, 0.0]]), feasible, np.asarray([False]), 1e-12
        )
    with pytest.raises(RuntimeError, match="unavailable"):
        _validate_normalized_feasible_weights(
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[False, True, True]]),
            np.asarray([False]),
            1e-12,
        )


def test_hull_projection_boundary_and_substantive_failure() -> None:
    experts = np.asarray([[0.0, 1.0, 0.5], [0.2, 0.8, 0.4], [0.3, 0.3, 0.3]])
    availability = np.ones_like(experts, dtype=bool)
    weights = np.asarray(
        [[0.0, 1.0 + 5e-13, 0.0], [1.0 + 5e-13, -5e-13, 0.0], [0.2, 0.3, 0.5]]
    )
    routed, diagnostics = routed_scores_in_feasible_hull_float64(
        weights, experts, availability
    )
    np.testing.assert_allclose(routed, [1.0, 0.2, 0.3], rtol=0, atol=1e-15)
    assert diagnostics.rows_projected_to_hull == 2
    assert diagnostics.max_projection_delta <= HULL_PROJECTION_TOLERANCE
    with pytest.raises(ValueError, match="substantively violates"):
        routed_scores_in_feasible_hull_float64(
            np.asarray([[0.0, 1.0 + 1e-8, 0.0]]),
            experts[:1],
            availability[:1],
        )


@pytest.mark.parametrize(
    "weights,scores,availability,message",
    [
        (np.ones(3), np.ones((1, 3)), np.ones((1, 3), dtype=bool), "align"),
        (np.asarray([[np.nan, 0, 0]]), np.ones((1, 3)), np.ones((1, 3), dtype=bool), "finite"),
        (np.asarray([[1, 0, 0]]), np.asarray([[1.1, 0, 0]]), np.ones((1, 3), dtype=bool), "probabilities"),
        (np.asarray([[1, 0, 0]]), np.ones((1, 3)), np.asarray([[False, True, True]]), "unavailable"),
    ],
)
def test_hull_input_validation(
    weights: np.ndarray,
    scores: np.ndarray,
    availability: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        routed_scores_in_feasible_hull_float64(weights, scores, availability)
    with pytest.raises(ValueError, match="projection tolerance"):
        routed_scores_in_feasible_hull_float64(
            np.ones((1, 3)) / 3,
            np.ones((1, 3)) / 2,
            np.ones((1, 3), dtype=bool),
            projection_tolerance=-1,
        )


def test_matched_oracle_and_regret_semantics_are_unchanged() -> None:
    labels = np.asarray([0, 1, 1, 0])
    experts = np.asarray([[0.1, 0.4], [0.8, 0.7], [0.9, 0.2], [0.9, 0.8]])
    availability = np.asarray([[True, True], [True, True], [False, True], [False, False]])
    oracle = feasible_row_oracle_brier_with_abstention(
        labels=labels,
        expert_scores=experts,
        availability=availability,
        abstention_cost=0.2,
    )
    np.testing.assert_allclose(oracle, [0.01, 0.04, 0.2, 0.2])
    metrics = v5_matched_action_brier_metrics(
        labels=labels,
        method_scores=np.asarray([0.1, 0.8, 0.0, 0.0]),
        method_abstains=np.asarray([False, False, True, True]),
        expert_scores=experts,
        availability=availability,
        abstention_cost=0.2,
    )
    assert metrics["contract_regret_vs_feasible_row_oracle"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="below the frozen numeric tolerance"):
        v5_matched_action_brier_metrics(
            labels=np.asarray([1.0]),
            method_scores=np.asarray([0.9000001]),
            method_abstains=np.asarray([False]),
            expert_scores=np.asarray([[0.9, 0.9]]),
            availability=np.asarray([[True, True]]),
            abstention_cost=0.2,
        )


def test_large_dgraphfin_scale_float64_path_is_deterministic() -> None:
    rows = 170_207
    raw = np.broadcast_to(
        np.asarray([0.33333337, 0.33333337, 0.33333337], dtype=np.float32),
        (rows, 3),
    )
    experts = np.broadcast_to(np.asarray([0.9, 0.9, 0.9]), (rows, 3))
    availability = np.ones((rows, 3), dtype=bool)
    first_weights, first_forced, _ = normalize_feasible_weights_float64(
        raw, availability
    )
    first_scores, _ = routed_scores_in_feasible_hull_float64(
        first_weights, experts, availability
    )
    second_weights, second_forced, _ = normalize_feasible_weights_float64(
        raw, availability
    )
    second_scores, _ = routed_scores_in_feasible_hull_float64(
        second_weights, experts, availability
    )
    np.testing.assert_array_equal(first_weights, second_weights)
    np.testing.assert_array_equal(first_scores, second_scores)
    np.testing.assert_array_equal(first_forced, second_forced)
    assert first_scores.dtype == first_weights.dtype == np.dtype(np.float64)
