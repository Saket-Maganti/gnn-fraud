from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from coregraph.evaluation.corrections import benjamini_hochberg, bonferroni, holm
from coregraph.evaluation.metrics import binary_metrics, budget_curve_auc, deterministic_top_k
from coregraph.evaluation.statistics import (
    exact_wilcoxon,
    paired_permutation,
    sign_test,
)


@pytest.mark.parametrize("method", ["holm", "fdr_bh", "bonferroni"])
def test_corrections_match_statsmodels(method: str) -> None:
    values = [0.001, 0.012, 0.031, 0.2, 0.8]
    ours = {
        "holm": holm,
        "fdr_bh": benjamini_hochberg,
        "bonferroni": bonferroni,
    }[method](values)
    reject, adjusted, _, _ = multipletests(values, method=method)
    assert np.allclose(ours.adjusted, adjusted)
    assert np.array_equal(ours.reject, reject)


def test_holm_stops_after_first_failure_and_rejects_nan() -> None:
    result = holm([0.01, 0.03, 0.04], alpha=0.05)
    assert result.reject == (True, False, False)
    with pytest.raises(ValueError, match="NaN"):
        holm([0.01, float("nan")])


def test_exact_inference_detects_consistent_wins() -> None:
    left = [1, 2, 3, 4, 5, 6]
    right = [0, 0, 0, 0, 0, 0]
    assert sign_test(left, right).p_value <= 0.05
    assert exact_wilcoxon(left, right).mean_difference > 0
    assert paired_permutation(left, right).p_value <= 0.05


def test_budget_topk_is_stable_on_ties() -> None:
    selected = deterministic_top_k([0.7, 0.7, 0.7], 2, identifiers=["c", "a", "b"])
    assert selected.tolist() == [1, 2]
    auc = budget_curve_auc([1, 0, 1], [0.9, 0.2, 0.8], [0.34, 0.67, 1.0])
    assert 0 <= auc <= 1
    with pytest.raises(ValueError, match="finite"):
        binary_metrics([1, 2], [0.5, float("inf")])
