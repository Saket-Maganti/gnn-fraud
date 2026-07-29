"""Verified multiplicity corrections with strict input validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CorrectionResult:
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]
    method: str
    alpha: float


def _validate(p_values: Sequence[float], alpha: float) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if np.isnan(p).any():
        raise ValueError("NaN p-values are not permitted; define the family explicitly")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    return p


def holm(p_values: Sequence[float], alpha: float = 0.05) -> CorrectionResult:
    p = _validate(p_values, alpha)
    m = len(p)
    if m == 0:
        return CorrectionResult((), (), "holm", alpha)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    # Explicit step-down decisions: after the first failure, all later tests
    # fail even when their individual threshold comparison would pass.
    reject_sorted = np.zeros(m, dtype=bool)
    active = True
    for rank, value in enumerate(sorted_p):
        if active and value <= alpha / (m - rank):
            reject_sorted[rank] = True
        else:
            active = False
    reject = np.empty(m, dtype=bool)
    reject[order] = reject_sorted
    return CorrectionResult(
        tuple(float(v) for v in adjusted),
        tuple(bool(v) for v in reject),
        "holm",
        alpha,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> CorrectionResult:
    p = _validate(p_values, alpha)
    m = len(p)
    if m == 0:
        return CorrectionResult((), (), "benjamini_hochberg", alpha)
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    raw = sorted_p * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= alpha
    return CorrectionResult(
        tuple(float(v) for v in adjusted),
        tuple(bool(v) for v in reject),
        "benjamini_hochberg",
        alpha,
    )


def bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> CorrectionResult:
    p = _validate(p_values, alpha)
    adjusted = np.minimum(1.0, p * len(p))
    return CorrectionResult(
        tuple(float(v) for v in adjusted),
        tuple(bool(v) for v in (adjusted <= alpha)),
        "bonferroni",
        alpha,
    )


def westfall_young_placeholder(
    statistic_fn: Callable[[np.ndarray], np.ndarray],
    observed: np.ndarray,
    permutations: np.ndarray,
) -> CorrectionResult:
    """Deliberately unavailable until a resampling design is declared.

    The signature prevents callers from treating an unimplemented method as a
    number-producing baseline.
    """

    del statistic_fn, observed, permutations
    raise NotImplementedError(
        "Westfall–Young requires a predeclared exchangeability unit and joint "
        "permutation design; no generic fallback is scientifically safe"
    )
