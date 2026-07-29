"""Seed-blocked exact and resampling-based inference."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


def _diffs(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired inference requires equal shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite values must be resolved before defining the seed family")
    return left - right


@dataclass(frozen=True)
class PairedInference:
    method: str
    statistic: float
    p_value: float
    n_blocks: int
    mean_difference: float
    median_difference: float
    paired_effect_size: float
    interpretation: str = "fixed_dataset_seed_block"


def exact_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> PairedInference:
    diffs = _diffs(a, b)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return PairedInference("exact_wilcoxon", 0.0, 1.0, len(diffs), 0.0, 0.0, 0.0)
    result = wilcoxon(
        nonzero,
        alternative=alternative,
        method="exact",
        zero_method="wilcox",
    )
    return PairedInference(
        method="exact_wilcoxon",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_blocks=len(diffs),
        mean_difference=float(diffs.mean()),
        median_difference=float(np.median(diffs)),
        paired_effect_size=paired_effect_size(diffs),
    )


def sign_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> PairedInference:
    diffs = _diffs(a, b)
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    n = wins + losses
    p = 1.0 if n == 0 else float(binomtest(wins, n, 0.5, alternative=alternative).pvalue)
    return PairedInference(
        "sign_test",
        float(wins),
        p,
        len(diffs),
        float(diffs.mean()),
        float(np.median(diffs)),
        paired_effect_size(diffs),
    )


def paired_permutation(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
    n_permutations: int = 100_000,
    seed: int = 20260729,
) -> PairedInference:
    diffs = _diffs(a, b)
    observed = float(diffs.mean())
    n = len(diffs)
    if n <= 20:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(n_permutations, n))
    null = (signs * diffs).mean(axis=1)
    if alternative == "two-sided":
        extreme = np.abs(null) >= abs(observed)
    elif alternative == "greater":
        extreme = null >= observed
    elif alternative == "less":
        extreme = null <= observed
    else:
        raise ValueError("alternative must be two-sided greater or less")
    p = float((extreme.sum() + (0 if n <= 20 else 1)) / (len(null) + (0 if n <= 20 else 1)))
    return PairedInference(
        "paired_permutation_exact" if n <= 20 else "paired_permutation_monte_carlo",
        observed,
        p,
        n,
        observed,
        float(np.median(diffs)),
        paired_effect_size(diffs),
    )


def paired_effect_size(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return 0.0 if std == 0 else float(values.mean() / std)


def bootstrap_seed_blocks(
    differences: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260729,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if not np.isfinite(values).all() or values.ndim != 1:
        raise ValueError("seed-block differences must be a finite vector")
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_contexts_within_seed(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    seed_key: str = "seed",
    value_key: str = "value",
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row[seed_key]), []).append(float(row[value_key]))
    return {seed: float(np.mean(values)) for seed, values in sorted(grouped.items())}
