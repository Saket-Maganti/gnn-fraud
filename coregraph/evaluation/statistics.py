"""Seed-blocked exact and resampling-based inference."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

import numpy as np
from scipy.stats import binomtest, wilcoxon

FloatArrayLike: TypeAlias = Sequence[float] | np.ndarray


def _diffs(a: FloatArrayLike, b: FloatArrayLike) -> np.ndarray:
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


@dataclass(frozen=True)
class PairedSeedBlocks:
    seeds: tuple[int, ...]
    method_values: tuple[float, ...]
    baseline_values: tuple[float, ...]
    contexts_per_seed: tuple[int, ...]
    aggregation: str = "mean_matched_target_contracts_within_expert_prediction_seed"


def build_paired_seed_blocks(
    rows: Sequence[Mapping[str, float | int | str]],
    *,
    method: str,
    baseline: str,
    metric: str,
) -> PairedSeedBlocks:
    """Pair exact target contexts inside exactly one dataset."""

    selected = [
        row
        for row in rows
        if str(row["metric"]) == metric
        and str(row["method"]) in {method, baseline}
    ]
    values: dict[tuple[str, str, int, str, str], float] = {}
    for row in selected:
        key = (
            str(row["dataset"]),
            str(
                row.get(
                    "target_protocol_id",
                    row.get("target_contract", ""),
                )
            ),
            int(
                row["expert_prediction_seed"]
                if "expert_prediction_seed" in row
                else row["seed"]
            ),
            str(row.get("fold", "")),
            str(row["method"]),
        )
        if key in values:
            raise ValueError(f"duplicate paired context row: {key}")
        values[key] = float(row["value"])
    contexts = {
        (dataset, contract, seed, fold)
        for dataset, contract, seed, fold, _ in values
    }
    missing = [
        context
        for context in sorted(contexts)
        if (*context, method) not in values or (*context, baseline) not in values
    ]
    if missing:
        raise ValueError(f"paired seed blocks have missing method rows: {missing}")
    if not contexts:
        raise ValueError("paired seed blocks contain no aligned rows")
    datasets = {context[0] for context in contexts}
    if len(datasets) != 1:
        raise ValueError(
            "paired seed blocks must be dataset-stratified; "
            "numeric seeds have no cross-dataset pairing meaning"
        )
    by_seed: dict[int, list[tuple[float, float]]] = {}
    for context in sorted(contexts):
        seed = context[2]
        by_seed.setdefault(seed, []).append(
            (values[(*context, method)], values[(*context, baseline)])
        )
    counts = {len(pairs) for pairs in by_seed.values()}
    if len(counts) != 1:
        raise ValueError("paired seeds have unequal target-context coverage")
    seeds = tuple(sorted(by_seed))
    return PairedSeedBlocks(
        seeds=seeds,
        method_values=tuple(
            float(np.mean([pair[0] for pair in by_seed[seed]]))
            for seed in seeds
        ),
        baseline_values=tuple(
            float(np.mean([pair[1] for pair in by_seed[seed]]))
            for seed in seeds
        ),
        contexts_per_seed=tuple(len(by_seed[seed]) for seed in seeds),
    )


def exact_wilcoxon(
    a: FloatArrayLike,
    b: FloatArrayLike,
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
    a: FloatArrayLike,
    b: FloatArrayLike,
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
    a: FloatArrayLike,
    b: FloatArrayLike,
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


def paired_effect_size(differences: FloatArrayLike) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return 0.0 if std == 0 else float(values.mean() / std)


def bootstrap_seed_blocks(
    differences: FloatArrayLike,
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


def hierarchical_dataset_bootstrap(
    differences_by_dataset: Mapping[str, FloatArrayLike],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260729,
) -> tuple[float, float]:
    """Bootstrap datasets first, then seed blocks inside sampled datasets."""

    if not differences_by_dataset:
        raise ValueError("dataset-stratified bootstrap requires datasets")
    datasets = tuple(sorted(differences_by_dataset))
    values: dict[str, np.ndarray] = {}
    for dataset in datasets:
        array = np.asarray(differences_by_dataset[dataset], dtype=float)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError(
                "every dataset requires a finite non-empty seed-block vector"
            )
        values[dataset] = array
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            array = values[str(dataset)]
            sampled = rng.choice(array, size=len(array), replace=True)
            dataset_means.append(float(sampled.mean()))
        estimates[index] = float(np.mean(dataset_means))
    low, high = np.quantile(estimates, [0.025, 0.975])
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
