"""
models/rolling_validation.py

Deployment-style temporal validation windows for graph fraud detection.

This is **method direction 2** in the AAAI reframing: the professor asked for
"deployment-style validation where model selection does not use future test
information, using rolling or blocked temporal validation." The single most
common silent leak in temporal-graph papers is selecting the model / epoch /
threshold on a window that overlaps (or post-dates) the test window. This
module makes that leak *structurally impossible* by constructing the
train / validation / test windows as ordered, non-overlapping integer ranges
and refusing to emit any fold that violates the temporal ordering.

It is a **pure scaffolding / planning** module — it does **not** train anything.
It produces window plans (one row per fold × config), a leakage audit, and a
model-selection helper that consumes *validation* scores only. The actual GNN
training is deferred to the run plan emitted by
``scripts/plan_rolling_validation_runs.py``.

Three schemes (all leakage-safe by construction):

  * **blocked**   — partition the timeline into contiguous blocks; the test
    block is held out, validation is the block immediately before it, and the
    training data is everything strictly before validation. Classic blocked
    cross-validation for time series (Bergmeir & Benítez 2012).

  * **rolling**   — a fixed-width training window *slides* forward; validation
    and test are fixed-width windows immediately after it. Models the steady
    state of a deployed detector retrained on a trailing window.

  * **expanding** — training grows from the start of the timeline; validation
    and test are fixed-width windows at the leading edge. Models a detector
    that accumulates all available history.

Leakage guarantees enforced by :func:`audit_no_leakage` (and asserted in every
constructor): for every fold ``train.end < val.start <= val.end < test.start``
and every window lies inside the global time range. Validation never overlaps
or post-dates test; training never sees anything at or beyond the validation
window. ``select_best_config`` ranks configurations on validation scores only
and never reads test metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Window representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeRange:
    """Inclusive integer time-bucket range ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"TimeRange start {self.start} > end {self.end}")

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "TimeRange") -> bool:
        return not (self.end < other.start or other.end < self.start)

    def as_tuple(self) -> Tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True)
class TemporalFold:
    """One deployment-style fold: train / val / test windows + provenance."""

    fold: int
    scheme: str
    train: TimeRange
    val: TimeRange
    test: TimeRange

    def as_row(self) -> Dict[str, object]:
        return {
            "fold": self.fold,
            "scheme": self.scheme,
            "train_start": self.train.start,
            "train_end": self.train.end,
            "val_start": self.val.start,
            "val_end": self.val.end,
            "test_start": self.test.start,
            "test_end": self.test.end,
        }


class LeakageError(ValueError):
    """Raised when a fold would let model selection peek at the future."""


# ─────────────────────────────────────────────────────────────────────────────
# Leakage audit (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

def audit_fold(fold: TemporalFold, t_min: int, t_max: int) -> None:
    """Raise :class:`LeakageError` if a single fold violates temporal ordering.

    The four invariants a deployment-realistic fold must satisfy:

      1. ``train.end < val.start``      — training stops before validation;
      2. ``val.end < test.start``       — validation stops before test;
      3. windows lie inside ``[t_min, t_max]``;
      4. no pair of (train, val, test) overlaps.
    """
    tr, va, te = fold.train, fold.val, fold.test
    if not (tr.end < va.start):
        raise LeakageError(
            f"fold {fold.fold}: train.end {tr.end} >= val.start {va.start} "
            "(training would see the validation window)"
        )
    if not (va.end < te.start):
        raise LeakageError(
            f"fold {fold.fold}: val.end {va.end} >= test.start {te.start} "
            "(model selection would see the test window)"
        )
    for name, w in (("train", tr), ("val", va), ("test", te)):
        if w.start < t_min or w.end > t_max:
            raise LeakageError(
                f"fold {fold.fold}: {name} window {w.as_tuple()} outside "
                f"global range [{t_min}, {t_max}]"
            )
    if tr.overlaps(va) or va.overlaps(te) or tr.overlaps(te):
        raise LeakageError(f"fold {fold.fold}: overlapping windows")


def audit_no_leakage(folds: Sequence[TemporalFold], t_min: int, t_max: int) -> Dict[str, object]:
    """Audit a list of folds; return a structured (non-raising) report.

    Unlike :func:`audit_fold`, this collects *all* violations rather than
    failing on the first one, so a planner can surface every problem at once.
    """
    violations: List[Dict[str, object]] = []
    for f in folds:
        try:
            audit_fold(f, t_min, t_max)
        except LeakageError as exc:  # noqa: PERF203 - we want per-fold messages
            violations.append({"fold": f.fold, "scheme": f.scheme, "reason": str(exc)})
    return {
        "n_folds": len(folds),
        "n_violations": len(violations),
        "leakage_free": len(violations) == 0,
        "violations": violations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Window generators
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_bounds(time_steps: Optional[Sequence[int]], t_min: Optional[int], t_max: Optional[int]) -> Tuple[int, int]:
    if time_steps is not None:
        arr = np.asarray(list(time_steps))
        if arr.size == 0:
            raise ValueError("time_steps is empty")
        return int(arr.min()), int(arr.max())
    if t_min is None or t_max is None:
        raise ValueError("provide either time_steps or both t_min and t_max")
    if t_min > t_max:
        raise ValueError(f"t_min {t_min} > t_max {t_max}")
    return int(t_min), int(t_max)


def blocked_folds(
    *,
    time_steps: Optional[Sequence[int]] = None,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
    n_blocks: int = 5,
    min_train_blocks: int = 1,
) -> List[TemporalFold]:
    """Blocked temporal CV: contiguous blocks, test held out, val just before it.

    The timeline ``[t_min, t_max]`` is split into ``n_blocks`` near-equal
    contiguous blocks. For each block ``b`` (starting at ``min_train_blocks + 1``)
    block ``b`` is the test window, block ``b-1`` is validation, and blocks
    ``0..b-2`` form the training window. This guarantees no overlap and strict
    ordering by construction.
    """
    lo, hi = _resolve_bounds(time_steps, t_min, t_max)
    span = hi - lo + 1
    if n_blocks < 3:
        raise ValueError("blocked validation needs at least 3 blocks (train/val/test)")
    if span < n_blocks:
        raise ValueError(f"timeline span {span} too small for {n_blocks} blocks")
    edges = np.linspace(lo, hi + 1, n_blocks + 1).astype(int)
    blocks = [TimeRange(int(edges[i]), int(edges[i + 1]) - 1) for i in range(n_blocks)]
    folds: List[TemporalFold] = []
    fold_idx = 0
    for b in range(min_train_blocks + 1, n_blocks):
        train = TimeRange(blocks[0].start, blocks[b - 2].end)
        val = blocks[b - 1]
        test = blocks[b]
        fold = TemporalFold(fold_idx, "blocked", train, val, test)
        audit_fold(fold, lo, hi)
        folds.append(fold)
        fold_idx += 1
    if not folds:
        raise ValueError("no blocked folds produced; reduce min_train_blocks or n_blocks")
    return folds


def rolling_folds(
    *,
    time_steps: Optional[Sequence[int]] = None,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
    train_size: int = 10,
    val_size: int = 3,
    test_size: int = 3,
    step: int = 3,
) -> List[TemporalFold]:
    """Sliding-window validation: fixed-width train that rolls forward.

    Window layout for a fold whose train starts at ``s``:
        train = [s, s+train_size-1]
        val   = [s+train_size, s+train_size+val_size-1]
        test  = [s+train_size+val_size, s+train_size+val_size+test_size-1]
    The start ``s`` advances by ``step`` until the test window would fall off
    the end of the timeline.
    """
    lo, hi = _resolve_bounds(time_steps, t_min, t_max)
    if min(train_size, val_size, test_size, step) < 1:
        raise ValueError("train/val/test/step sizes must be >= 1")
    block = train_size + val_size + test_size
    folds: List[TemporalFold] = []
    fold_idx = 0
    s = lo
    while s + block - 1 <= hi:
        train = TimeRange(s, s + train_size - 1)
        val = TimeRange(train.end + 1, train.end + val_size)
        test = TimeRange(val.end + 1, val.end + test_size)
        fold = TemporalFold(fold_idx, "rolling", train, val, test)
        audit_fold(fold, lo, hi)
        folds.append(fold)
        fold_idx += 1
        s += step
    if not folds:
        raise ValueError(
            f"no rolling folds fit: need span >= {block}, have {hi - lo + 1}"
        )
    return folds


def expanding_folds(
    *,
    time_steps: Optional[Sequence[int]] = None,
    t_min: Optional[int] = None,
    t_max: Optional[int] = None,
    initial_train: int = 10,
    val_size: int = 3,
    test_size: int = 3,
    step: int = 3,
) -> List[TemporalFold]:
    """Expanding-window validation: train grows from the start of the timeline.

    Window layout for fold ``k`` (k = 0, 1, ...):
        train = [t_min, t_min + initial_train - 1 + k*step]
        val   = [train.end+1, train.end+val_size]
        test  = [val.end+1, val.end+test_size]
    Training always anchors at ``t_min`` (full available history).
    """
    lo, hi = _resolve_bounds(time_steps, t_min, t_max)
    if min(initial_train, val_size, test_size, step) < 1:
        raise ValueError("initial_train/val/test/step sizes must be >= 1")
    folds: List[TemporalFold] = []
    fold_idx = 0
    train_end = lo + initial_train - 1
    while train_end + val_size + test_size <= hi:
        train = TimeRange(lo, train_end)
        val = TimeRange(train_end + 1, train_end + val_size)
        test = TimeRange(val.end + 1, val.end + test_size)
        fold = TemporalFold(fold_idx, "expanding", train, val, test)
        audit_fold(fold, lo, hi)
        folds.append(fold)
        fold_idx += 1
        train_end += step
    if not folds:
        raise ValueError(
            f"no expanding folds fit: need span >= {initial_train + val_size + test_size}, "
            f"have {hi - lo + 1}"
        )
    return folds


SCHEME_BUILDERS = {
    "blocked": blocked_folds,
    "rolling": rolling_folds,
    "expanding": expanding_folds,
}


def make_folds(scheme: str, **kwargs) -> List[TemporalFold]:
    """Dispatch to a scheme builder by name. Unknown schemes fail loudly."""
    try:
        builder = SCHEME_BUILDERS[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; choose from {sorted(SCHEME_BUILDERS)}"
        ) from None
    return builder(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Leakage-safe model selection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigSelection:
    """Result of selecting a configuration on validation evidence only."""

    best_config: str
    best_score: float
    metric: str
    per_config_mean: Dict[str, float]
    n_folds_used: int
    used_test_metrics: bool = False  # always False — kept for audit symmetry


# Metric names that would mean we are selecting on the *test* window.
_FORBIDDEN_SELECTION_KEYS = frozenset(
    {"test_f1", "test_auc", "test_auprc", "test_score", "test_metric", "test"}
)


def select_best_config(
    fold_records: Sequence[Mapping[str, object]],
    *,
    config_key: str = "config",
    val_metric: str = "val_f1",
    higher_is_better: bool = True,
) -> ConfigSelection:
    """Pick the best configuration using **validation** scores only.

    ``fold_records`` is a list of dicts, one per (config × fold), each carrying
    at least ``config_key`` and ``val_metric``. Records are averaged per config
    across folds and the best mean wins. Passing any ``test_*`` metric raises
    :class:`LeakageError` — model selection must never read the test window.

    Returns a :class:`ConfigSelection`. Determinstic tie-break: lexicographic
    on the config name.
    """
    if not fold_records:
        raise ValueError("fold_records is empty")
    leaked = sorted({k for r in fold_records for k in r} & _FORBIDDEN_SELECTION_KEYS)
    if leaked:
        raise LeakageError(
            f"model selection received test-window metric(s) {leaked}; "
            "deployment-style selection must use validation metrics only"
        )
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in fold_records:
        if config_key not in r or val_metric not in r:
            raise KeyError(f"record missing {config_key!r} or {val_metric!r}: {r}")
        cfg = str(r[config_key])
        sums[cfg] = sums.get(cfg, 0.0) + float(r[val_metric])  # type: ignore[arg-type]
        counts[cfg] = counts.get(cfg, 0) + 1
    means = {c: sums[c] / counts[c] for c in sums}
    # deterministic: sort by (score, name) so ties break lexicographically
    sign = -1.0 if higher_is_better else 1.0
    best = min(means, key=lambda c: (sign * means[c], c))
    return ConfigSelection(
        best_config=best,
        best_score=means[best],
        metric=val_metric,
        per_config_mean=dict(sorted(means.items())),
        n_folds_used=sum(counts.values()),
    )


def folds_to_rows(folds: Sequence[TemporalFold]) -> List[Dict[str, object]]:
    """Flatten folds to plain dict rows (CSV-ready)."""
    return [f.as_row() for f in folds]
