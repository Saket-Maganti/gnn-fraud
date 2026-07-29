"""
utils/metrics.py
Evaluation metrics for fraud detection.
All metrics are computed on the fraud (illicit=1) class only.
"""

from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch_geometric.data import Data

Number = Union[int, float]


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _resolve_k(k: Number, n: int) -> int:
    """Map integer or fractional *k* to a top-*n* count capped at *n*."""
    if n <= 0:
        return 0
    if isinstance(k, float) and not float(k).is_integer() and 0.0 < k < 1.0:
        return min(n, max(0, int(np.ceil(k * n))))
    k_int = int(k)
    if k_int <= 0:
        return 0
    return min(k_int, n)


def _labeled_mask(y: np.ndarray) -> np.ndarray:
    """Drop unknown labels (``0`` in the Elliptic convention)."""
    return y != 0


def _prepare_at_k(
    y_true,
    scores,
    positive_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _to_numpy(y_true).reshape(-1)
    s = _to_numpy(scores).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError(
            f"y_true and scores length mismatch: {y.shape[0]} vs {s.shape[0]}"
        )

    keep = _labeled_mask(y)
    y = y[keep]
    s = s[keep]
    pos = y == positive_label
    return y, s, pos


def _top_k_indices(scores: np.ndarray, k_eff: int) -> np.ndarray:
    n = scores.shape[0]
    if k_eff <= 0 or n == 0:
        return np.empty(0, dtype=int)
    k_eff = min(k_eff, n)
    # Stable ranking: higher score first; break ties by original index.
    order = np.lexsort((np.arange(n), -scores))
    return order[:k_eff]


def precision_at_k(
    y_true,
    scores,
    k: Number,
    positive_label: int = 1,
) -> float:
    """Precision among the top-*k* scored labeled cases.

    *k* may be an integer count (``100``) or a fraction in ``(0, 1)``
    (``0.01`` → top 1% of labeled cases). Unknown labels (``0``) are ignored.
    """
    _, s, pos = _prepare_at_k(y_true, scores, positive_label)
    n = s.shape[0]
    k_eff = _resolve_k(k, n)
    if k_eff == 0:
        return 0.0
    top = _top_k_indices(s, k_eff)
    return float(pos[top].sum()) / k_eff


def recall_at_k(
    y_true,
    scores,
    k: Number,
    positive_label: int = 1,
) -> float:
    """Recall captured by the top-*k* scored labeled cases.

    Denominator is the total number of positives among labeled nodes.
    """
    _, s, pos = _prepare_at_k(y_true, scores, positive_label)
    n_pos = int(pos.sum())
    if n_pos == 0:
        return 0.0
    k_eff = _resolve_k(k, s.shape[0])
    if k_eff == 0:
        return 0.0
    top = _top_k_indices(s, k_eff)
    return float(pos[top].sum()) / n_pos


def precision_recall_at_ks(
    y_true,
    scores,
    ks: Sequence[Number],
    positive_label: int = 1,
) -> Dict[str, Dict[Number, float]]:
    """Compute precision and recall for several *k* values."""
    precision: Dict[Number, float] = {}
    recall: Dict[Number, float] = {}
    for k in ks:
        precision[k] = precision_at_k(y_true, scores, k, positive_label)
        recall[k] = recall_at_k(y_true, scores, k, positive_label)
    return {"precision": precision, "recall": recall}


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask:   torch.Tensor,
) -> Dict[str, float]:
    """
    Compute F1, precision, recall, accuracy on masked nodes.
    Positive class = illicit (label=1).
    """
    preds  = logits[mask].argmax(dim=-1).cpu().numpy()
    y_true = labels[mask].cpu().numpy()

    preds_bin  = (preds  == 1).astype(int)
    y_true_bin = (y_true == 1).astype(int)

    f1        = f1_score(y_true_bin,        preds_bin, zero_division=0)
    precision = precision_score(y_true_bin, preds_bin, zero_division=0)
    recall    = recall_score(y_true_bin,    preds_bin, zero_division=0)
    acc       = (preds == y_true).mean()

    # AUC-ROC on fraud probability
    probs = torch.softmax(logits[mask], dim=-1)[:, 1].detach().cpu().numpy()
    try:
        auc = roc_auc_score(y_true_bin, probs)
    except ValueError:
        auc = float("nan")

    return {
        "f1":        round(float(f1),        4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "accuracy":  round(float(acc),       4),
        "auc":       round(float(auc),       4),
    }


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 20260614,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Returns ``(low, high)`` at the requested confidence level. With fewer than
    two values the interval collapses to the point estimate. Deterministic for a
    fixed ``seed`` so reported CIs are reproducible.
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    low, high = np.percentile(means, [lo, hi])
    return float(low), float(high)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable effect-size / multiple-comparison / rank-correlation helpers
#
# These are the public, importable versions of the statistics the RUNS analyzers
# compute inline (scripts/analyze_runs_statistics.py, analyze_runs_rank_reversal.py).
# Pulled out so experiments, the paper-table exporters and tests share one
# audited implementation. All are numpy-only (no scipy dependency) for the point
# estimates; the correlation p-values use scipy when present and return ``None``
# (not a fabricated value) when it is not.
# ─────────────────────────────────────────────────────────────────────────────

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Cliff's delta effect size for ``x`` vs ``y`` (nonparametric, in [-1, 1]).

    δ = (#(x_i > y_j) − #(x_i < y_j)) / (n_x · n_y). δ > 0 means ``x`` tends to
    exceed ``y``. Robust to non-normal, small samples — the right effect size to
    pair with a Wilcoxon/Mann-Whitney test. Returns ``None`` if either input is
    empty after dropping NaNs.
    """
    a = np.asarray([v for v in x if v is not None and not np.isnan(v)], dtype=float)
    b = np.asarray([v for v in y if v is not None and not np.isnan(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    diff = a[:, None] - b[None, :]
    greater = int((diff > 0).sum())
    less = int((diff < 0).sum())
    return float((greater - less) / (a.size * b.size))


def cliffs_delta_magnitude(delta: Optional[float]) -> str:
    """Romano et al. (2006) magnitude label for a Cliff's delta value."""
    if delta is None:
        return "unavailable"
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Paired Cohen's d = mean(x − y) / std(x − y, ddof=1).

    Inputs must be the same length (seed-aligned). Returns ``None`` if fewer than
    two matched pairs survive NaN-dropping or if the paired differences have zero
    variance (d undefined).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"cohens_d_paired requires equal-length inputs: {a.shape} vs {b.shape}")
    diffs = a - b
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size < 2:
        return None
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return None
    return float(np.mean(diffs) / std)


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, list]:
    """Holm-Bonferroni step-down correction.

    Returns ``{"adjusted": [...], "reject": [...]}`` in the *original* input
    order. ``None``/NaN p-values are passed through as ``None`` (adjusted) and
    ``False`` (reject) without consuming a comparison slot. Adjusted p-values are
    monotone non-decreasing along the sorted order, as the procedure requires.
    """
    idx_p = [(i, float(p)) for i, p in enumerate(p_values)
             if p is not None and not (isinstance(p, float) and np.isnan(p))]
    adjusted: list = [None] * len(p_values)
    reject: list = [False] * len(p_values)
    if not idx_p:
        return {"adjusted": adjusted, "reject": reject}
    idx_p.sort(key=lambda t: t[1])
    m = len(idx_p)
    running = 0.0
    for rank, (orig_idx, p) in enumerate(idx_p):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[orig_idx] = running
        reject[orig_idx] = running < alpha
    return {"adjusted": adjusted, "reject": reject}


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared — matches Spearman's tie handling."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=float)
    ranks[order] = np.arange(1, a.shape[0] + 1, dtype=float)
    # Average tied ranks.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.shape[0])
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman_rank_correlation(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Spearman ρ and its p-value for two rankings/score vectors.

    Returns ``(rho, p_value)``. ``p_value`` is ``None`` when scipy is unavailable
    or n < 3 (we never invent a p-value). ``rho`` is computed numpy-only as the
    Pearson correlation of the average ranks, so it is always available.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("spearman_rank_correlation requires equal-length inputs")
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None, None
    ra, rb = _rank(a), _rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    p_value: Optional[float] = None
    if a.size >= 3:
        try:
            from scipy import stats  # type: ignore
            p_value = float(stats.spearmanr(a, b).pvalue)
        except Exception:  # noqa: BLE001
            p_value = None
    return rho, p_value


def _kendall_tau_b(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Kendall's τ-b (numpy-only): concordant−discordant over the tie-corrected
    denominator. Returns ``None`` when undefined (n < 2 or a degenerate denom).
    Single source of truth shared by :func:`kendall_tau_correlation` and
    :func:`kendall_tau_permutation_test`.
    """
    n = a.size
    if n < 2:
        return None
    concordant = discordant = 0
    ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = a[i] - a[j]
            dy = b[i] - b[j]
            s = dx * dy
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
            else:
                if dx == 0:
                    ties_x += 1
                if dy == 0:
                    ties_y += 1
    n0 = n * (n - 1) / 2
    denom = np.sqrt((n0 - ties_x) * (n0 - ties_y))
    return float((concordant - discordant) / denom) if denom > 0 else None


def kendall_tau_correlation(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Kendall's τ-b and its p-value. ``(tau, p_value)``.

    τ is computed numpy-only (concordant−discordant over the tie-corrected
    denominator); the p-value uses scipy when present and is ``None`` otherwise.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("kendall_tau_correlation requires equal-length inputs")
    if a.size < 2:
        return None, None
    tau = _kendall_tau_b(a, b)
    p_value: Optional[float] = None
    try:
        from scipy import stats  # type: ignore
        p_value = float(stats.kendalltau(a, b).pvalue)
    except Exception:  # noqa: BLE001
        p_value = None
    return tau, p_value


def kendall_tau_permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    alternative: str = "less",
    n_permutations: int = 10000,
    exact_max_n: int = 8,
    seed: int = 20260621,
) -> Dict[str, object]:
    """Permutation significance test for the Kendall-τ between two leaderboards.

    The asymptotic Kendall p-value is unreliable for the tiny N-model
    leaderboards this paper compares (often N = 4–7). This builds the null
    distribution of τ-b by permuting ``y`` against ``x`` — exactly (all N!
    permutations) when ``N <= exact_max_n``, else by Monte-Carlo with
    ``n_permutations`` draws.

    ``alternative``:
      * ``"less"``      — H1: the orderings are *anti*-correlated (reversed);
        p = P(τ_null <= τ_obs). The headline "rankings reverse" test.
      * ``"greater"``   — H1: the orderings are preserved; p = P(τ_null >= τ_obs).
      * ``"two-sided"`` — H1: τ != 0; p = P(|τ_null| >= |τ_obs|).

    Returns a dict: ``tau``, ``p_value``, ``alternative``, ``exact`` (bool),
    ``n_permutations`` (actually evaluated), ``null_mean``, ``null_std``.
    Add-one smoothing is applied to the Monte-Carlo p-value so it is never 0.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape:
        raise ValueError("kendall_tau_permutation_test requires equal-length inputs")
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    n = a.size
    tau_obs = _kendall_tau_b(a, b)
    if n < 2 or tau_obs is None:
        return {"tau": tau_obs, "p_value": None, "alternative": alternative,
                "exact": False, "n_permutations": 0, "null_mean": None, "null_std": None}

    import math
    from itertools import permutations

    exact = n <= exact_max_n and math.factorial(n) <= 200000
    null = []
    if exact:
        for perm in permutations(range(n)):
            null.append(_kendall_tau_b(a, b[list(perm)]))
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        for _ in range(n_permutations):
            null.append(_kendall_tau_b(a, b[rng.permutation(idx)]))
    null_arr = np.asarray([t for t in null if t is not None], dtype=float)
    m = null_arr.size

    if alternative == "less":
        hits = int((null_arr <= tau_obs + 1e-12).sum())
    elif alternative == "greater":
        hits = int((null_arr >= tau_obs - 1e-12).sum())
    else:
        hits = int((np.abs(null_arr) >= abs(tau_obs) - 1e-12).sum())

    if exact:
        p_value = hits / m if m else None
    else:
        p_value = (hits + 1) / (m + 1) if m else None

    return {
        "tau": float(tau_obs),
        "p_value": None if p_value is None else float(p_value),
        "alternative": alternative,
        "exact": bool(exact),
        "n_permutations": int(m),
        "null_mean": float(null_arr.mean()) if m else None,
        "null_std": float(null_arr.std()) if m else None,
    }


def _calibration_bins(
    y_true,
    probs_pos,
    n_bins: int = 15,
    positive_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Equal-width reliability bins for the positive-class probability.

    Returns ``(bin_conf, bin_acc, bin_frac, n)`` where each array has one entry
    per non-empty bin: mean predicted probability, empirical positive rate, and
    the fraction of (labeled) samples in that bin. Unknown labels (``0``) are
    dropped, matching the rest of this module.
    """
    y = _to_numpy(y_true).reshape(-1)
    p = _to_numpy(probs_pos).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true and probs length mismatch: {y.shape[0]} vs {p.shape[0]}")
    keep = _labeled_mask(y)
    y = (y[keep] == positive_label).astype(float)
    p = np.clip(p[keep], 0.0, 1.0)
    n = p.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, fracs = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        confs.append(float(p[mask].mean()))
        accs.append(float(y[mask].mean()))
        fracs.append(cnt / n)
    return np.array(confs), np.array(accs), np.array(fracs), n


def expected_calibration_error(
    y_true,
    probs_pos,
    n_bins: int = 15,
    positive_label: int = 1,
) -> float:
    """Expected Calibration Error (ECE) of the positive-class probability.

    ECE = Σ_b (n_b / N) · |acc_b − conf_b| over equal-width probability bins.
    This is the standard metric for the temperature-scaling / calibration story;
    0 is perfectly calibrated.
    """
    conf, acc, frac, n = _calibration_bins(y_true, probs_pos, n_bins, positive_label)
    if n == 0 or conf.size == 0:
        return float("nan")
    return float(np.sum(frac * np.abs(acc - conf)))


def maximum_calibration_error(
    y_true,
    probs_pos,
    n_bins: int = 15,
    positive_label: int = 1,
) -> float:
    """Maximum Calibration Error (MCE): worst-case bin gap |acc_b − conf_b|."""
    conf, acc, _, n = _calibration_bins(y_true, probs_pos, n_bins, positive_label)
    if n == 0 or conf.size == 0:
        return float("nan")
    return float(np.max(np.abs(acc - conf)))


def reliability_curve(
    y_true,
    probs_pos,
    n_bins: int = 15,
    positive_label: int = 1,
) -> Dict[str, list]:
    """Reliability-diagram data (per non-empty bin) for plotting/export."""
    conf, acc, frac, _ = _calibration_bins(y_true, probs_pos, n_bins, positive_label)
    return {
        "confidence": conf.tolist(),
        "accuracy": acc.tolist(),
        "fraction": frac.tolist(),
    }


def aggregate_seed_metrics(seed_results: list) -> Dict[str, Dict[str, float]]:
    """
    Given a list of metric dicts (one per seed), compute mean ± std plus a 95%
    bootstrap CI for the mean.
    Returns { metric_name: {"mean", "std", "ci95_low", "ci95_high", "n"} }
    """
    keys = seed_results[0].keys()
    agg  = {}
    for k in keys:
        vals = [r[k] for r in seed_results if not np.isnan(r[k])]
        low, high = bootstrap_ci(vals)
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
            "ci95_low":  round(low, 4),
            "ci95_high": round(high, 4),
            "n": len(vals),
        }
    return agg


def format_metrics_table(agg: Dict) -> str:
    """Pretty-print aggregated metrics table."""
    lines = [f"{'Metric':<12} {'Mean':>8} {'±Std':>8}"]
    lines.append("-" * 30)
    for k, v in agg.items():
        lines.append(f"{k:<12} {v['mean']:>8.4f} {v['std']:>8.4f}")
    return "\n".join(lines)


@torch.no_grad()
def full_report(model, data: Data, device: torch.device,
                model_type: str = "static"):
    """Print a detailed test-set classification report."""
    model.eval()
    data = data.to(device)

    if model_type == "evolve":
        # Build snapshots for EvolveGCN
        from utils.temporal import build_snapshots
        snapshots = build_snapshots(data, time_range=range(35, 50))
        logits = model(snapshots)
        mask   = data.test_mask
    else:
        logits = model(data.x, data.edge_index,
                       getattr(data, "edge_attr", None))
        mask   = data.test_mask

    mask   = mask.to(device)
    preds  = logits[mask].argmax(-1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    print("\n── CLASSIFICATION REPORT (test set) ──")
    print(classification_report(
        y_true, preds,
        labels=[1, 2],
        target_names=["Illicit (fraud)", "Licit (legit)"],
        digits=4,
    ))
    cm = confusion_matrix(y_true, preds, labels=[1, 2])
    tn, fp, fn, tp = cm.ravel()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Fraud Precision : {tp/(tp+fp+1e-8):.4f}")
    print(f"Fraud Recall    : {tp/(tp+fn+1e-8):.4f}")
