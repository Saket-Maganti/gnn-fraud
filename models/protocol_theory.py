"""
models/protocol_theory.py

Mechanistic theory of *protocol-induced rank reversal* under temporal shift.

This module is the **analytic backbone** of the paper's central claim —
"evaluation protocols can reverse model rankings under temporal shift". The
empirical experiments observe the reversal; this module *explains and predicts*
it from first principles, in closed form, and pins down exactly which part of
the gap the TPC+TTA solution can and cannot repair.

The model
=========
We use the classical equal-variance **binormal** model of a binary classifier
(Green & Swets 1966; Pepe 2003): for a model with separation ``mu`` (a.k.a.
d-prime / signal-to-noise), the positive-class score is

    s | y=1 ~ N(+mu/2, 1)
    s | y=0 ~ N(-mu/2, 1)

Everything a leaderboard cares about is then closed form in ``mu`` and the
class prior ``pi`` (and a threshold ``tau``):

    AUC            = Phi(mu / sqrt(2))                     # prior-free
    TPR(tau)       = Phi(mu/2 - tau)
    FPR(tau)       = Phi(-mu/2 - tau)
    precision(tau) = pi*TPR / (pi*TPR + (1-pi)*FPR)
    recall(tau)    = TPR
    F1(tau)        = 2*pi*TPR / (pi*TPR + pi + (1-pi)*FPR)

where ``Phi`` is the standard-normal CDF.

How a protocol changes a model
==============================
Each model is described by two intrinsic numbers:

  * ``feature_sep`` (a_m)        — separation from *intrinsic node features*
                                   alone (what survives with no usable graph).
  * ``structural_reliance`` (rho_m in [0,1]) — how effectively the model turns
                                   available graph structure into extra
                                   separation.

The environment supplies:

  * ``graph_info`` (G >= 0)      — total discriminative signal latent in the
                                   graph structure under a *stable, homophilous*
                                   regime (what a perfect structural reader could
                                   extract transductively).
  * ``homophily_decay`` (h in [0,1]) — fraction of the structural signal that has
                                   *decayed away* by test time (homophily erosion
                                   + loss of train<->test edges). h=0: structure
                                   as good at test time as train; h=1: structure
                                   carries no usable signal at test time.
  * ``prior`` (pi)               — training-time fraud base rate.
  * ``prior_drift`` (delta)      — change in the fraud base rate from train to
                                   test (pi_test = clip(pi + delta)).

The two protocols then map to two binormal models:

    transductive:  mu_T = a_m + rho_m * G                  # full graph available
    inductive:     mu_I = a_m + rho_m * G * (1 - h)        # structure decayed

and they differ in *calibration* too:

    transductive:  threshold fit and evaluated at prior pi  (in-distribution)
    inductive (stale): threshold fit at the *train* prior pi, then evaluated
                   under the *drifted* prior pi+delta  -- the leakage-free
                   deployment reality.
    inductive (TPC):   prior re-estimated to pi+delta and threshold re-fit for
                   that regime -- the optimum the TPC+TTA wrapper targets.

The decomposition this buys us
==============================
The transductive->inductive gap splits cleanly into two orthogonal axes:

  (1) **representation decay** (h): the structural separation a model loses.
      This is *prior-free*, so it moves AUC -- and **TPC+TTA cannot recover
      it** (no amount of post-hoc calibration restores a separation the
      backbone never produced at test time).
  (2) **calibration / prior drift** (delta): a stale threshold under a shifted
      prior. This is exactly what TPC+TTA repairs (Saerens 2002 prior
      correction + threshold re-fit).

Hence the headline theoretical result:

    * The **AUC** ranking reverses at a closed-form critical decay ``h*`` that
      depends only on the two models' (a, rho) and G -- independent of the
      prior. TPC+TTA leaves this reversal untouched.
    * The **F1** ranking reverses over a 2-D region in ``(h, delta)`` space.
      TPC+TTA shrinks that region back toward the AUC (representation-only)
      boundary, but cannot cross it.

That "shrinks-but-cannot-cross" statement is a *prediction*, validated against
simulated temporal graphs in ``experiments/run_protocol_phase_diagram.py``.

References
----------
* Green & Swets (1966); Pepe (2003) -- binormal ROC model.
* Saerens, Latinne & Decaestecker (2002) -- prior correction under label shift.
* Lipton, Wang & Smola (2018) -- Black-Box Shift Estimation.
* Storkey (2009); Moreno-Torres et al. (2012) -- a taxonomy of dataset shift.

Everything here is pure NumPy/SciPy and deterministic -- no training, no graph,
no randomness -- so it is cheap to unit-test and to overlay on any plot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

def _trapz_compat(y, x):
    """Compatibility shim for NumPy 1.x/2.x trapezoid integration."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


try:  # SciPy is a hard dep of the project; the fallback keeps import-time safe.
    from scipy.stats import norm as _norm

    def _Phi(x):
        return _norm.cdf(x)
except Exception:  # pragma: no cover - SciPy always present in this repo
    from math import erf, sqrt

    def _Phi(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2.0)))(x))


# ─────────────────────────────────────────────────────────────────────────────
# Binormal classifier algebra (closed-form metrics)
# ─────────────────────────────────────────────────────────────────────────────


def auc_from_separation(mu: float) -> float:
    """ROC-AUC of an equal-variance binormal classifier with separation ``mu``.

    ``AUC = Phi(mu / sqrt(2))``.  Monotone increasing in ``mu``; equals 0.5 at
    ``mu = 0`` and approaches 1 as ``mu -> inf``.  Prior-independent, which is
    the whole reason AUC isolates *representation* from *calibration*.
    """
    return float(_Phi(mu / np.sqrt(2.0)))


def separation_from_auc(auc: float) -> float:
    """Invert ``AUC = Phi(mu/sqrt2)`` -> ``mu = sqrt(2) * Phi^{-1}(AUC)``.

    The binormal separation implied by a measured ROC-AUC.  This is how we map a
    *real* model's measured AUROC under a protocol back onto the theory's
    separation axis (clipped off the 0.5/1 boundary for numerical safety).
    """
    try:
        from scipy.stats import norm as _n

        ppf = _n.ppf
    except Exception:  # pragma: no cover
        from statistics import NormalDist

        ppf = NormalDist().inv_cdf
    a = float(np.clip(auc, 0.5 + 1e-6, 1.0 - 1e-9))
    return float(np.sqrt(2.0) * ppf(a))


def tpr_at(mu: float, tau: float) -> float:
    """P(score >= tau | y = 1) for separation ``mu``."""
    return float(_Phi(mu / 2.0 - tau))


def fpr_at(mu: float, tau: float) -> float:
    """P(score >= tau | y = 0) for separation ``mu``."""
    return float(_Phi(-mu / 2.0 - tau))


def precision_at(mu: float, pi: float, tau: float) -> float:
    """Closed-form precision at threshold ``tau`` under prior ``pi``."""
    tpr = tpr_at(mu, tau)
    fpr = fpr_at(mu, tau)
    denom = pi * tpr + (1.0 - pi) * fpr
    if denom <= 0.0:
        return 0.0
    return float(pi * tpr / denom)


def f1_at(mu: float, pi: float, tau: float) -> float:
    """Closed-form F1 at threshold ``tau`` under prior ``pi`` (positive class).

    ``F1 = 2*pi*TPR / (pi*TPR + pi + (1-pi)*FPR)`` -- derived from
    ``2TP / (2TP + FP + FN)`` in population-rate terms.
    """
    tpr = tpr_at(mu, tau)
    fpr = fpr_at(mu, tau)
    denom = pi * tpr + pi + (1.0 - pi) * fpr
    if denom <= 0.0:
        return 0.0
    return float(2.0 * pi * tpr / denom)


_DEFAULT_TAU_GRID = np.linspace(-6.0, 6.0, 4801)  # ~2.5e-3 resolution


def best_f1_threshold(
    mu: float, pi: float, tau_grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Threshold that maximises binormal F1 at ``(mu, pi)``.

    Returns ``(tau_star, f1_star)``.  F1 is unimodal in ``tau`` for the binormal
    model, so a fine grid is exact to grid resolution and far cheaper than a
    root-find on the stationarity condition.
    """
    grid = _DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, float)
    tpr = _Phi(mu / 2.0 - grid)
    fpr = _Phi(-mu / 2.0 - grid)
    denom = pi * tpr + pi + (1.0 - pi) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * pi * tpr / denom, 0.0)
    j = int(np.argmax(f1))
    return float(grid[j]), float(f1[j])


# ─────────────────────────────────────────────────────────────────────────────
# Unequal-variance binormal extension  (robustness of the equal-variance model)
#
# Real classifier scores rarely have equal positive/negative spread, so a
# reviewer will ask whether the equal-variance assumption (sigma = 1 both
# classes) drives the reversal result. This block generalises every closed form
# to the *unequal-variance* binormal model (Pepe 2003, §4.3):
#
#     s | y=1 ~ N(+mu/2, 1)        (positive-class spread fixed at 1, the unit)
#     s | y=0 ~ N(-mu/2, sigma^2)  (negative-class spread sigma >= 0)
#
# with sigma = 1 recovering the equal-variance functions above *exactly*. Only
# the negative-class tail (FPR) and the AUC normaliser change:
#
#     AUC      = Phi(mu / sqrt(1 + sigma^2))           # was Phi(mu/sqrt(2))
#     TPR(tau) = Phi(mu/2 - tau)                       # unchanged
#     FPR(tau) = Phi(-(mu/2 + tau) / sigma)            # was Phi(-mu/2 - tau)
#
# The payoff is `auc_reversal_decay_uv`: the AUC ranking still reverses, and the
# critical decay h* moves *predictably* with the per-model score spreads — so the
# phenomenon is a property of the separation crossing, not of equal variance.
# ─────────────────────────────────────────────────────────────────────────────


def auc_from_separation_uv(mu: float, sigma_neg: float = 1.0) -> float:
    """Unequal-variance binormal AUC ``Phi(mu / sqrt(1 + sigma_neg^2))``.

    ``sigma_neg`` is the negative class's score spread (positive class fixed at
    1). ``sigma_neg = 1`` reproduces :func:`auc_from_separation` exactly.
    """
    s = max(float(sigma_neg), 1e-9)
    return float(_Phi(mu / np.sqrt(1.0 + s * s)))


def tpr_at_uv(mu: float, tau: float) -> float:
    """P(score >= tau | y = 1) — positive-class spread fixed at 1 (== :func:`tpr_at`)."""
    return float(_Phi(mu / 2.0 - tau))


def fpr_at_uv(mu: float, tau: float, sigma_neg: float = 1.0) -> float:
    """P(score >= tau | y = 0) for negative-class spread ``sigma_neg``."""
    s = max(float(sigma_neg), 1e-9)
    return float(_Phi(-(mu / 2.0 + tau) / s))


def precision_at_uv(mu: float, pi: float, tau: float, sigma_neg: float = 1.0) -> float:
    """Closed-form precision at ``tau`` under prior ``pi`` (unequal variance)."""
    tpr = tpr_at_uv(mu, tau)
    fpr = fpr_at_uv(mu, tau, sigma_neg)
    denom = pi * tpr + (1.0 - pi) * fpr
    if denom <= 0.0:
        return 0.0
    return float(pi * tpr / denom)


def f1_at_uv(mu: float, pi: float, tau: float, sigma_neg: float = 1.0) -> float:
    """Closed-form positive-class F1 at ``tau`` under prior ``pi`` (unequal variance)."""
    tpr = tpr_at_uv(mu, tau)
    fpr = fpr_at_uv(mu, tau, sigma_neg)
    denom = pi * tpr + pi + (1.0 - pi) * fpr
    if denom <= 0.0:
        return 0.0
    return float(2.0 * pi * tpr / denom)


def best_f1_threshold_uv(
    mu: float, pi: float, sigma_neg: float = 1.0, tau_grid: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Threshold maximising the unequal-variance binormal F1 at ``(mu, pi, sigma_neg)``."""
    grid = _DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, float)
    s = max(float(sigma_neg), 1e-9)
    tpr = _Phi(mu / 2.0 - grid)
    fpr = _Phi(-(mu / 2.0 + grid) / s)
    denom = pi * tpr + pi + (1.0 - pi) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * pi * tpr / denom, 0.0)
    j = int(np.argmax(f1))
    return float(grid[j]), float(f1[j])


# ─────────────────────────────────────────────────────────────────────────────
# AUPRC (average precision) — the fraud-critical ranking metric
#
# Under extreme class imbalance, AUPRC (= area under the precision–recall curve,
# == average precision) is the metric practitioners trust, and it is what the
# empirical tables report. AUC and F1 already have closed forms above; this block
# completes the analytic backbone with AUPRC so *every* downstream reversal
# analysis (detect_reversal, ranking_instability, protocol_robust_selection, the
# phase maps) can be run with ``metric="auprc"``.
#
# Two properties make AUPRC qualitatively different from AUC in this framework:
#   1. it is **prior-dependent** (a random classifier scores AUPRC = pi), so —
#      like F1 — its reversal region depends on the prior drift; but
#   2. it is **threshold-free / ranking-based**, so the TPC+TTA corrections
#      (temperature scaling + a per-class log-prior shift) are *monotone* in the
#      positive-class score and therefore **leave AUPRC unchanged**. AUPRC is a
#      representation metric: the fix repairs the operating point, not the
#      ranking — so an AUPRC reversal, like an AUC reversal, is unrepairable.
# ─────────────────────────────────────────────────────────────────────────────


def auprc_baseline(pi: float) -> float:
    """AUPRC of a random (non-discriminative) classifier == the prevalence ``pi``."""
    return float(np.clip(pi, 0.0, 1.0))


def auprc_binormal(
    mu: float, pi: float, sigma_neg: float = 1.0, tau_grid: Optional[np.ndarray] = None
) -> float:
    """Closed-form average precision (AUPRC) of the binormal classifier.

    Integrates precision over recall along the binormal PR curve traced by the
    threshold ``tau``: ``recall(tau) = TPR(tau)`` and
    ``precision(tau) = pi*TPR / (pi*TPR + (1-pi)*FPR)``. ``sigma_neg`` is the
    negative-class score spread (1 == equal variance). Reduces to the prevalence
    ``pi`` at ``mu = 0`` (no signal) and approaches 1 as ``mu -> inf``.
    """
    grid = _DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, float)
    s = max(float(sigma_neg), 1e-9)
    tpr = _Phi(mu / 2.0 - grid)                       # recall
    fpr = _Phi(-(mu / 2.0 + grid) / s)
    denom = pi * tpr + (1.0 - pi) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(denom > 0, pi * tpr / denom, 1.0)
    # Integrate precision d(recall) with recall ascending. The grid spans the
    # full recall range (tau in [-6, 6] -> recall ~ 0..1) at fine resolution.
    order = np.argsort(tpr)
    r = tpr[order]
    p = prec[order]
    ap = float(_trapz_compat(p, r))
    return float(np.clip(ap, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Model + environment descriptors
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelProfile:
    """Two-number summary of a model for the protocol theory.

    feature_sep         : a_m  -- separation from intrinsic node features alone.
    structural_reliance : rho_m in [0, 1] -- how well it exploits graph signal.
    name                : label for reporting / legends.
    """

    feature_sep: float
    structural_reliance: float
    name: str = "model"

    def __post_init__(self) -> None:
        if not (0.0 <= self.structural_reliance <= 1.0):
            raise ValueError("structural_reliance must lie in [0, 1]")
        if self.feature_sep < 0.0:
            raise ValueError("feature_sep must be non-negative")


@dataclass(frozen=True)
class Environment:
    """Protocol environment knobs.

    graph_info      : G >= 0 -- discriminative signal latent in stable structure.
    homophily_decay : h in [0, 1] -- structural signal lost by test time.
    prior           : pi in (0, 1) -- training-time fraud base rate.
    prior_drift     : delta -- change in fraud rate from train to test.
    transductive_prior_mode : "train" | "test" | "mixed" -- which prior the
        transductive protocol calibrates+scores at.  "train" (default) makes the
        transductive run fully in-distribution, isolating the inductive penalty.
    """

    graph_info: float = 1.2
    homophily_decay: float = 0.0
    prior: float = 0.1
    prior_drift: float = 0.0
    transductive_prior_mode: str = "train"

    def __post_init__(self) -> None:
        if self.graph_info < 0.0:
            raise ValueError("graph_info must be non-negative")
        if not (0.0 <= self.homophily_decay <= 1.0):
            raise ValueError("homophily_decay must lie in [0, 1]")
        if not (0.0 < self.prior < 1.0):
            raise ValueError("prior must lie in (0, 1)")
        if self.transductive_prior_mode not in ("train", "test", "mixed"):
            raise ValueError("transductive_prior_mode must be train|test|mixed")

    @property
    def test_prior(self) -> float:
        """Drifted test-time prior, clipped to a safe open interval."""
        return float(np.clip(self.prior + self.prior_drift, 1e-4, 1.0 - 1e-4))

    @property
    def transductive_prior(self) -> float:
        if self.transductive_prior_mode == "train":
            return float(self.prior)
        if self.transductive_prior_mode == "test":
            return self.test_prior
        return float(np.clip(0.5 * (self.prior + self.test_prior), 1e-4, 1.0 - 1e-4))


# ─────────────────────────────────────────────────────────────────────────────
# Protocol -> separation -> metrics
# ─────────────────────────────────────────────────────────────────────────────


def transductive_separation(model: ModelProfile, env: Environment) -> float:
    """``mu_T = a + rho * G`` -- full graph structure available."""
    return float(model.feature_sep + model.structural_reliance * env.graph_info)


def inductive_separation(model: ModelProfile, env: Environment) -> float:
    """``mu_I = a + rho * G * (1 - h)`` -- structure decayed by test time."""
    return float(
        model.feature_sep
        + model.structural_reliance * env.graph_info * (1.0 - env.homophily_decay)
    )


@dataclass(frozen=True)
class ProtocolMetrics:
    """All metrics for one model under one environment, both protocols."""

    model: str
    mu_transductive: float
    mu_inductive: float
    auc_transductive: float
    auc_inductive: float
    auprc_transductive: float
    auprc_inductive: float
    f1_transductive: float
    f1_inductive_stale: float
    f1_inductive_tpc: float
    tau_transductive: float
    tau_inductive_stale: float
    tau_inductive_tpc: float

    def metric(self, name: str, protocol: str) -> float:
        """Generic accessor: ``metric("auc", "inductive_stale")`` etc.

        AUC and AUPRC are ranking metrics with a single inductive value (the
        TPC+TTA corrections are monotone in the score, so they cannot change a
        ranking metric); the inductive_stale / inductive_tpc protocol variants
        are aliased to that single value. F1 keeps its three distinct variants.
        """
        key = f"{name}_{protocol}"
        # Ranking metrics (auc, auprc) have a single inductive value.
        if name in ("auc", "auprc") and protocol.startswith("inductive"):
            key = f"{name}_inductive"
        if name in ("auc", "auprc") and protocol == "transductive":
            key = f"{name}_transductive"
        if not hasattr(self, key):
            raise ValueError(f"no metric {name!r} for protocol {protocol!r}")
        return float(getattr(self, key))


def evaluate_model(model: ModelProfile, env: Environment) -> ProtocolMetrics:
    """Closed-form metrics for ``model`` under ``env`` across both protocols.

    The three inductive F1 variants encode the calibration story:

      * ``f1_inductive_stale`` -- threshold fit on the *validation window*, which
        in the inductive protocol is composed of **train-era** nodes whose graph
        structure has not yet decayed (separation ~ ``mu_T``) at the *train*
        prior, then deployed under the decayed separation ``mu_I`` and *drifted*
        prior ``pi_test``.  This is the leakage-free deployment number, and it is
        precisely why the structure-reliant model -- whose separation collapses
        most from ``mu_T`` to ``mu_I`` -- carries the most-miscalibrated
        threshold.
      * ``f1_inductive_tpc``   -- prior re-estimated to ``pi_test`` and threshold
        re-fit for the *deployed* regime ``(mu_I, pi_test)``: the optimum TPC+TTA
        targets.  Same (decayed) separation, repaired calibration.

    The split is the crux of the theory: ``stale`` carries both a *representation*
    penalty (``mu_T -> mu_I``) and a *calibration* penalty (stale threshold +
    prior drift); ``tpc`` removes the calibration penalty only.  What remains in
    ``tpc`` is representation-limited and AUC-aligned -- TPC can shrink the
    reversal region but never cross the AUC boundary.
    """
    mu_t = transductive_separation(model, env)
    mu_i = inductive_separation(model, env)

    pi_t = env.transductive_prior
    pi_train = env.prior
    pi_test = env.test_prior

    tau_t, f1_t = best_f1_threshold(mu_t, pi_t)
    # Inductive stale: threshold calibrated on the train-era validation window
    # (structure intact -> separation ~ mu_T, prior pi_train), then deployed on
    # the decayed separation mu_I under the drifted prior pi_test.
    tau_stale, _ = best_f1_threshold(mu_t, pi_train)
    f1_stale = f1_at(mu_i, pi_test, tau_stale)
    # Inductive TPC: re-estimate prior and re-fit tau for the deployed regime.
    tau_tpc, f1_tpc = best_f1_threshold(mu_i, pi_test)

    # AUPRC is threshold-free (ranking) but prior-dependent: evaluate it at each
    # protocol's separation and deployment prior. There is a single inductive
    # value because the TPC corrections cannot move a ranking metric.
    auprc_t = auprc_binormal(mu_t, pi_t)
    auprc_i = auprc_binormal(mu_i, pi_test)

    return ProtocolMetrics(
        model=model.name,
        mu_transductive=mu_t,
        mu_inductive=mu_i,
        auc_transductive=auc_from_separation(mu_t),
        auc_inductive=auc_from_separation(mu_i),
        auprc_transductive=auprc_t,
        auprc_inductive=auprc_i,
        f1_transductive=f1_t,
        f1_inductive_stale=f1_stale,
        f1_inductive_tpc=f1_tpc,
        tau_transductive=tau_t,
        tau_inductive_stale=tau_stale,
        tau_inductive_tpc=tau_tpc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rank reversal: detection + closed-form boundaries
# ─────────────────────────────────────────────────────────────────────────────


def _leader(va: float, vb: float, eps: float = 1e-9) -> int:
    """Return +1 if A leads, -1 if B leads, 0 if tied within ``eps``."""
    if va - vb > eps:
        return 1
    if vb - va > eps:
        return -1
    return 0


@dataclass(frozen=True)
class ReversalVerdict:
    """Whether the A-vs-B ranking reverses between two protocols, and by how much."""

    metric: str
    transductive_leader: str           # model name or "tie"
    inductive_leader: str              # model name or "tie"
    reversed: bool
    transductive_margin: float         # value_A - value_B (transductive)
    inductive_margin: float            # value_A - value_B (inductive)


def detect_reversal(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    metric: str = "f1",
    inductive_protocol: str = "inductive_stale",
) -> ReversalVerdict:
    """Detect whether the ``A`` vs ``B`` ranking flips transductive->inductive.

    ``metric`` is ``"auc"``, ``"auprc"`` or ``"f1"``; ``inductive_protocol`` is
    ``"inductive_stale"`` (deployment) or ``"inductive_tpc"`` (post-fix). For the
    ranking metrics (auc, auprc) the two inductive variants coincide. A reversal
    requires a *strict* leader under each protocol and the two leaders to differ.
    """
    ma = evaluate_model(model_a, env)
    mb = evaluate_model(model_b, env)

    t_a, t_b = ma.metric(metric, "transductive"), mb.metric(metric, "transductive")
    i_a, i_b = ma.metric(metric, inductive_protocol), mb.metric(metric, inductive_protocol)

    lead_t = _leader(t_a, t_b)
    lead_i = _leader(i_a, i_b)
    name = {1: model_a.name, -1: model_b.name, 0: "tie"}
    reversed_ = lead_t != 0 and lead_i != 0 and lead_t != lead_i
    return ReversalVerdict(
        metric=metric,
        transductive_leader=name[lead_t],
        inductive_leader=name[lead_i],
        reversed=reversed_,
        transductive_margin=float(t_a - t_b),
        inductive_margin=float(i_a - i_b),
    )


def auc_reversal_decay(
    model_a: ModelProfile, model_b: ModelProfile, env: Environment
) -> Optional[float]:
    """Closed-form critical homophily decay ``h*`` where the AUC ranking flips.

    AUC is monotone in separation, so the AUC ranking flips exactly when the
    inductive separations cross: ``mu_A^I = mu_B^I``.  Solving

        a_A + rho_A G (1-h) = a_B + rho_B G (1-h)

    gives ``1 - h* = (a_A - a_B) / ((rho_B - rho_A) G)``.  Returns ``h*`` if it
    lies in ``[0, 1]`` (a reachable reversal) and ``None`` otherwise (the models
    never cross within the decay range, e.g. equal structural reliance).
    """
    da = model_a.feature_sep - model_b.feature_sep
    drho = model_b.structural_reliance - model_a.structural_reliance
    G = env.graph_info
    if abs(drho) < 1e-12 or G <= 0.0:
        return None
    one_minus_h = da / (drho * G)
    h_star = 1.0 - one_minus_h
    if 0.0 <= h_star <= 1.0:
        return float(h_star)
    return None


def auc_reversal_decay_uv(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    sigma_a: float = 1.0,
    sigma_b: float = 1.0,
) -> Optional[float]:
    """Critical AUC-reversal decay ``h*`` under *unequal-variance* scores.

    Generalises :func:`auc_reversal_decay` to per-model negative-class score
    spreads ``sigma_a``/``sigma_b``. With ``k_m = 1/sqrt(1 + sigma_m^2)`` the AUC
    of model ``m`` is ``Phi(k_m * mu_m^I)``, so (AUC monotone in its argument) the
    ranking flips where ``k_A mu_A^I = k_B mu_B^I``. Substituting
    ``mu_m^I = a_m + rho_m G (1-h)`` and solving for ``h``:

        1 - h* = (k_B a_B - k_A a_A) / (G (k_A rho_A - k_B rho_B)).

    Returns ``h*`` if it lands in ``[0, 1]``, else ``None`` (no reachable
    crossing). ``sigma_a = sigma_b = 1`` reproduces :func:`auc_reversal_decay`
    exactly — i.e. the reversal is driven by the separation crossing, and unequal
    variance only *shifts* the boundary rather than creating or removing it.
    """
    G = env.graph_info
    if G <= 0.0:
        return None
    ka = 1.0 / np.sqrt(1.0 + max(float(sigma_a), 1e-9) ** 2)
    kb = 1.0 / np.sqrt(1.0 + max(float(sigma_b), 1e-9) ** 2)
    denom = G * (ka * model_a.structural_reliance - kb * model_b.structural_reliance)
    if abs(denom) < 1e-12:
        return None
    one_minus_h = (kb * model_b.feature_sep - ka * model_a.feature_sep) / denom
    h_star = 1.0 - one_minus_h
    if 0.0 <= h_star <= 1.0:
        return float(h_star)
    return None


def auprc_reversal_decay(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    n_scan: int = 257,
) -> Optional[float]:
    """Critical homophily decay ``h*`` where the *AUPRC* ranking flips.

    AUPRC has no single-line closed form for ``h*`` (it is prior-dependent and
    the PR integral is non-linear in the separation), so this scans ``h`` in
    ``[0, 1]`` for the first sign change of the inductive AUPRC margin
    ``AUPRC_A^I(h) - AUPRC_B^I(h)`` and returns a linearly-interpolated crossing.
    Returns ``None`` when the AUPRC leaderboard never crosses over the decay
    range. Holds the other environment knobs (prior, drift) at ``env``.
    """
    hs = np.linspace(0.0, 1.0, int(max(3, n_scan)))
    margins = []
    for h in hs:
        e = replace(env, homophily_decay=float(h))
        ma = evaluate_model(model_a, e).auprc_inductive
        mb = evaluate_model(model_b, e).auprc_inductive
        margins.append(ma - mb)
    margins = np.asarray(margins)
    for k in range(1, margins.size):
        a0, a1 = margins[k - 1], margins[k]
        if a0 == 0.0:
            return float(hs[k - 1])
        if a0 * a1 < 0.0:
            # linear interpolation of the zero crossing between hs[k-1], hs[k]
            frac = a0 / (a0 - a1)
            return float(hs[k - 1] + frac * (hs[k] - hs[k - 1]))
    return None


def reversal_phase_map(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env_base: Environment,
    h_grid: np.ndarray,
    delta_grid: np.ndarray,
    metric: str = "f1",
) -> Dict[str, np.ndarray]:
    """Analytic reversal classification over an ``(h, delta)`` grid.

    Returns a dict of ``len(delta_grid) x len(h_grid)`` integer arrays
    (``delta`` indexes rows, ``h`` indexes columns), each cell in:

        0 -- no reversal under either protocol,
        1 -- reversal under the stale inductive protocol that **TPC repairs**
             (reversed stale, not reversed after TPC),
        2 -- reversal that **persists after TPC** (representation-limited).

    ``status`` is the combined code above; ``stale`` / ``tpc`` are the raw
    boolean reversal masks for each protocol.  This is the data the phase-diagram
    figure overlays its analytic boundary on.
    """
    H = np.asarray(h_grid, float)
    D = np.asarray(delta_grid, float)
    stale = np.zeros((D.size, H.size), dtype=int)
    tpc = np.zeros((D.size, H.size), dtype=int)
    for i, d in enumerate(D):
        for j, h in enumerate(H):
            env = replace(env_base, homophily_decay=float(h), prior_drift=float(d))
            stale[i, j] = int(
                detect_reversal(model_a, model_b, env, metric, "inductive_stale").reversed
            )
            tpc[i, j] = int(
                detect_reversal(model_a, model_b, env, metric, "inductive_tpc").reversed
            )
    status = np.where(stale & ~tpc.astype(bool), 1, 0)
    status = np.where(tpc.astype(bool), 2, status)
    return {"status": status, "stale": stale, "tpc": tpc, "h_grid": H, "delta_grid": D}


# ─────────────────────────────────────────────────────────────────────────────
# N-model leaderboard instability  (generalises the pairwise reversal)
# ─────────────────────────────────────────────────────────────────────────────


def _rank_correlations(t_vals, i_vals) -> Tuple[float, float]:
    """Kendall tau-b and Spearman rho between two value vectors.

    Higher metric == better, so we correlate the raw values (ties handled by the
    -b / average-rank conventions).  All-tied inputs -> perfectly stable (1.0).
    """
    t = np.asarray(t_vals, float)
    i = np.asarray(i_vals, float)
    if t.size < 2 or np.allclose(t, t[0]) or np.allclose(i, i[0]):
        return 1.0, 1.0
    try:
        from scipy.stats import kendalltau, spearmanr

        tau = kendalltau(t, i).correlation
        rho = spearmanr(t, i).correlation
    except Exception:  # pragma: no cover
        tau = rho = float(np.corrcoef(t, i)[0, 1])
    tau = 1.0 if tau is None or np.isnan(tau) else float(tau)
    rho = 1.0 if rho is None or np.isnan(rho) else float(rho)
    return tau, rho


def leaderboard_values(
    profiles: List["ModelProfile"], env: "Environment", metric: str, protocol: str
) -> List[float]:
    """Metric value for every profile under one protocol (aligned with ``profiles``)."""
    return [evaluate_model(p, env).metric(metric, protocol) for p in profiles]


@dataclass(frozen=True)
class LeaderboardInstability:
    """How much an N-model leaderboard reorders between two protocols."""

    metric: str
    inductive_protocol: str
    kendall_tau: float                 # +1 identical order ... -1 fully inverted
    spearman_rho: float
    n_discordant_pairs: int            # model pairs whose order flips
    n_pairs: int
    top1_preserved: bool               # does the #1 model survive the protocol?
    transductive_order: List[str]      # best -> worst
    inductive_order: List[str]


def ranking_instability(
    profiles: List["ModelProfile"],
    env: "Environment",
    metric: str = "f1",
    inductive_protocol: str = "inductive_stale",
) -> LeaderboardInstability:
    """Leaderboard reordering between the transductive and an inductive protocol.

    Generalises :func:`detect_reversal` (a 2-model special case) to a full
    leaderboard: the Kendall tau between the transductive and inductive metric
    vectors is the *ranking stability* (1 = leaderboard preserved, -1 = fully
    inverted), and ``n_discordant_pairs`` counts the model pairs that flip.
    """
    names = [p.name for p in profiles]
    t_vals = leaderboard_values(profiles, env, metric, "transductive")
    i_vals = leaderboard_values(profiles, env, metric, inductive_protocol)
    tau, rho = _rank_correlations(t_vals, i_vals)

    n = len(profiles)
    disc = 0
    for a in range(n):
        for b in range(a + 1, n):
            st = _leader(t_vals[a], t_vals[b])
            si = _leader(i_vals[a], i_vals[b])
            if st != 0 and si != 0 and st != si:
                disc += 1
    t_order = [names[k] for k in np.argsort(t_vals)[::-1]]
    i_order = [names[k] for k in np.argsort(i_vals)[::-1]]
    return LeaderboardInstability(
        metric=metric,
        inductive_protocol=inductive_protocol,
        kendall_tau=tau,
        spearman_rho=rho,
        n_discordant_pairs=disc,
        n_pairs=n * (n - 1) // 2,
        top1_preserved=(t_order[0] == i_order[0]),
        transductive_order=t_order,
        inductive_order=i_order,
    )


def leaderboard_stability_map(
    profiles: List["ModelProfile"],
    env_base: "Environment",
    h_grid: np.ndarray,
    delta_grid: np.ndarray,
    metric: str = "f1",
    inductive_protocol: str = "inductive_stale",
) -> Dict[str, np.ndarray]:
    """Kendall-tau ranking-stability surface over an ``(h, delta)`` grid.

    Returns ``kendall_tau`` and ``n_discordant`` arrays of shape
    ``[len(delta_grid), len(h_grid)]`` plus the grids -- the data the
    leaderboard-instability figure renders.
    """
    H = np.asarray(h_grid, float)
    D = np.asarray(delta_grid, float)
    tau = np.ones((D.size, H.size))
    disc = np.zeros((D.size, H.size), dtype=int)
    for i, d in enumerate(D):
        for j, h in enumerate(H):
            env = replace(env_base, homophily_decay=float(h), prior_drift=float(d))
            inst = ranking_instability(profiles, env, metric, inductive_protocol)
            tau[i, j] = inst.kendall_tau
            disc[i, j] = inst.n_discordant_pairs
    return {"kendall_tau": tau, "n_discordant": disc, "h_grid": H, "delta_grid": D}


# ─────────────────────────────────────────────────────────────────────────────
# Theory-driven protocol-robust model selection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SelectionVerdict:
    """Naive (transductive-leaderboard) vs deployment-robust model selection."""

    metric: str
    inductive_protocol: str
    transductive_pick: str             # what naive leaderboard selection chooses
    deployment_pick: str               # what you should deploy (robust choice)
    selection_reversed: bool           # do they differ?
    naive_deployment_value: float      # the naive pick's *deployment* metric
    best_deployment_value: float       # the robust pick's deployment metric
    selection_regret: float            # best - naive (>= 0): cost of naive selection
    rank_shift: Dict[str, int]         # per model: transductive_rank - inductive_rank
    reversal_risks: List[str]          # models that fell at deployment (rank_shift < 0)


def protocol_robust_selection(
    profiles: List["ModelProfile"],
    env: "Environment",
    metric: str = "f1",
    inductive_protocol: str = "inductive_stale",
) -> SelectionVerdict:
    """Pick the deployment-optimal model, and quantify naive-selection regret.

    The paper's practical payoff: do **not** select on the transductive
    leaderboard.  This returns the transductive winner (what naive selection
    picks), the inductive/deployment winner (the robust choice), and the
    ``selection_regret`` -- the deployment-metric gap you pay by trusting the
    transductive leaderboard.  ``rank_shift[m] = transductive_rank - inductive_rank``
    (positive => the model *rose* at deployment; negative => it *fell*).
    """
    names = [p.name for p in profiles]
    t_vals = leaderboard_values(profiles, env, metric, "transductive")
    i_vals = leaderboard_values(profiles, env, metric, inductive_protocol)

    t_pick = int(np.argmax(t_vals))
    d_pick = int(np.argmax(i_vals))
    t_rank = {names[k]: r + 1 for r, k in enumerate(np.argsort(t_vals)[::-1])}
    i_rank = {names[k]: r + 1 for r, k in enumerate(np.argsort(i_vals)[::-1])}
    shift = {nm: int(t_rank[nm] - i_rank[nm]) for nm in names}
    risks = sorted((nm for nm in names if shift[nm] < 0), key=lambda nm: shift[nm])
    regret = float(i_vals[d_pick] - i_vals[t_pick])
    return SelectionVerdict(
        metric=metric,
        inductive_protocol=inductive_protocol,
        transductive_pick=names[t_pick],
        deployment_pick=names[d_pick],
        selection_reversed=(names[t_pick] != names[d_pick]),
        naive_deployment_value=float(i_vals[t_pick]),
        best_deployment_value=float(i_vals[d_pick]),
        selection_regret=max(regret, 0.0),
        rank_shift=shift,
        reversal_risks=risks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# (1) Finite-sample reversal detectability  (theory <-> empirics bridge)
#
# A leaderboard reversal is measured from *finitely many seeds*. This block
# answers, in closed form: given the per-seed variance, is an observed reversal
# statistically real, and how many seeds would a claim need? It is the power /
# minimum-detectable-effect (MDE) analysis specialised to the two-margin reversal
# test, and it pairs with utils.metrics.kendall_tau_permutation_test.
# ─────────────────────────────────────────────────────────────────────────────


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(min(max(p, 1e-12), 1.0 - 1e-12)))


def _norm_sf(z: float) -> float:
    from statistics import NormalDist
    return float(1.0 - NormalDist().cdf(z))


def minimum_detectable_effect(
    sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True
) -> float:
    """Smallest paired margin detectable at significance ``alpha`` and ``power``.

    ``MDE = (z_{1-alpha[/2]} + z_{power}) * sd_diff / sqrt(n)`` (normal approx).
    ``sd_diff`` is the standard deviation of the per-seed *paired* metric
    difference. Returns ``inf`` for degenerate inputs (no power with n<1).
    """
    if n < 1 or sd_diff <= 0:
        return float("inf") if sd_diff > 0 else 0.0
    za = _norm_ppf(1 - alpha / 2) if two_sided else _norm_ppf(1 - alpha)
    zb = _norm_ppf(power)
    return float((za + zb) * sd_diff / np.sqrt(n))


def seeds_required(
    effect: float, sd_diff: float, alpha: float = 0.05, power: float = 0.8,
    two_sided: bool = True,
) -> Optional[int]:
    """Seeds needed to detect a paired margin of size ``effect`` at ``power``.

    Inverts :func:`minimum_detectable_effect`: ``n = ((z_a + z_b) sd / effect)^2``,
    rounded up. Returns ``None`` if ``effect`` is zero (undetectable at any n).
    """
    if effect == 0.0:
        return None
    if sd_diff <= 0:
        return 1
    import math
    za = _norm_ppf(1 - alpha / 2) if two_sided else _norm_ppf(1 - alpha)
    zb = _norm_ppf(power)
    return int(math.ceil(((za + zb) * sd_diff / abs(effect)) ** 2))


@dataclass(frozen=True)
class ReversalPowerVerdict:
    """Whether a finitely-sampled rank reversal is statistically detectable."""

    transductive_margin: float
    inductive_margin: float
    transductive_p: float
    inductive_p: float
    transductive_significant: bool
    inductive_significant: bool
    reversal_observed: bool         # opposite-signed margins (point estimate)
    reversal_detectable: bool       # opposite-signed AND both individually significant
    n_seeds: int
    mde: float                      # min detectable margin at this n/power
    seeds_for_reversal: Optional[int]  # n so BOTH margins reach significance


def reversal_detectability(
    margin_trans: float,
    margin_ind: float,
    sd_trans: float,
    sd_ind: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> ReversalPowerVerdict:
    """Is an observed transductive->inductive reversal real given seed noise?

    ``margin_*`` are the per-seed *paired* mean margins (model A minus model B)
    under each protocol; ``sd_*`` their per-seed standard deviations; ``n`` the
    seed count. Each margin is z-tested against 0; a reversal is *detectable*
    when the two margins are opposite-signed and both individually significant.
    ``seeds_for_reversal`` is the n at which both would reach significance.
    """
    def _p(margin: float, sd: float) -> float:
        se = sd / np.sqrt(n) if n > 0 else 0.0
        if se <= 0:
            return 0.0 if abs(margin) > 0 else 1.0
        return float(2.0 * _norm_sf(abs(margin) / se))

    p_t = _p(margin_trans, sd_trans)
    p_i = _p(margin_ind, sd_ind)
    sig_t = p_t < alpha
    sig_i = p_i < alpha
    observed = (margin_trans > 0) != (margin_ind > 0) and margin_trans != 0 and margin_ind != 0
    detectable = observed and sig_t and sig_i
    mde = minimum_detectable_effect(max(sd_trans, sd_ind), n, alpha, power)
    need: Optional[int] = None
    if observed:
        nt = seeds_required(margin_trans, sd_trans, alpha, power)
        ni = seeds_required(margin_ind, sd_ind, alpha, power)
        cands = [x for x in (nt, ni) if x is not None]
        need = max(cands) if cands else None
    return ReversalPowerVerdict(
        transductive_margin=float(margin_trans),
        inductive_margin=float(margin_ind),
        transductive_p=p_t,
        inductive_p=p_i,
        transductive_significant=sig_t,
        inductive_significant=sig_i,
        reversal_observed=observed,
        reversal_detectable=detectable,
        n_seeds=int(n),
        mde=float(mde),
        seeds_for_reversal=need,
    )


def reversal_detectability_from_samples(
    trans_a, trans_b, ind_a, ind_b, alpha: float = 0.05, power: float = 0.8
) -> ReversalPowerVerdict:
    """Practical entry: per-seed (seed-aligned) metric samples for A and B.

    Computes the paired margins ``mean(A - B)`` and their seed SDs under each
    protocol, then defers to :func:`reversal_detectability`. Arrays must be
    seed-aligned and equal length.
    """
    ta = np.asarray(trans_a, float); tb = np.asarray(trans_b, float)
    ia = np.asarray(ind_a, float); ib = np.asarray(ind_b, float)
    if not (ta.shape == tb.shape == ia.shape == ib.shape):
        raise ValueError("all four sample arrays must be the same (seed-aligned) length")
    n = ta.size
    dt = ta - tb
    di = ia - ib
    sd_t = float(np.std(dt, ddof=1)) if n > 1 else 0.0
    sd_i = float(np.std(di, ddof=1)) if n > 1 else 0.0
    return reversal_detectability(float(dt.mean()), float(di.mean()), sd_t, sd_i, n, alpha, power)


# ─────────────────────────────────────────────────────────────────────────────
# (2) TPC+TTA optimality / regret bound
#
# Formalises the "shrinks but cannot cross" claim as a provable two-part split of
# the deployment F1 gap (both parts >= 0 by construction):
#   * representation_gap  = best-F1 at the un-decayed vs decayed separation, at
#     the SAME (deployment) prior -> the AUC/AUPRC-boundary part TPC cannot touch;
#   * calibration_regret  = best-F1 at the deployed regime minus the stale-
#     threshold F1 -> exactly what TPC recovers, and >= 0 because a re-fit
#     threshold is the argmax over thresholds. TPC therefore weakly dominates the
#     stale operating point and drives its own operating-point regret to zero.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TPCRegretBound:
    """Provable decomposition of one model's deployment F1 gap under TPC+TTA."""

    model: str
    f1_transductive: float            # optimistic-protocol number (mu_T, pi_t)
    f1_ceiling_at_test_prior: float   # best-F1 at (mu_T, pi_test): no-decay ceiling
    f1_inductive_stale: float         # deployed with the stale threshold
    f1_inductive_tpc: float           # deployed with a re-fit threshold (TPC)
    representation_gap: float         # ceiling - tpc  (>= 0, irreducible)
    calibration_regret: float         # tpc - stale    (>= 0, recovered by TPC)
    tpc_dominates_stale: bool         # calibration_regret >= 0 (provably true)
    residual_gap_after_tpc: float     # == representation_gap


def tpc_regret_bound(model: ModelProfile, env: Environment) -> TPCRegretBound:
    """Closed-form TPC+TTA regret decomposition for ``model`` under ``env``.

    Both ``representation_gap`` and ``calibration_regret`` are non-negative by
    construction (best-F1 is monotone in separation at a fixed prior, and a
    re-fit threshold is the per-regime argmax), so TPC weakly dominates the stale
    operating point and the only irreducible loss is the representation gap — the
    formal version of "TPC shrinks the reversal region but cannot cross the
    AUC/AUPRC boundary".
    """
    mu_t = transductive_separation(model, env)
    mu_i = inductive_separation(model, env)
    pi_t = env.transductive_prior
    pi_train = env.prior
    pi_test = env.test_prior

    _, f1_trans = best_f1_threshold(mu_t, pi_t)
    _, f1_ceiling = best_f1_threshold(mu_t, pi_test)
    tau_stale, _ = best_f1_threshold(mu_t, pi_train)
    f1_stale = f1_at(mu_i, pi_test, tau_stale)
    _, f1_tpc = best_f1_threshold(mu_i, pi_test)

    rep = float(f1_ceiling - f1_tpc)
    cal = float(f1_tpc - f1_stale)
    return TPCRegretBound(
        model=model.name,
        f1_transductive=float(f1_trans),
        f1_ceiling_at_test_prior=float(f1_ceiling),
        f1_inductive_stale=float(f1_stale),
        f1_inductive_tpc=float(f1_tpc),
        representation_gap=rep,
        calibration_regret=cal,
        tpc_dominates_stale=bool(cal >= -1e-9),
        residual_gap_after_tpc=rep,
    )


# ─────────────────────────────────────────────────────────────────────────────
# (3) Cost-sensitive (expected-cost) reversal
#
# Fraud detection is asymmetric: a missed fraud (FN) and a false alarm (FP) carry
# different costs. The expected per-instance cost at threshold tau is
#     C(tau) = pi (1 - TPR) c_fn + (1 - pi) FPR c_fp
# (cost of missed positives + cost of false positives). Lower is better, so the
# reversal logic is mirrored. This is often the true deployment objective.
# ─────────────────────────────────────────────────────────────────────────────


def expected_cost_at(
    mu: float, pi: float, tau: float, c_fp: float = 1.0, c_fn: float = 1.0,
    sigma_neg: float = 1.0,
) -> float:
    """Expected per-instance cost at threshold ``tau`` (lower is better)."""
    tpr = tpr_at_uv(mu, tau)
    fpr = fpr_at_uv(mu, tau, sigma_neg)
    return float(pi * (1.0 - tpr) * c_fn + (1.0 - pi) * fpr * c_fp)


def best_cost_threshold(
    mu: float, pi: float, c_fp: float = 1.0, c_fn: float = 1.0, sigma_neg: float = 1.0,
    tau_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Threshold minimising expected cost at ``(mu, pi, c_fp, c_fn)``."""
    grid = _DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, float)
    s = max(float(sigma_neg), 1e-9)
    tpr = _Phi(mu / 2.0 - grid)
    fpr = _Phi(-(mu / 2.0 + grid) / s)
    cost = pi * (1.0 - tpr) * c_fn + (1.0 - pi) * fpr * c_fp
    j = int(np.argmin(cost))
    return float(grid[j]), float(cost[j])


@dataclass(frozen=True)
class CostMetrics:
    """Expected-cost analogue of :class:`ProtocolMetrics` (lower is better)."""

    model: str
    cost_transductive: float
    cost_inductive_stale: float
    cost_inductive_tpc: float

    def metric(self, protocol: str) -> float:
        key = {"transductive": "cost_transductive",
               "inductive_stale": "cost_inductive_stale",
               "inductive_tpc": "cost_inductive_tpc"}.get(protocol)
        if key is None:
            raise ValueError(f"unknown protocol {protocol!r}")
        return float(getattr(self, key))


def evaluate_cost(
    model: ModelProfile, env: Environment, c_fp: float = 1.0, c_fn: float = 5.0
) -> CostMetrics:
    """Expected cost under both protocols (default c_fn=5: a miss costs 5x a FP)."""
    mu_t = transductive_separation(model, env)
    mu_i = inductive_separation(model, env)
    pi_t = env.transductive_prior
    pi_train = env.prior
    pi_test = env.test_prior

    _, cost_t = best_cost_threshold(mu_t, pi_t, c_fp, c_fn)
    tau_stale, _ = best_cost_threshold(mu_t, pi_train, c_fp, c_fn)
    cost_stale = expected_cost_at(mu_i, pi_test, tau_stale, c_fp, c_fn)
    _, cost_tpc = best_cost_threshold(mu_i, pi_test, c_fp, c_fn)
    return CostMetrics(
        model=model.name,
        cost_transductive=float(cost_t),
        cost_inductive_stale=float(cost_stale),
        cost_inductive_tpc=float(cost_tpc),
    )


@dataclass(frozen=True)
class CostReversalVerdict:
    """A->B expected-cost reversal (leader = the *lower*-cost model)."""

    c_fp: float
    c_fn: float
    inductive_protocol: str
    transductive_leader: str
    inductive_leader: str
    reversed: bool
    transductive_cost_margin: float    # cost_A - cost_B (transductive)
    inductive_cost_margin: float       # cost_A - cost_B (inductive)


def cost_reversal(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    inductive_protocol: str = "inductive_stale",
) -> CostReversalVerdict:
    """Detect whether the cheaper model flips between protocols (lower cost wins)."""
    ca = evaluate_cost(model_a, env, c_fp, c_fn)
    cb = evaluate_cost(model_b, env, c_fp, c_fn)
    t_a, t_b = ca.metric("transductive"), cb.metric("transductive")
    i_a, i_b = ca.metric(inductive_protocol), cb.metric(inductive_protocol)
    # lower cost is better -> leader is the smaller; reuse _leader on negated cost.
    lead_t = _leader(-t_a, -t_b)
    lead_i = _leader(-i_a, -i_b)
    name = {1: model_a.name, -1: model_b.name, 0: "tie"}
    return CostReversalVerdict(
        c_fp=float(c_fp),
        c_fn=float(c_fn),
        inductive_protocol=inductive_protocol,
        transductive_leader=name[lead_t],
        inductive_leader=name[lead_i],
        reversed=(lead_t != 0 and lead_i != 0 and lead_t != lead_i),
        transductive_cost_margin=float(t_a - t_b),
        inductive_cost_margin=float(i_a - i_b),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (4) Protocol-robustness certificate  (certified, worst-case selection)
#
# Given a model's profile and an *assumed bound* on the unknown deployment decay
# (h <= h_max) and prior drift (|delta| <= delta_max), this returns a GUARANTEED
# lower bound on its deployment metric — the worst case over that box — turning
# the descriptive reversal analysis into a robust-optimisation (maximin) model
# selection tool with a deployment guarantee.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RobustnessCertificate:
    """Worst-case (guaranteed) deployment metrics over a decay/drift box."""

    model: str
    h_max: float
    delta_max: float
    transductive_auc: float
    transductive_f1: float
    guaranteed_auc: float             # worst-case inductive AUC over the box
    guaranteed_auprc: float
    guaranteed_f1: float              # worst-case TPC-achievable F1 over the box
    worst_case_separation: float
    worst_case_auc_drop: float        # transductive_auc - guaranteed_auc

    def guaranteed(self, metric: str) -> float:
        key = {"auc": "guaranteed_auc", "auprc": "guaranteed_auprc",
               "f1": "guaranteed_f1"}.get(metric)
        if key is None:
            raise ValueError(f"unknown metric {metric!r}")
        return float(getattr(self, key))


def protocol_robustness_certificate(
    model: ModelProfile,
    env: Environment,
    h_max: float,
    delta_max: float = 0.0,
    n_grid: int = 21,
) -> RobustnessCertificate:
    """Guaranteed worst-case deployment metrics for ``model`` over the box
    ``h in [0, h_max], |delta| <= delta_max``.

    AUC is prior-free and monotone-decreasing in decay, so its worst case is at
    ``h = h_max`` (prior-free). F1 (TPC-achievable) and AUPRC are prior-sensitive,
    so they are minimised over an ``(h, delta)`` grid. The F1 guarantee assumes
    the deployed system uses the fix (best-F1 at the deployed regime).
    """
    h_max = float(np.clip(h_max, 0.0, 1.0))
    mu_t = transductive_separation(model, env)
    mu_worst = inductive_separation(model, replace(env, homophily_decay=h_max))
    guaranteed_auc = auc_from_separation(mu_worst)

    hs = np.linspace(0.0, h_max, max(2, n_grid))
    ds = np.linspace(-delta_max, delta_max, max(1, n_grid)) if delta_max > 0 else np.array([0.0])
    worst_f1 = np.inf
    worst_auprc = np.inf
    for h in hs:
        mu_i = inductive_separation(model, replace(env, homophily_decay=float(h)))
        for d in ds:
            e = replace(env, homophily_decay=float(h), prior_drift=float(d))
            pi_test = e.test_prior
            worst_f1 = min(worst_f1, best_f1_threshold(mu_i, pi_test)[1])
            worst_auprc = min(worst_auprc, auprc_binormal(mu_i, pi_test))

    _, f1_trans = best_f1_threshold(mu_t, env.transductive_prior)
    auc_trans = auc_from_separation(mu_t)
    return RobustnessCertificate(
        model=model.name,
        h_max=h_max,
        delta_max=float(delta_max),
        transductive_auc=float(auc_trans),
        transductive_f1=float(f1_trans),
        guaranteed_auc=float(guaranteed_auc),
        guaranteed_auprc=float(worst_auprc),
        guaranteed_f1=float(worst_f1),
        worst_case_separation=float(mu_worst),
        worst_case_auc_drop=float(auc_trans - guaranteed_auc),
    )


@dataclass(frozen=True)
class CertifiedSelectionVerdict:
    """Naive (transductive) vs certified (worst-case maximin) model selection."""

    metric: str
    h_max: float
    delta_max: float
    transductive_pick: str            # naive: best transductive metric
    certified_pick: str               # maximin: best guaranteed deployment metric
    selection_reversed: bool
    naive_guaranteed_value: float     # the naive pick's worst-case guarantee
    certified_guaranteed_value: float # the robust pick's worst-case guarantee
    certified_regret: float           # certified - naive (>= 0): cost of naive choice
    reversal_risks: List[str]         # models whose guarantee is below the naive pick


def certified_protocol_robust_selection(
    profiles: List[ModelProfile],
    env: Environment,
    h_max: float,
    delta_max: float = 0.0,
    metric: str = "f1",
) -> CertifiedSelectionVerdict:
    """Maximin model selection with a deployment guarantee.

    Selects the model with the best *worst-case* deployment metric over the
    decay/drift box, contrasts it with the naive transductive-leaderboard pick,
    and reports the certified regret and the models whose guarantee falls below
    the naive pick's transductive standing (the certified reversal risks).
    """
    names = [p.name for p in profiles]
    certs = [protocol_robustness_certificate(p, env, h_max, delta_max) for p in profiles]
    t_vals = [evaluate_model(p, env).metric(metric, "transductive") for p in profiles]
    g_vals = [c.guaranteed(metric) for c in certs]

    t_pick = int(np.argmax(t_vals))
    c_pick = int(np.argmax(g_vals))
    naive_g = float(g_vals[t_pick])
    best_g = float(g_vals[c_pick])
    # reversal risks: models that look good transductively but whose guarantee is
    # worse than the naive pick's guarantee (they would disappoint at deployment).
    risks = sorted(
        (names[k] for k in range(len(profiles))
         if t_vals[k] >= t_vals[t_pick] - 1e-9 or g_vals[k] < naive_g - 1e-9),
        key=lambda nm: g_vals[names.index(nm)],
    )
    risks = [nm for nm in risks if nm != names[c_pick]]
    return CertifiedSelectionVerdict(
        metric=metric,
        h_max=float(np.clip(h_max, 0.0, 1.0)),
        delta_max=float(delta_max),
        transductive_pick=names[t_pick],
        certified_pick=names[c_pick],
        selection_reversed=(names[t_pick] != names[c_pick]),
        naive_guaranteed_value=naive_g,
        certified_guaranteed_value=best_g,
        certified_regret=max(best_g - naive_g, 0.0),
        reversal_risks=risks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric-fragility ordering  +  unified reversal taxonomy
#
# Pulls the whole framework together: as structure decays (h grows), *different
# metrics reverse at different decays*. The central theory result, stated as an
# ordering of critical decays:
#
#     h*_f1(stale)  <=  h*_f1(tpc)  <=  h*_auc  ==  h*_auprc
#
# i.e. a stale operating-point metric is the *most fragile* (reverses first); the
# fix (TPC) pushes its reversal later, up to but not past the representation
# (AUC/AUPRC) boundary; ranking metrics are the *least fragile*. This is exactly
# "TPC shrinks the reversal region but cannot cross the boundary", re-expressed as
# a one-dimensional fragility hierarchy that a single table can report.
# ─────────────────────────────────────────────────────────────────────────────


def metric_critical_decay(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    metric: str = "f1",
    protocol: str = "inductive_stale",
    n_scan: int = 401,
) -> Optional[float]:
    """First homophily decay ``h*`` at which the ``A`` vs ``B`` ranking flips.

    Generic numerical companion to the closed-form :func:`auc_reversal_decay`:
    scans ``h`` in ``[0, 1]`` for the first sign change of the inductive metric
    margin ``metric_A^I(h) - metric_B^I(h)`` (the transductive leader is decay-
    independent, so this is the reversal onset), and linearly interpolates the
    crossing. Works for any metric the accessor supports (``auc``, ``auprc``,
    ``f1``) under either inductive protocol. ``None`` if no crossing occurs.
    """
    hs = np.linspace(0.0, 1.0, int(max(3, n_scan)))
    margins = np.empty(hs.size)
    for k, h in enumerate(hs):
        e = replace(env, homophily_decay=float(h))
        margins[k] = (evaluate_model(model_a, e).metric(metric, protocol)
                      - evaluate_model(model_b, e).metric(metric, protocol))
    for k in range(1, margins.size):
        a0, a1 = margins[k - 1], margins[k]
        if a0 == 0.0:
            return float(hs[k - 1])
        if a0 * a1 < 0.0:
            frac = a0 / (a0 - a1)
            return float(hs[k - 1] + frac * (hs[k] - hs[k - 1]))
    return None


@dataclass(frozen=True)
class ReversalDecayOrdering:
    """Per-metric critical decays + the fragility-ordering check."""

    f1_stale: Optional[float]
    f1_tpc: Optional[float]
    auc: Optional[float]
    auprc: Optional[float]
    ordering_holds: bool          # f1_stale <= f1_tpc <= auc (ignoring None)
    most_fragile: Optional[str]   # metric/protocol that reverses earliest


def reversal_decay_ordering(
    model_a: ModelProfile, model_b: ModelProfile, env: Environment, tol: float = 1e-3
) -> ReversalDecayOrdering:
    """Critical decays for each metric and the fragility-ordering verdict.

    Verifies the theory's prediction that the stale-F1 reversal arrives no later
    than the TPC-F1 reversal, which in turn arrives no later than the AUC/AUPRC
    (representation) reversal — a decay-ordering restatement of the TPC bound.
    ``None`` entries (no reversal for that metric) are skipped in the check.
    """
    f1s = metric_critical_decay(model_a, model_b, env, "f1", "inductive_stale")
    f1t = metric_critical_decay(model_a, model_b, env, "f1", "inductive_tpc")
    auc = auc_reversal_decay(model_a, model_b, env)
    auprc = auprc_reversal_decay(model_a, model_b, env)

    seq = [("f1_stale", f1s), ("f1_tpc", f1t), ("auc", auc)]
    present = [(nm, v) for nm, v in seq if v is not None]
    holds = all(present[i][1] <= present[i + 1][1] + tol for i in range(len(present) - 1))
    labelled = [(nm, v) for nm, v in
                [("f1_stale", f1s), ("f1_tpc", f1t), ("auc", auc), ("auprc", auprc)]
                if v is not None]
    most_fragile = min(labelled, key=lambda t: t[1])[0] if labelled else None
    return ReversalDecayOrdering(
        f1_stale=f1s, f1_tpc=f1t, auc=auc, auprc=auprc,
        ordering_holds=bool(holds), most_fragile=most_fragile,
    )


@dataclass(frozen=True)
class ReversalTaxonomy:
    """One-call classification of how an A-vs-B pair reverses across metrics."""

    regime: str                   # named overall regime (see below)
    auc_reversed: bool
    auprc_reversed: bool
    f1_stale_reversed: bool
    f1_tpc_reversed: bool
    cost_reversed: bool
    ordering: ReversalDecayOrdering


def reversal_taxonomy(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
) -> ReversalTaxonomy:
    """Classify the pair's reversal behaviour at ``env`` into a named regime.

    Regimes (most to least severe):
      * ``"representation_reversal"`` — AUC/AUPRC reverse: a backbone-quality
        flip the fix cannot repair;
      * ``"calibration_reversal"`` — only operating-point metrics (stale F1 / cost)
        reverse; TPC repairs the F1 reversal (``f1_tpc`` not reversed);
      * ``"residual_reversal"`` — F1 reverses even after TPC but AUC does not
        (a thin band the fix narrows but cannot close);
      * ``"no_reversal"`` — the leaderboard order is protocol-stable here.
    """
    auc_r = detect_reversal(model_a, model_b, env, "auc", "inductive_stale").reversed
    auprc_r = detect_reversal(model_a, model_b, env, "auprc", "inductive_stale").reversed
    f1s_r = detect_reversal(model_a, model_b, env, "f1", "inductive_stale").reversed
    f1t_r = detect_reversal(model_a, model_b, env, "f1", "inductive_tpc").reversed
    cost_r = cost_reversal(model_a, model_b, env, c_fp, c_fn, "inductive_stale").reversed

    if auc_r or auprc_r:
        regime = "representation_reversal"
    elif f1t_r:
        regime = "residual_reversal"
    elif f1s_r or cost_r:
        regime = "calibration_reversal"
    else:
        regime = "no_reversal"
    return ReversalTaxonomy(
        regime=regime,
        auc_reversed=auc_r,
        auprc_reversed=auprc_r,
        f1_stale_reversed=f1s_r,
        f1_tpc_reversed=f1t_r,
        cost_reversed=cost_r,
        ordering=reversal_decay_ordering(model_a, model_b, env),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Robustness to non-Gaussianity  (distribution-free certification)
#
# The binormal model assumes Gaussian class-conditional scores. The obvious
# objection: real GNN scores are not Gaussian — how wrong can the closed forms
# be? This block answers it with *provable, distribution-free* bounds in terms of
# the Kolmogorov (sup-CDF) distance between the true score laws and the fitted
# binormal ones.
#
# AUC bound (tight). With AUC = P(S_1 > S_0) = ∫ F_0 dF_1 and binormal CDFs
# G_0, G_1, splitting the difference and integrating the second term by parts:
#
#     |AUC_true - AUC_binormal| <= sup|F_0 - G_0| + sup|F_1 - G_1| = eps_0 + eps_1.
#
# Reversal certification. If both protocols' AUC margins exceed 2(eps_0+eps_1),
# every per-model AUC can move by at most (eps_0+eps_1) and the *sign* of each
# margin — hence the reversal verdict — is preserved. So the reversal is
# certified robust to any non-Gaussianity with eps_0+eps_1 below a critical
# budget. This neutralises the "but scores aren't Gaussian" critique with a
# guarantee rather than an appeal to the CLT.
#
# F1 carries a Lipschitz (worst-case) bound that scales as 1/pi — i.e. operating-
# point metrics are *intrinsically less robust* to misspecification under heavy
# imbalance than the ranking AUC, an honest and useful caveat.
# ─────────────────────────────────────────────────────────────────────────────


def _empirical_cdf_sup_distance(samples: np.ndarray, mean: float, sd: float) -> float:
    """Kolmogorov distance between the empirical CDF of ``samples`` and N(mean, sd)."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    s = max(float(sd), 1e-12)
    G = _Phi((x - mean) / s)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(ecdf_hi - G)), np.max(np.abs(ecdf_lo - G))))


def binormal_kolmogorov_eps(
    pos_scores, neg_scores
) -> Dict[str, float]:
    """Fit a per-class Gaussian to real scores and return the binormal misfit.

    Returns ``{"eps_pos", "eps_neg", "mu", "mean_pos", "sd_pos", ...}`` where
    ``eps_*`` are the Kolmogorov distances between the empirical class score CDFs
    and the fitted Gaussians, and ``mu`` is the implied binormal separation
    ``(mean_pos - mean_neg)/pooled_sd``. These ``eps`` feed the robustness bounds
    so the certification can be applied to *measured* GNN scores.
    """
    p = np.asarray(pos_scores, dtype=float)
    q = np.asarray(neg_scores, dtype=float)
    mean_p, sd_p = float(p.mean()), float(p.std(ddof=1) if p.size > 1 else 1.0)
    mean_q, sd_q = float(q.mean()), float(q.std(ddof=1) if q.size > 1 else 1.0)
    pooled = float(np.sqrt(0.5 * (sd_p ** 2 + sd_q ** 2))) or 1e-9
    return {
        "eps_pos": _empirical_cdf_sup_distance(p, mean_p, sd_p),
        "eps_neg": _empirical_cdf_sup_distance(q, mean_q, sd_q),
        "mu": (mean_p - mean_q) / pooled,
        "mean_pos": mean_p, "sd_pos": sd_p,
        "mean_neg": mean_q, "sd_neg": sd_q,
    }


def auc_robustness_bound(eps_pos: float, eps_neg: Optional[float] = None) -> float:
    """Max AUC error under Kolmogorov-``eps`` non-Gaussianity: ``eps_pos + eps_neg``."""
    en = eps_pos if eps_neg is None else eps_neg
    return float(min(1.0, max(0.0, eps_pos) + max(0.0, en)))


def auc_interval(
    auc_model: float, eps_pos: float, eps_neg: Optional[float] = None
) -> Tuple[float, float]:
    """Guaranteed AUC interval ``[auc - b, auc + b]`` with ``b = eps_pos+eps_neg``."""
    b = auc_robustness_bound(eps_pos, eps_neg)
    return (float(max(0.0, auc_model - b)), float(min(1.0, auc_model + b)))


def f1_robustness_bound(
    pi: float, eps_pos: float, eps_neg: Optional[float] = None
) -> float:
    """Worst-case (Lipschitz) F1 error under Kolmogorov-``eps`` non-Gaussianity.

    ``|dF1| <= (2/pi)(eps_pos + eps_neg)`` (the binormal F1 is Lipschitz in TPR and
    FPR with constants bounded by ``2/pi``). The ``1/pi`` factor makes explicit
    that operating-point metrics are less robust to misspecification than AUC,
    especially under heavy imbalance. Capped at 1.
    """
    en = eps_pos if eps_neg is None else eps_neg
    pi = float(np.clip(pi, 1e-6, 1.0))
    return float(min(1.0, (2.0 / pi) * (max(0.0, eps_pos) + max(0.0, en))))


@dataclass(frozen=True)
class RobustReversalCertificate:
    """Is a reversal verdict robust to bounded non-Gaussianity of the scores?"""

    transductive_margin: float        # AUC margin (A - B) under transductive
    inductive_margin: float           # AUC margin (A - B) under inductive
    auc_error_budget: float           # b = eps_pos + eps_neg (per model)
    is_reversal: bool                 # do the model margins have opposite signs?
    margin_slack: float               # min(|m_T|,|m_B|) - 2b  (>0 => sign-stable)
    verdict_robust: bool              # both margin signs survive the perturbation
    reversal_certified: bool          # is_reversal AND verdict_robust
    critical_budget: float            # b* = min(|m_T|,|m_I|)/2 (max b preserving signs)
    critical_kolmogorov_eps: float    # b*/2 (symmetric per-class eps that still holds)


def certify_reversal_robustness(
    margin_trans: float,
    margin_ind: float,
    eps_pos: float = 0.0,
    eps_neg: Optional[float] = None,
) -> RobustReversalCertificate:
    """Certify a reversal verdict against Kolmogorov-``eps`` non-Gaussianity.

    ``margin_*`` are the binormal AUC margins (model A minus model B) under each
    protocol. Each per-model AUC can shift by at most ``b = eps_pos + eps_neg``,
    so each margin shifts by at most ``2b``; a margin's sign — and therefore the
    leaderboard verdict — is preserved when ``|margin| > 2b``. The reversal (both
    signs, opposite) is certified when the smaller margin clears that bar. Reports
    the critical budget ``b*`` (and the symmetric per-class ``eps* = b*/2``) up to
    which the conclusion provably holds.
    """
    b = auc_robustness_bound(eps_pos, eps_neg)
    smaller = min(abs(margin_trans), abs(margin_ind))
    is_rev = (margin_trans > 0) != (margin_ind > 0) and margin_trans != 0 and margin_ind != 0
    slack = smaller - 2.0 * b
    robust = slack > 0
    b_star = smaller / 2.0
    return RobustReversalCertificate(
        transductive_margin=float(margin_trans),
        inductive_margin=float(margin_ind),
        auc_error_budget=float(b),
        is_reversal=bool(is_rev),
        margin_slack=float(slack),
        verdict_robust=bool(robust),
        reversal_certified=bool(is_rev and robust),
        critical_budget=float(b_star),
        critical_kolmogorov_eps=float(b_star / 2.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-shift taxonomy  (places the protocol gap in the formal shift framework)
#
# The two environment knobs are exactly the two canonical dataset-shift types
# (Storkey 2009; Moreno-Torres et al. 2012):
#
#   * prior_drift (delta)     -> LABEL SHIFT      p(y) changes, p(x|y) fixed.
#     Leaves the score *ranking* (AUC/AUPRC) invariant; moves only the optimal
#     operating point. Recoverable by label-shift reweighting — exactly the
#     Saerens/BBSE correction inside TPC+TTA, which is consistent under this shift.
#   * homophily_decay (h)     -> REPRESENTATION / CONCEPT SHIFT  p(score|y) changes
#     (the class-conditional separation collapses as structure decays). Moves AUC
#     itself; NO label reweighting can recover a ranking the backbone no longer
#     produces.
#
# The headline theorem, made checkable: under PURE label shift (h = 0) the AUC is
# invariant across protocols, the irreducible representation gap is zero, and the
# whole F1 reversal is recoverable; under PURE representation shift the AUC moves
# and the gap is irreducible. The general protocol gap is their composition, and
# its recoverable fraction is the label-shift part.
# ─────────────────────────────────────────────────────────────────────────────


def protocol_shift_type(env: Environment, tol: float = 1e-9) -> str:
    """Classify the environment as ``label`` / ``representation`` / ``mixed`` / ``none``."""
    has_repr = env.homophily_decay > tol
    has_label = abs(env.prior_drift) > tol
    if has_repr and has_label:
        return "mixed"
    if has_repr:
        return "representation"
    if has_label:
        return "label"
    return "none"


@dataclass(frozen=True)
class ShiftComponents:
    """Decomposition of a model's protocol gap into the two shift types."""

    model: str
    shift_type: str                 # protocol_shift_type(env)
    auc_transductive: float
    auc_inductive: float
    auc_representation_drop: float  # auc_T - auc_I (>=0): prior-free representation loss
    auc_invariant: bool             # |auc_T - auc_I| < tol -> no representation shift
    representation_gap: float       # irreducible F1 loss at the deployment prior
    label_recoverable_gap: float    # F1 recoverable by re-thresholding (TPC)
    recoverable_fraction: float     # recoverable / (recoverable + irreducible)
    dominant: str                   # "representation" | "label" | "balanced" | "none"


def shift_decomposition(
    model: ModelProfile, env: Environment, auc_tol: float = 1e-6
) -> ShiftComponents:
    """Split ``model``'s protocol gap into label-shift vs representation-shift parts.

    Reuses the (provably non-negative) :func:`tpc_regret_bound` split — the
    irreducible ``representation_gap`` is the concept-shift loss the fix cannot
    recover, the ``calibration_regret`` is the re-thresholdable part that label-
    shift correction repairs — and adds the prior-free AUC drop and an
    ``auc_invariant`` flag that is the clean signature of pure label shift.
    """
    m = evaluate_model(model, env)
    rb = tpc_regret_bound(model, env)
    rep = max(rb.representation_gap, 0.0)
    lab = max(rb.calibration_regret, 0.0)
    denom = rep + lab
    frac = (lab / denom) if denom > 1e-12 else 0.0
    auc_drop = float(m.auc_transductive - m.auc_inductive)
    auc_inv = abs(auc_drop) < auc_tol
    if denom <= 1e-9:
        dominant = "none"
    elif rep > 1.5 * lab:
        dominant = "representation"
    elif lab > 1.5 * rep:
        dominant = "label"
    else:
        dominant = "balanced"
    return ShiftComponents(
        model=model.name,
        shift_type=protocol_shift_type(env),
        auc_transductive=float(m.auc_transductive),
        auc_inductive=float(m.auc_inductive),
        auc_representation_drop=auc_drop,
        auc_invariant=bool(auc_inv),
        representation_gap=float(rep),
        label_recoverable_gap=float(lab),
        recoverable_fraction=float(frac),
        dominant=dominant,
    )


# ─────────────────────────────────────────────────────────────────────────────
# (a) Multi-class / one-vs-rest extension
#
# Fraud is usually binary, but the framework generalises to K classes by treating
# each as a one-vs-rest binormal sub-problem with its own separation profile and
# base rate. Macro-averaged AUC / F1 then reverse exactly when the per-class
# binormal metrics, averaged, cross — so all the binary machinery is reused.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MulticlassModel:
    """A K-class model as K one-vs-rest binormal sub-models + their base rates."""

    name: str
    per_class: List[ModelProfile]      # one OvR ModelProfile per class
    class_priors: List[float]          # base rate of each class (need not sum to 1)

    def __post_init__(self) -> None:
        if len(self.per_class) != len(self.class_priors):
            raise ValueError("per_class and class_priors must have equal length")
        if len(self.per_class) < 2:
            raise ValueError("a multiclass model needs >= 2 classes")


def macro_metric(mc: MulticlassModel, env: Environment, metric: str, protocol: str) -> float:
    """Macro-averaged ``metric`` over the one-vs-rest sub-problems under ``protocol``.

    AUC/AUPRC are evaluated with each class's own base rate; F1 too. The shared
    ``env`` supplies the structural budget ``G`` and decay ``h`` (one regime); the
    per-class prior overrides ``env.prior`` for that sub-problem.
    """
    vals = []
    for prof, pi_c in zip(mc.per_class, mc.class_priors):
        e = replace(env, prior=float(np.clip(pi_c, 1e-4, 1.0 - 1e-4)))
        vals.append(evaluate_model(prof, e).metric(metric, protocol))
    return float(np.mean(vals))


def multiclass_reversal(
    mc_a: MulticlassModel,
    mc_b: MulticlassModel,
    env: Environment,
    metric: str = "auc",
    inductive_protocol: str = "inductive_stale",
) -> ReversalVerdict:
    """Does the macro-``metric`` leader flip between protocols for two K-class models?"""
    t_a = macro_metric(mc_a, env, metric, "transductive")
    t_b = macro_metric(mc_b, env, metric, "transductive")
    i_a = macro_metric(mc_a, env, metric, inductive_protocol)
    i_b = macro_metric(mc_b, env, metric, inductive_protocol)
    lead_t = _leader(t_a, t_b)
    lead_i = _leader(i_a, i_b)
    name = {1: mc_a.name, -1: mc_b.name, 0: "tie"}
    return ReversalVerdict(
        metric=f"macro_{metric}",
        transductive_leader=name[lead_t],
        inductive_leader=name[lead_i],
        reversed=(lead_t != 0 and lead_i != 0 and lead_t != lead_i),
        transductive_margin=float(t_a - t_b),
        inductive_margin=float(i_a - i_b),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (b) Analytic power curves  (seed count <-> detectable reversal magnitude)
# ─────────────────────────────────────────────────────────────────────────────


def detection_power(
    effect: float, sd_diff: float, n: int, alpha: float = 0.05, two_sided: bool = True
) -> float:
    """Probability of detecting a paired margin of size ``effect`` with ``n`` seeds.

    Standard normal power: with noncentrality ``lambda = effect/(sd/sqrt(n))`` and
    critical value ``z_a``, ``power = Phi(lambda - z_a) + Phi(-lambda - z_a)``
    (two-sided). Monotone increasing in ``n`` and ``|effect|``.
    """
    if sd_diff <= 0:
        return 1.0 if effect != 0 else float(alpha)
    if n < 1:
        return float(alpha)
    lam = effect / (sd_diff / np.sqrt(n))
    za = _norm_ppf(1 - alpha / 2) if two_sided else _norm_ppf(1 - alpha)
    if two_sided:
        return float(_Phi(lam - za) + _Phi(-lam - za))
    return float(_Phi(lam - za))


def power_curve(
    effect: float, sd_diff: float, n_values: Sequence[int],
    alpha: float = 0.05, two_sided: bool = True,
) -> List[Tuple[int, float]]:
    """Figure-ready ``[(n, power)]`` for a fixed effect across seed counts."""
    return [(int(n), detection_power(effect, sd_diff, int(n), alpha, two_sided))
            for n in n_values]


def mde_curve(
    sd_diff: float, n_values: Sequence[int],
    alpha: float = 0.05, power: float = 0.8,
) -> List[Tuple[int, float]]:
    """Figure-ready ``[(n, minimum_detectable_effect)]`` across seed counts."""
    return [(int(n), minimum_detectable_effect(sd_diff, int(n), alpha, power))
            for n in n_values]


# ─────────────────────────────────────────────────────────────────────────────
# (c) Information-theoretic structural budget  (what G means)
#
# The environment's ``graph_info`` G is the maximum extra binormal separation a
# perfect structural reader (rho = 1) can add. This block gives G an
# information-theoretic meaning: the equal-variance binormal separation mu carries
# symmetric KL ``mu^2`` between the class-conditional score laws, and a label<->
# score mutual information I(Y;S) that G upper-bounds. So G is, up to a square, the
# *information budget* latent in the graph.
# ─────────────────────────────────────────────────────────────────────────────


def binormal_symmetric_kl(mu: float) -> float:
    """Symmetric KL between the two class-conditional score laws at separation ``mu``.

    For ``N(+mu/2, 1)`` vs ``N(-mu/2, 1)``: ``KL = (mu)^2 / 2`` each way, so the
    symmetric divergence is ``mu^2``. Monotone in the separation; 0 at mu=0.
    """
    return float(mu * mu)


def binormal_mutual_information(
    mu: float, pi: float = 0.5, n: int = 4001, bits: bool = True
) -> float:
    """Mutual information ``I(Y; S)`` of the equal-variance binormal channel.

    ``I = H(Y) - H(Y|S)`` with ``H(Y|S) = ∫ m(s) h_b(P(Y=1|s)) ds`` over the score
    marginal ``m``. 0 at mu=0 (uninformative scores), increasing in mu, and bounded
    above by ``H(Y)``. Returned in bits by default (nats if ``bits=False``).
    """
    pi = float(np.clip(pi, 1e-9, 1 - 1e-9))
    grid = np.linspace(-8.0 - abs(mu), 8.0 + abs(mu), int(max(101, n)))
    f1 = np.exp(-0.5 * (grid - mu / 2.0) ** 2)
    f0 = np.exp(-0.5 * (grid + mu / 2.0) ** 2)
    f1 /= np.sqrt(2 * np.pi)
    f0 /= np.sqrt(2 * np.pi)
    m = pi * f1 + (1 - pi) * f0
    post = np.where(m > 0, pi * f1 / np.clip(m, 1e-300, None), 0.0)

    def _hb(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    h_y = -(pi * np.log(pi) + (1 - pi) * np.log(1 - pi))
    h_y_given_s = float(_trapz_compat(m * _hb(post), grid))
    info_nats = max(0.0, h_y - h_y_given_s)
    return float(info_nats / np.log(2.0)) if bits else float(info_nats)


def max_structural_separation_gain(env: Environment) -> float:
    """Max extra separation a perfect structural reader (rho=1) can add: ``G``."""
    return float(env.graph_info)


def structural_information_budget(env: Environment, feature_sep: float = 0.0) -> float:
    """Max extra symmetric-KL from structure: ``(a + G)^2 - a^2``.

    The information a feature-only model (rho=0, separation ``a``) is missing
    relative to a perfect structural reader (rho=1, separation ``a + G``).
    """
    a = float(max(feature_sep, 0.0))
    return float(binormal_symmetric_kl(a + env.graph_info) - binormal_symmetric_kl(a))


def max_structural_auc_gain(env: Environment, feature_sep: float = 0.0) -> float:
    """Max AUC a perfect structural reader gains over a feature-only model."""
    a = float(max(feature_sep, 0.0))
    return float(auc_from_separation(a + env.graph_info) - auc_from_separation(a))


# ─────────────────────────────────────────────────────────────────────────────
# (d) Finite-TEST-SET uncertainty  (orthogonal to the finite-seed power above)
#
# The seed-level power analysis asks whether a margin survives run-to-run noise.
# This asks the orthogonal question a reviewer also raises: is the AUC margin
# larger than the *test-set sampling* noise? The Hanley & McNeil (1982) closed
# form gives Var(AUC) from (AUC, n_pos, n_neg) with no resampling, so an AUC CI
# and a reversal significance test are available analytically.
# ─────────────────────────────────────────────────────────────────────────────


def auc_variance_hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil (1982) sampling variance of an empirical AUC.

    ``Var = [AUC(1-AUC) + (n_pos-1)(Q1 - AUC^2) + (n_neg-1)(Q2 - AUC^2)] /
    (n_pos n_neg)`` with ``Q1 = AUC/(2-AUC)``, ``Q2 = 2 AUC^2/(1+AUC)``.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    a = float(np.clip(auc, 1e-9, 1 - 1e-9))
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(max(var, 0.0))


def auc_confidence_interval(
    auc: float, n_pos: int, n_neg: int, conf: float = 0.95
) -> Tuple[float, float]:
    """Normal-approximation AUC confidence interval from the Hanley-McNeil variance."""
    se = np.sqrt(auc_variance_hanley_mcneil(auc, n_pos, n_neg))
    z = _norm_ppf(0.5 + conf / 2.0)
    return (float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0)))


@dataclass(frozen=True)
class TestSetReversalVerdict:
    """Is a protocol reversal beyond finite-test-set sampling noise?"""

    transductive_margin: float
    inductive_margin: float
    transductive_p: float
    inductive_p: float
    transductive_significant: bool
    inductive_significant: bool
    reversal_observed: bool
    reversal_significant: bool
    note: str


def reversal_significance_test_set(
    auc_trans_a: float, auc_trans_b: float,
    auc_ind_a: float, auc_ind_b: float,
    n_pos: int, n_neg: int,
    alpha: float = 0.05,
) -> TestSetReversalVerdict:
    """Test whether a transductive->inductive AUC reversal beats test-set noise.

    Each protocol's between-model AUC margin is z-tested using the Hanley-McNeil
    variances (treated as independent, which is *conservative* — positively
    correlated AUCs on a shared test set have smaller margin variance, so a
    significant verdict here is a lower bound on significance). A reversal is
    significant when both margins are individually significant and opposite-signed.
    """
    def _p(a1, a2):
        v = auc_variance_hanley_mcneil(a1, n_pos, n_neg) + auc_variance_hanley_mcneil(a2, n_pos, n_neg)
        if v <= 0:
            return 0.0 if a1 != a2 else 1.0
        z = abs(a1 - a2) / np.sqrt(v)
        return float(2.0 * _norm_sf(z))

    p_t = _p(auc_trans_a, auc_trans_b)
    p_i = _p(auc_ind_a, auc_ind_b)
    mt = auc_trans_a - auc_trans_b
    mi = auc_ind_a - auc_ind_b
    observed = (mt > 0) != (mi > 0) and mt != 0 and mi != 0
    sig = observed and p_t < alpha and p_i < alpha
    return TestSetReversalVerdict(
        transductive_margin=float(mt), inductive_margin=float(mi),
        transductive_p=p_t, inductive_p=p_i,
        transductive_significant=p_t < alpha, inductive_significant=p_i < alpha,
        reversal_observed=observed, reversal_significant=sig,
        note="Hanley-McNeil variances, independent-sample (conservative) margin test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# (e) Reversal-boundary confidence region  (parameter uncertainty -> h* CI)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReversalDecayUncertainty:
    """Distribution of the AUC critical decay h* under parameter uncertainty."""

    h_star_mean: Optional[float]
    h_star_ci_low: Optional[float]
    h_star_ci_high: Optional[float]
    p_reversal_reachable: float     # P(h* in [0,1]): the reversal is reachable
    n_samples: int


def reversal_decay_uncertainty(
    model_a: ModelProfile,
    model_b: ModelProfile,
    env: Environment,
    param_sd: float = 0.05,
    n_samples: int = 4000,
    conf: float = 0.95,
    seed: int = 20260621,
) -> ReversalDecayUncertainty:
    """Monte-Carlo CI for the AUC reversal decay ``h*`` under noisy (a, rho).

    Perturbs each model's ``feature_sep`` and ``structural_reliance`` by Gaussian
    noise of std ``param_sd`` (rho clipped to [0,1], a to >=0), recomputes the
    closed-form ``h* = 1 - (a_A - a_B)/((rho_B - rho_A) G)`` per draw, and reports
    the mean / percentile CI over draws where ``h*`` is reachable, plus the
    probability the reversal is reachable at all. Connects finite-seed parameter
    estimation to boundary uncertainty.
    """
    rng = np.random.default_rng(seed)
    G = env.graph_info
    aA = np.clip(model_a.feature_sep + rng.normal(0, param_sd, n_samples), 0.0, None)
    aB = np.clip(model_b.feature_sep + rng.normal(0, param_sd, n_samples), 0.0, None)
    rA = np.clip(model_a.structural_reliance + rng.normal(0, param_sd, n_samples), 0.0, 1.0)
    rB = np.clip(model_b.structural_reliance + rng.normal(0, param_sd, n_samples), 0.0, 1.0)
    drho = rB - rA
    valid = np.abs(drho) > 1e-9
    h = np.full(n_samples, np.nan)
    h[valid] = 1.0 - (aA[valid] - aB[valid]) / (drho[valid] * G)
    reachable = valid & (h >= 0.0) & (h <= 1.0)
    p_reach = float(reachable.mean())
    if reachable.sum() == 0:
        return ReversalDecayUncertainty(None, None, None, p_reach, int(n_samples))
    hr = h[reachable]
    lo = (100.0 - conf * 100.0) / 2.0
    return ReversalDecayUncertainty(
        h_star_mean=float(np.mean(hr)),
        h_star_ci_low=float(np.percentile(hr, lo)),
        h_star_ci_high=float(np.percentile(hr, 100.0 - lo)),
        p_reversal_reachable=p_reach,
        n_samples=int(n_samples),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (f) Optimal protocol design  (the constructive recommendation)
#
# The protocol's simulated structure decay h_eval is a *knob*. Transductive
# (h_eval=0) is the most optimistic estimate of deployment ranking quality;
# strict-inductive (h_eval=1) the most pessimistic. The deployment-faithful choice
# sets h_eval to the believed deployment decay; the evaluation bias is the AUC the
# protocol reports minus the AUC at the true deployment decay.
# ─────────────────────────────────────────────────────────────────────────────


def auc_at_decay(model: ModelProfile, env: Environment, h: float) -> float:
    """AUC the inductive protocol reports when structure is decayed by ``h``."""
    mu = inductive_separation(model, replace(env, homophily_decay=float(np.clip(h, 0.0, 1.0))))
    return auc_from_separation(mu)


def protocol_evaluation_bias(
    model: ModelProfile, env: Environment, h_eval: float, h_deploy: float
) -> float:
    """Signed AUC bias of evaluating at decay ``h_eval`` vs the deployment ``h_deploy``.

    Positive => the protocol *over-estimates* deployment ranking quality
    (transductive's failure mode); negative => it under-estimates (overly strict).
    Zero exactly when ``h_eval == h_deploy``.
    """
    return float(auc_at_decay(model, env, h_eval) - auc_at_decay(model, env, h_deploy))


def protocol_bias_curve(
    model: ModelProfile, env: Environment, h_deploy: float, h_evals: Sequence[float]
) -> List[Tuple[float, float]]:
    """Figure-ready ``[(h_eval, bias)]`` showing the protocol-choice bias curve."""
    return [(float(h), protocol_evaluation_bias(model, env, float(h), h_deploy)) for h in h_evals]


def recommend_evaluation_decay(
    h_deploy_low: float, h_deploy_high: Optional[float] = None
) -> float:
    """Deployment-faithful evaluation decay.

    For a point belief ``h_deploy_low`` about the deployment decay, the unbiased
    protocol uses that decay. For an interval ``[low, high]`` of plausible decays,
    the midpoint minimises the worst-case absolute AUC bias (the bias is monotone
    in ``h_eval``, so the minimax point is the interval centre).
    """
    if h_deploy_high is None:
        return float(np.clip(h_deploy_low, 0.0, 1.0))
    lo, hi = sorted((float(h_deploy_low), float(h_deploy_high)))
    return float(np.clip(0.5 * (lo + hi), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: canonical archetypes from the paper
# ─────────────────────────────────────────────────────────────────────────────


def canonical_panel(n: int = 6, graph_info: float = 1.2) -> List["ModelProfile"]:
    """An N-model panel spanning the structural-reliance spectrum.

    Reliance ``rho`` increases from feature-reliant to structure-reliant while
    intrinsic feature separation follows a **convex** ``a(rho) = 1.25 - 0.9 rho
    + 0.35 rho^2``.  The convexity is deliberate: with a *linear* ``a(rho)`` every
    model pair would cross at the same decay (a single simultaneous flip);
    curvature *staggers* the pairwise crossings so the leaderboard's Kendall tau
    declines **gradually** from ``+1`` (no decay, structure-reliant models lead)
    toward ``-1`` (full decay, feature-reliant models lead) -- the realistic,
    informative instability surface rather than a step function.
    """
    n = max(2, int(n))
    rhos = np.linspace(0.12, 0.95, n)
    feats = 1.25 - 0.9 * rhos + 0.35 * rhos ** 2
    return [
        ModelProfile(feature_sep=float(max(a, 0.0)), structural_reliance=float(r),
                     name=f"m{k + 1}_rho{r:.2f}")
        for k, (r, a) in enumerate(zip(rhos, feats))
    ]


@dataclass(frozen=True)
class GapDecomposition:
    """Representation-vs-calibration split of a measured protocol F1 gap.

    The split is **difference-based** so it is robust to the binormal model's
    (known) bias on real, non-Gaussian scores: the representation term is a
    *difference* of two binormal F1 values evaluated at the **same** prior, so a
    constant model-misfit cancels.  The remainder of the observed F1 gap is
    attributed to calibration (operating point / prior drift), which is exactly
    what a re-thresholding fix such as TPC+TTA can recover -- a gap that leaves
    AUROC (ranking quality) intact is recoverable by re-thresholding by
    construction.

    separation_optimistic / separation_strict : mu inferred from each AUROC.
    total_gap              : observed F1(optimistic) - observed F1(strict).
    representation_component: binormal best-F1 change attributable to the
        separation change, at the strict prior (TPC **cannot** recover this).
    calibration_component   : total_gap - representation_component (TPC can).
    calibration_share       : |calib| / (|calib| + |repr|).  Near 1 =>
        calibration-dominated (TPC-addressable); near 0 => representation-limited.
    """

    separation_optimistic: float
    separation_strict: float
    total_gap: float
    representation_component: float
    calibration_component: float
    calibration_share: float


def decompose_protocol_gap(
    auc_optimistic: float,
    auc_strict: float,
    prior_optimistic: float,
    prior_strict: float,
    observed_strict_f1: float,
    observed_optimistic_f1: float,
) -> GapDecomposition:
    """Split a real model's protocol F1 gap into representation vs calibration.

    Maps the two measured AUROCs onto binormal separations and attributes the
    observed F1 gap (optimistic - strict) to (a) a representation change -- the
    binormal best-F1 difference the separation change alone would produce at the
    deployment prior -- and (b) a calibration remainder a re-thresholding fix can
    recover.  Closed form; inputs are measured AUROCs, priors, and observed F1s.
    """
    mu_o = separation_from_auc(auc_optimistic)
    mu_s = separation_from_auc(auc_strict)
    # Representation component: hold the prior at the deployment (strict) value so
    # the binormal absolute bias cancels in the difference.
    repr_component = (best_f1_threshold(mu_o, prior_strict)[1]
                      - best_f1_threshold(mu_s, prior_strict)[1])
    total_gap = float(observed_optimistic_f1 - observed_strict_f1)
    calib_component = total_gap - repr_component
    denom = abs(calib_component) + abs(repr_component)
    share = float(abs(calib_component) / denom) if denom > 1e-12 else 0.0
    return GapDecomposition(
        separation_optimistic=mu_o,
        separation_strict=mu_s,
        total_gap=total_gap,
        representation_component=float(repr_component),
        calibration_component=float(calib_component),
        calibration_share=share,
    )


def canonical_pair() -> Tuple[ModelProfile, ModelProfile]:
    """The paper's archetype: a structure-reliant model vs a feature-reliant one.

    ``A`` ("structure_reliant", SAGE-like) wins transductively by leaning on the
    graph; ``B`` ("feature_reliant", MLP/tree-like) is weaker on the stable graph
    but more robust when structure decays.  These defaults put the AUC reversal
    near ``h* ~ 0.48`` with ``G = 1.2``.
    """
    a = ModelProfile(feature_sep=0.6, structural_reliance=0.9, name="structure_reliant")
    b = ModelProfile(feature_sep=1.1, structural_reliance=0.1, name="feature_reliant")
    return a, b


def summarize_pair(
    model_a: ModelProfile, model_b: ModelProfile, env: Environment
) -> Dict[str, object]:
    """One-call dump used by the experiment + tests: metrics, verdicts, h*."""
    ma = evaluate_model(model_a, env)
    mb = evaluate_model(model_b, env)
    return {
        "env": {
            "graph_info": env.graph_info,
            "homophily_decay": env.homophily_decay,
            "prior": env.prior,
            "prior_drift": env.prior_drift,
            "test_prior": env.test_prior,
        },
        "models": {model_a.name: ma.__dict__, model_b.name: mb.__dict__},
        "auc_reversal": detect_reversal(model_a, model_b, env, "auc", "inductive_stale").__dict__,
        "auprc_reversal": detect_reversal(model_a, model_b, env, "auprc", "inductive_stale").__dict__,
        "f1_reversal_stale": detect_reversal(
            model_a, model_b, env, "f1", "inductive_stale"
        ).__dict__,
        "f1_reversal_tpc": detect_reversal(
            model_a, model_b, env, "f1", "inductive_tpc"
        ).__dict__,
        "auc_critical_decay": auc_reversal_decay(model_a, model_b, env),
        "auprc_critical_decay": auprc_reversal_decay(model_a, model_b, env),
    }


__all__ = [
    "auc_from_separation",
    "separation_from_auc",
    "tpr_at",
    "fpr_at",
    "precision_at",
    "f1_at",
    "best_f1_threshold",
    "auc_from_separation_uv",
    "tpr_at_uv",
    "fpr_at_uv",
    "precision_at_uv",
    "f1_at_uv",
    "best_f1_threshold_uv",
    "auc_reversal_decay_uv",
    "auprc_baseline",
    "auprc_binormal",
    "auprc_reversal_decay",
    "GapDecomposition",
    "decompose_protocol_gap",
    "ModelProfile",
    "Environment",
    "transductive_separation",
    "inductive_separation",
    "ProtocolMetrics",
    "evaluate_model",
    "ReversalVerdict",
    "detect_reversal",
    "auc_reversal_decay",
    "reversal_phase_map",
    "leaderboard_values",
    "LeaderboardInstability",
    "ranking_instability",
    "leaderboard_stability_map",
    "SelectionVerdict",
    "protocol_robust_selection",
    "minimum_detectable_effect",
    "seeds_required",
    "ReversalPowerVerdict",
    "reversal_detectability",
    "reversal_detectability_from_samples",
    "TPCRegretBound",
    "tpc_regret_bound",
    "expected_cost_at",
    "best_cost_threshold",
    "CostMetrics",
    "evaluate_cost",
    "CostReversalVerdict",
    "cost_reversal",
    "RobustnessCertificate",
    "protocol_robustness_certificate",
    "CertifiedSelectionVerdict",
    "certified_protocol_robust_selection",
    "metric_critical_decay",
    "ReversalDecayOrdering",
    "reversal_decay_ordering",
    "ReversalTaxonomy",
    "reversal_taxonomy",
    "binormal_kolmogorov_eps",
    "auc_robustness_bound",
    "auc_interval",
    "f1_robustness_bound",
    "RobustReversalCertificate",
    "certify_reversal_robustness",
    "protocol_shift_type",
    "ShiftComponents",
    "shift_decomposition",
    "MulticlassModel",
    "macro_metric",
    "multiclass_reversal",
    "detection_power",
    "power_curve",
    "mde_curve",
    "binormal_symmetric_kl",
    "binormal_mutual_information",
    "max_structural_separation_gain",
    "structural_information_budget",
    "max_structural_auc_gain",
    "auc_variance_hanley_mcneil",
    "auc_confidence_interval",
    "TestSetReversalVerdict",
    "reversal_significance_test_set",
    "ReversalDecayUncertainty",
    "reversal_decay_uncertainty",
    "auc_at_decay",
    "protocol_evaluation_bias",
    "protocol_bias_curve",
    "recommend_evaluation_decay",
    "canonical_panel",
    "canonical_pair",
    "summarize_pair",
]
