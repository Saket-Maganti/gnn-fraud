"""
models/temporal_calibration.py

TPC+TTA — Temporal Prior Correction with Test-Time Adaptation.

This is the paper's **proposed solution** to the "evaluation protocols
reverse rankings under temporal shift" problem.  Three mechanisms are
bundled into one wrapper you apply on top of *any* classifier that can
produce class posteriors p(y|x):

  1. **Temperature scaling** on the validation window
     Standard Platt-style calibration so the post-softmax scores are
     proper probabilities rather than overconfident ones.  Fit once on val.

  2. **Temporal prior correction** (Saerens-Latinne-Decaestecker 2002)
     At inference we do *not* assume the class prior equals the training
     prior.  We estimate the target prior p_test(y) from the **inputs only**
     (classifier posteriors on a rolling window of recent nodes — never test
     labels) and rescale the posteriors by
          log p_test(y) - log p_train(y)
     which is the optimal Bayes-plug-in adjustment under label shift
     (a.k.a. prior-probability / target-shift).  The target prior can be
     estimated three ways, in increasing rigour:
       * ``argmax`` — empirical distribution of hard pseudo-labels (fast,
         but biased under strong class imbalance);
       * ``soft``   — normalised soft posterior mass;
       * ``em``     — the full Saerens EM iterated to convergence;
       * ``bbse``   — Black-Box Shift Estimation (Lipton et al. 2018) using
         the validation confusion matrix.  This is the estimator with the
         strongest guarantees and the one a reviewer will expect to see.

  3. **Adaptive decision threshold** learned on the held-out validation
     window with F1 as the objective.  Instead of the default 0.5 cut we
     pick τ* that maximises F1 on val.  By default the threshold is applied
     to the **prior-corrected** posteriors, so the prior shift already feeds
     through to the decision boundary.  ``predict`` additionally exposes a
     ``threshold_mode`` switch (``validation_selected`` [default/legacy],
     ``fixed``, ``bayes_odds``, ``prior_ratio``) for the threshold-shift
     ablations a reviewer may ask for — ``prior_ratio`` applies the
     Saerens-style prior-odds shift directly to the *uncorrected* posterior so
     the two correction routes (rescale-then-cut vs shift-the-cut) can be
     compared.  The default preserves the historical behaviour exactly.

Leakage-safe surface
--------------------
:class:`LeakageSafeTPC` wraps the calibrator in an auditable ``fit`` / ``adapt``
/ ``predict`` contract where ``adapt`` and ``predict`` accept **no** label
argument and raise :class:`LeakageError` if one is smuggled in under a common
alias.  Use it when you want the no-test-label guarantee enforced by the type
signature rather than by convention.

All three steps are closed-form or trivially optimised (binary line search /
fixed-point EM). There is no extra gradient step through the GNN, which means
this can be bolted onto *any* backbone — the contribution is a *protocol-level*
fix, not a new architecture.

Leakage guarantee
-----------------
Every target-prior estimator consumes **logits/posteriors only** and never the
test labels.  The temperature and threshold are fit exclusively on the
validation split.  ``apply_tpc_tta`` enforces this by construction: ``val_y``
feeds ``fit``; the test labels are used solely for the final metric computation
by the caller, never inside the calibrator.

Typical flow
------------
    logits_val  = model(full_graph).detach()
    tpc = TPCCalibrator.fit(logits_val[val], y_val, y_train=y_train)
    prior = tpc.rolling_prior(logits_test, time_step_test, method="bbse")
    preds = tpc.predict(logits_test, time_step_test, target_prior=prior)

The methods all consume **numpy** arrays for portability (so this runs against
XGBoost / sklearn baselines too), while ``fit_from_torch`` accepts torch
tensors and handles the device dance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Small numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    if tp == 0 and (fp + fn) == 0:
        return 0.0
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _fit_temperature(
    logits: np.ndarray,
    y:      np.ndarray,
    grid:   Optional[np.ndarray] = None,
) -> float:
    """Grid-search temperature that minimises NLL on held-out val.

    This is a closed-form-adjacent approximation (3 decades on a log-grid).
    We use NLL rather than raw accuracy because calibration is the point —
    the downstream threshold search handles the discriminative edge
    separately.
    """
    grid = grid if grid is not None else np.r_[
        np.linspace(0.3, 1.0, 8), np.linspace(1.0, 3.0, 11)[1:]
    ]
    best_t, best_nll = 1.0, np.inf
    N = logits.shape[0]
    ce_idx = np.arange(N)
    for t in grid:
        p = _softmax(logits, temperature=float(t))
        nll = -np.log(p[ce_idx, y] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def _fit_threshold(
    probs:    np.ndarray,
    y:        np.ndarray,
    pos_label: int = 1,
) -> float:
    """Pick threshold on the positive-class probability that maximises F1.

    ``probs`` is an [N, C] probability matrix. The final prediction is
    ``pos_label`` iff ``probs[:, pos_label] >= τ``. We grid-search τ on 200
    points between 0.02 and 0.98 — cheap and robust, since F1 is piecewise
    constant in τ with at most N breakpoints.
    """
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.02, 0.98, 200):
        y_hat = _threshold_predict(probs, float(tau), pos_label)
        f1 = _binary_f1(y, y_hat, pos_label=pos_label)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


def _threshold_predict(probs: np.ndarray, tau: float, pos_label: int) -> np.ndarray:
    """Hard labels: predict ``pos_label`` iff p_pos >= tau, else argmax of the rest.

    Single source of truth for the thresholding rule, shared by ``_fit_threshold``
    and ``TPCCalibrator.predict`` so fit and inference can never diverge.
    """
    p_pos = probs[:, pos_label]
    fallback = probs.copy()
    fallback[:, pos_label] = -np.inf
    rest_argmax = fallback.argmax(-1)
    return np.where(p_pos >= tau, pos_label, rest_argmax)


def _prior_from_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.zeros(n_classes)
    for c in range(n_classes):
        p[c] = max(float((y == c).sum()), 1.0)
    p = p / p.sum()
    return p


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Joint confusion ``C[i, j] = P(pred = i, true = j)`` (rows = predicted)."""
    C = np.zeros((n_classes, n_classes), dtype=float)
    for i, j in zip(y_pred.astype(int), y_true.astype(int)):
        if 0 <= i < n_classes and 0 <= j < n_classes:
            C[i, j] += 1.0
    total = C.sum()
    return C / total if total > 0 else C


def estimate_prior_em(
    probs:        np.ndarray,
    source_prior: np.ndarray,
    n_iters:      int = 100,
    tol:          float = 1e-6,
) -> np.ndarray:
    """Saerens-Latinne-Decaestecker (2002) EM estimate of the target prior.

    Given calibrated source posteriors ``probs`` (= p_s(y|x)) on the target
    inputs and the source prior ``source_prior`` (= p_s(y)), iterate the
    label-shift fixed point to convergence.  Uses **inputs only** — no labels.
    Returns the estimated target prior p_t(y), a length-C vector summing to 1.
    """
    eps = 1e-12
    src = np.asarray(source_prior, dtype=float) + eps
    src = src / src.sum()
    prior = src.copy()
    for _ in range(max(1, n_iters)):
        ratio = prior / src                          # [C]
        w = probs * ratio[None, :]                   # [N, C]
        w = w / np.clip(w.sum(axis=1, keepdims=True), eps, None)
        new_prior = w.mean(axis=0)
        new_prior = new_prior / new_prior.sum()
        if np.abs(new_prior - prior).max() < tol:
            prior = new_prior
            break
        prior = new_prior
    return prior


# ─────────────────────────────────────────────────────────────────────────────
# Main calibrator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TPCCalibrator:
    """Temperature + temporal-prior + adaptive-threshold wrapper.

    Fields populated by ``fit``:

    temperature     : float               softmax temperature T*
    train_prior     : np.ndarray [C]      source-domain class prior
    threshold       : float               τ* on the positive class (val F1)
    n_classes       : int                 number of output classes
    pos_label       : int                 index of the fraud class (default 1)
    val_confusion   : np.ndarray [C, C]   joint confusion on val (for BBSE)
    """

    temperature: float
    train_prior: np.ndarray
    threshold: float
    n_classes: int
    pos_label: int = 1
    val_confusion: Optional[np.ndarray] = field(default=None)

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        val_logits: np.ndarray,
        val_y:      np.ndarray,
        y_train:    Optional[np.ndarray] = None,
        pos_label:  int = 1,
    ) -> "TPCCalibrator":
        """Fit temperature, threshold and the val confusion matrix.

        ``y_train`` is the *source-domain* label array used to estimate
        ``p_train(y)``. If ``None``, we fall back to the empirical val
        prior (fine when val is drawn from the source regime).

        The validation confusion matrix is stored so BBSE prior estimation is
        available at inference without re-touching validation labels.
        """
        n_classes = int(val_logits.shape[1])
        T         = _fit_temperature(val_logits, val_y)
        probs     = _softmax(val_logits, temperature=T)
        threshold = _fit_threshold(probs, val_y, pos_label=pos_label)
        src_y     = y_train if y_train is not None else val_y
        train_prior = _prior_from_labels(src_y, n_classes)
        val_pred  = probs.argmax(-1)
        confusion = _confusion_matrix(val_y, val_pred, n_classes)
        return cls(
            temperature   = T,
            train_prior   = train_prior,
            threshold     = threshold,
            n_classes     = n_classes,
            pos_label     = pos_label,
            val_confusion = confusion,
        )

    # ── Target-prior estimators (inputs only — no test labels) ───────────────

    def estimate_prior_em(self, logits: np.ndarray, n_iters: int = 100) -> np.ndarray:
        """EM (Saerens 2002) target prior over the supplied inputs."""
        probs = _softmax(logits, temperature=self.temperature)
        return estimate_prior_em(probs, self.train_prior, n_iters=n_iters)

    def estimate_prior_bbse(self, logits: np.ndarray) -> np.ndarray:
        """Black-Box Shift Estimation (Lipton et al. 2018) target prior.

        Solves ``C w = mu_hat`` for the importance weights ``w[j] =
        p_t(j)/p_s(j)`` where ``C`` is the joint validation confusion matrix and
        ``mu_hat`` is the predicted-class distribution on the target inputs.
        The target prior is ``p_t(j) = w[j] * p_s(j)`` (clipped non-negative,
        renormalised).  Falls back to the soft estimate if ``C`` is singular.
        """
        probs = _softmax(logits, temperature=self.temperature)
        pred = probs.argmax(-1)
        mu_hat = _prior_from_labels(pred, self.n_classes)
        C = self.val_confusion
        if C is None:
            return mu_hat
        try:
            w = np.linalg.solve(C, mu_hat)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, mu_hat, rcond=None)[0]
        prior = np.clip(w * self.train_prior, 0.0, None)
        total = prior.sum()
        if not np.isfinite(total) or total <= 0:
            return mu_hat
        return prior / total

    # ── Prior estimation on a rolling test window ────────────────────────────

    def rolling_prior(
        self,
        logits:     np.ndarray,
        time_step:  np.ndarray,
        window:     int = 3,
        use_argmax: bool = True,
        method:     Optional[str] = None,
        n_em_iters: int = 100,
    ) -> np.ndarray:
        """Estimate p_test(y) per time bucket from a rolling window.

        For each bucket ``t`` we look at nodes in ``[t-window, t]`` and estimate
        a class prior from their classifier posteriors (the inputs) — **never
        from labels**, which is what makes this test-time adaptation rather than
        oracle correction.

        ``method`` selects the estimator (overrides ``use_argmax`` when given):
          * ``"argmax"`` — empirical distribution of hard pseudo-labels;
          * ``"soft"``   — normalised soft posterior mass;
          * ``"em"``     — Saerens EM iterated to convergence (recommended);
          * ``"bbse"``   — Black-Box Shift Estimation via the val confusion.

        ``use_argmax`` is kept for backward compatibility: when ``method`` is
        ``None`` it maps to ``"argmax"`` (True) or ``"soft"`` (False).
        Returns a ``[max_t + 1, C]`` array — one prior vector per bucket.
        """
        if method is None:
            method = "argmax" if use_argmax else "soft"
        probs = _softmax(logits, temperature=self.temperature)
        buckets = np.unique(time_step)
        out = np.zeros((int(buckets.max()) + 1, self.n_classes))
        for t in buckets:
            sel = (time_step >= t - window) & (time_step <= t)
            if not sel.any():
                out[t] = self.train_prior
                continue
            if method == "argmax":
                preds = probs[sel].argmax(-1)
                out[t] = _prior_from_labels(preds, self.n_classes)
            elif method == "soft":
                mass = probs[sel].sum(axis=0)
                out[t] = mass / max(mass.sum(), 1e-12)
            elif method == "em":
                out[t] = estimate_prior_em(probs[sel], self.train_prior, n_iters=n_em_iters)
            elif method == "bbse":
                out[t] = self.estimate_prior_bbse(logits[sel])
            else:
                raise ValueError(f"Unknown prior estimator method: {method!r}")
        return out

    # ── Prior-corrected prediction ──────────────────────────────────────────

    def predict(
        self,
        logits:       np.ndarray,
        time_step:    np.ndarray,
        target_prior: Optional[np.ndarray] = None,
        use_threshold: bool = True,
        threshold_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Return hard labels after temperature + prior + threshold.

        ``target_prior`` has shape ``[max_t + 1, C]`` — one prior vector per
        time bucket, as returned by ``rolling_prior``. If ``None`` we apply
        temperature (+ threshold) only (ablation: calibration alone).

        The prior correction is applied in log-space:
            log p'(y|x) ∝ log p(y|x) + log p_test(y) - log p_train(y)
        and the threshold is then applied to the positive-class probability via
        the shared ``_threshold_predict`` rule.

        ``threshold_mode`` selects the decision rule (``None`` preserves the
        historical behaviour — the val-F1-optimal τ* on the prior-corrected
        posterior, identical to ``"validation_selected"``):

          * ``"validation_selected"`` — τ* learned on val F1 (default/legacy);
          * ``"fixed"``               — fixed τ = 0.5 cut on the positive prob;
          * ``"bayes_odds"``          — Bayes rule on the *prior-corrected*
            posterior (predict positive iff corrected p_pos ≥ 0.5, i.e. odds≥1);
          * ``"prior_ratio"``         — Saerens-style threshold *shift* applied
            to the **uncorrected** posterior: the per-node τ is derived from the
            source/target prior odds, so the prior shift moves the boundary
            without re-normalising the posterior. Requires ``target_prior``.

        All modes consume scores only — no labels — and so are leakage-safe.
        """
        mode = threshold_mode or "validation_selected"
        probs_raw = _softmax(logits, temperature=self.temperature)

        probs = probs_raw
        if target_prior is not None:
            ratio = np.log(target_prior + 1e-12) - np.log(self.train_prior + 1e-12)
            per_node = ratio[time_step]              # bucket lookup
            logp = np.log(probs_raw + 1e-12) + per_node
            probs = np.exp(logp - logp.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)

        if not use_threshold:
            return probs.argmax(-1)

        if mode == "validation_selected":
            return _threshold_predict(probs, self.threshold, self.pos_label)
        if mode == "fixed":
            return _threshold_predict(probs, 0.5, self.pos_label)
        if mode == "bayes_odds":
            # Bayes-optimal 0/1 rule on the (prior-corrected) posterior.
            return _threshold_predict(probs, 0.5, self.pos_label)
        if mode == "prior_ratio":
            if target_prior is None:
                raise ValueError("threshold_mode='prior_ratio' requires target_prior")
            tau = self._prior_ratio_threshold(target_prior, time_step)
            # Applied to the *uncorrected* posterior: the prior shift lives in τ.
            return _threshold_predict(probs_raw, tau, self.pos_label)
        raise ValueError(f"Unknown threshold_mode: {mode!r}")

    def _prior_ratio_threshold(
        self, target_prior: np.ndarray, time_step: np.ndarray
    ) -> np.ndarray:
        """Per-node positive-class threshold from the source/target prior odds.

        Under label shift the Bayes rule decides positive iff the *source*
        posterior odds exceed ``R = (π_t[neg]·π_s[pos]) / (π_t[pos]·π_s[neg])``.
        For a binary positive-vs-rest decision this maps to a threshold
        ``τ = R / (1 + R)`` on the uncorrected positive probability. Returns a
        length-``N`` vector (one τ per node) so it can be passed straight to
        ``_threshold_predict``. Priors are clipped off the simplex boundary.
        """
        eps = 1e-6
        pos = self.pos_label
        src_pos = float(np.clip(self.train_prior[pos], eps, 1 - eps))
        src_neg = 1.0 - src_pos
        tgt = np.clip(target_prior[time_step], eps, 1 - eps)   # [N, C]
        tgt_pos = tgt[:, pos]
        tgt_neg = 1.0 - tgt_pos
        R = (tgt_neg * src_pos) / (tgt_pos * src_neg)
        return R / (1.0 + R)

    # ── Convenience: fit directly from torch tensors ────────────────────────

    @classmethod
    def fit_from_torch(
        cls,
        val_logits,
        val_y,
        y_train=None,
        pos_label: int = 1,
    ) -> "TPCCalibrator":
        import torch
        if isinstance(val_logits, torch.Tensor):
            val_logits = val_logits.detach().cpu().numpy()
        if isinstance(val_y, torch.Tensor):
            val_y = val_y.detach().cpu().numpy()
        if y_train is not None and isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        return cls.fit(val_logits, val_y, y_train=y_train, pos_label=pos_label)


# ─────────────────────────────────────────────────────────────────────────────
# Leakage-safe facade  (fit / adapt / predict)
# ─────────────────────────────────────────────────────────────────────────────

class LeakageError(ValueError):
    """Raised when an API that must not see test labels is handed labels."""


# Argument names that would smuggle ground-truth into a test-time call.
_FORBIDDEN_LABEL_KEYS = frozenset(
    {"y", "labels", "label", "test_labels", "test_y", "y_test", "targets", "target", "gold"}
)


def _reject_labels(kwargs: Mapping[str, object]) -> None:
    leaked = sorted(set(kwargs) & _FORBIDDEN_LABEL_KEYS)
    if leaked:
        raise LeakageError(
            "test-time adaptation/prediction must not receive labels; "
            f"refusing forbidden argument(s): {leaked}. The target prior is "
            "estimated from scores only (label shift / TTA contract)."
        )
    if kwargs:  # any other unexpected kwarg is also an error, fail loud
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")


@dataclass
class LeakageSafeTPC:
    """Test-time-adaptation facade that makes label leakage *structurally* hard.

    The three-call contract a reviewer can audit at a glance:

        cal = LeakageSafeTPC.fit(val_scores, val_labels, y_train=...)   # val only
        cal.adapt(test_scores, time_step=...)                          # scores only
        preds = cal.predict(test_scores, time_step=..., use_adapted_prior=True)

    ``adapt`` and ``predict`` accept **no** label argument. Passing one (under
    any common alias) raises :class:`LeakageError` rather than silently using it.
    The stored adapted prior is estimated from the supplied scores via BBSE/EM —
    never from labels — which is what makes this adaptation rather than oracle
    correction. Wraps :class:`TPCCalibrator`; semantics are identical, only the
    surface is locked down.
    """

    calibrator: "TPCCalibrator"
    adapted_prior: Optional[np.ndarray] = field(default=None)

    @classmethod
    def fit(
        cls,
        validation_scores: np.ndarray,
        validation_labels: np.ndarray,
        *,
        y_train: Optional[np.ndarray] = None,
        pos_label: int = 1,
    ) -> "LeakageSafeTPC":
        cal = TPCCalibrator.fit(
            np.asarray(validation_scores),
            np.asarray(validation_labels),
            y_train=None if y_train is None else np.asarray(y_train),
            pos_label=pos_label,
        )
        return cls(calibrator=cal)

    def adapt(
        self,
        test_scores: np.ndarray,
        *,
        time_step: Optional[np.ndarray] = None,
        test_metadata: Optional[Mapping[str, object]] = None,
        method: str = "bbse",
        window: int = 3,
        **forbidden: object,
    ) -> "LeakageSafeTPC":
        """Estimate the target prior from *scores only* and store it.

        ``time_step`` buckets the rolling-window estimate; when omitted all rows
        are treated as a single bucket. ``test_metadata`` is accepted and ignored
        (hook for future covariates) — it must never carry labels.
        """
        _reject_labels(forbidden)
        scores = np.asarray(test_scores)
        ts = np.zeros(scores.shape[0], dtype=int) if time_step is None else np.asarray(time_step)
        self.adapted_prior = self.calibrator.rolling_prior(
            scores, ts, window=window, method=method
        )
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        *,
        time_step: Optional[np.ndarray] = None,
        use_adapted_prior: bool = True,
        threshold_mode: Optional[str] = None,
        use_threshold: bool = True,
        **forbidden: object,
    ) -> np.ndarray:
        _reject_labels(forbidden)
        scores = np.asarray(test_scores)
        ts = np.zeros(scores.shape[0], dtype=int) if time_step is None else np.asarray(time_step)
        prior = self.adapted_prior if use_adapted_prior else None
        if use_adapted_prior and prior is None:
            raise RuntimeError("call adapt() before predict(use_adapted_prior=True)")
        return self.calibrator.predict(
            scores, ts, target_prior=prior, use_threshold=use_threshold,
            threshold_mode=threshold_mode,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Protocol-aware adapter facade  (the AAAI method-direction-1 surface)
# ─────────────────────────────────────────────────────────────────────────────

def _as_logits(scores: np.ndarray) -> np.ndarray:
    """Coerce a score array to an ``[N, C]`` logit matrix.

    Accepts (a) a 2-D logit/score matrix as-is, or (b) a 1-D vector of
    positive-class probabilities, which is lifted to a 2-column log-prob matrix
    ``[log(1-p), log(p)]`` so the binary fraud case ("just give me p_fraud")
    flows through the same Saerens/BBSE machinery as multi-class logits.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        p = np.clip(arr, 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(1 - p), np.log(p)])
    raise ValueError("scores must be 1-D (positive-class prob) or 2-D (logits)")


@dataclass
class ProtocolAwareTemporalAdapter:
    """Protocol-aware temporal calibration/adaptation under prior shift.

    Method-direction 1 from the professor's guidance, exposed under the exact
    ``fit`` / ``adapt`` / ``predict`` contract the AAAI method section describes.
    It is a thin, audited facade over :class:`LeakageSafeTPC` (temperature
    scaling + Saerens/BBSE/EM prior correction + validation-selected threshold);
    the heavy lifting and the leakage guards live there and are unit-tested.

    Contract::

        adp = ProtocolAwareTemporalAdapter(pos_label=1, prior_method="bbse")
        adp.fit(val_scores, val_labels, validation_time=val_t, y_train=train_y)
        adp.adapt(test_scores, test_time=test_t)          # scores only, no labels
        preds = adp.predict(test_scores, threshold_mode="prior_ratio")

    ``adapt`` and ``predict`` accept **no** test labels; passing one raises
    :class:`LeakageError`. The target prior is estimated from the test *scores*
    on a rolling window, optionally bucketed by ``test_time`` — never from labels.
    Accepts either 1-D positive-class probabilities or 2-D logits.
    """

    pos_label: int = 1
    prior_method: str = "bbse"
    window: int = 3
    _safe: Optional["LeakageSafeTPC"] = field(default=None, repr=False)
    _val_time: Optional[np.ndarray] = field(default=None, repr=False)
    _fitted: bool = False

    def fit(
        self,
        validation_scores: np.ndarray,
        validation_labels: np.ndarray,
        *,
        validation_time: Optional[np.ndarray] = None,
        validation_metadata: Optional[Mapping[str, object]] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> "ProtocolAwareTemporalAdapter":
        """Fit temperature + threshold on validation; store source prior.

        ``validation_time`` and ``validation_metadata`` are accepted for the
        deployment-style interface and retained for callers; the closed-form
        calibration itself uses the validation scores+labels. ``validation_metadata``
        must never carry labels (it is treated as opaque side info).
        """
        logits = _as_logits(validation_scores)
        self._safe = LeakageSafeTPC.fit(
            logits, np.asarray(validation_labels),
            y_train=None if y_train is None else np.asarray(y_train),
            pos_label=self.pos_label,
        )
        self._val_time = None if validation_time is None else np.asarray(validation_time)
        self._fitted = True
        return self

    def adapt(
        self,
        test_scores: np.ndarray,
        *,
        test_time: Optional[np.ndarray] = None,
        test_metadata: Optional[Mapping[str, object]] = None,
        **forbidden: object,
    ) -> "ProtocolAwareTemporalAdapter":
        """Estimate the target prior from test scores only (rolling window)."""
        if not self._fitted or self._safe is None:
            raise RuntimeError("call fit() before adapt()")
        self._safe.adapt(
            _as_logits(test_scores), time_step=None if test_time is None else np.asarray(test_time),
            test_metadata=test_metadata, method=self.prior_method, window=self.window,
            **forbidden,
        )
        return self

    def predict(
        self,
        test_scores: np.ndarray,
        *,
        test_time: Optional[np.ndarray] = None,
        threshold_mode: Optional[str] = None,
        use_threshold: bool = True,
        use_adapted_prior: bool = True,
        **forbidden: object,
    ) -> np.ndarray:
        """Hard labels after temperature + prior + threshold. Score-only."""
        if not self._fitted or self._safe is None:
            raise RuntimeError("call fit() before predict()")
        if use_adapted_prior and self._safe.adapted_prior is None:
            raise RuntimeError("call adapt() before predict(use_adapted_prior=True)")
        return self._safe.predict(
            _as_logits(test_scores), time_step=None if test_time is None else np.asarray(test_time),
            use_adapted_prior=use_adapted_prior, threshold_mode=threshold_mode,
            use_threshold=use_threshold, **forbidden,
        )

    @property
    def target_prior(self) -> Optional[np.ndarray]:
        return None if self._safe is None else self._safe.adapted_prior


# ─────────────────────────────────────────────────────────────────────────────
# One-shot helper for experiment scripts
# ─────────────────────────────────────────────────────────────────────────────

def apply_tpc_tta(
    logits_full,
    y_all,
    time_step,
    train_mask,
    val_mask,
    test_mask,
    pos_label: int = 1,
    window: int = 3,
    prior_method: str = "argmax",
) -> Dict[str, np.ndarray]:
    """Full TPC+TTA pipeline on a single forward pass.

    Returns the complete mechanism ablation so the GNN path matches the
    feature-only study (``scripts/run_tpc_tta_eval.py``):
        preds_baseline : argmax on raw logits          (raw   — nothing applied)
        preds_temp     : argmax after temperature       (temp  — calibration only)
        preds_prior    : temp + prior, no threshold     (prior — prior only)
        preds_thresh   : temp + threshold, no prior     (thresh — threshold only)
        preds_tpc_tta  : temp + prior + threshold       (full  — proposed method)
        calibrator     : fitted TPCCalibrator (for reporting / logging)

    ``prior_method`` chooses the target-prior estimator ("argmax", "soft",
    "em", "bbse").  All inputs are torch tensors or numpy arrays.

    Leakage note: ``val_y`` is used to fit T*/τ*; the target prior is estimated
    from logits only.  Test labels are never read here.
    """
    to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    logits = to_np(logits_full)
    y      = to_np(y_all)
    t      = to_np(time_step)
    vm     = to_np(val_mask).astype(bool)
    trm    = to_np(train_mask).astype(bool)

    cal = TPCCalibrator.fit(
        val_logits = logits[vm],
        val_y      = y[vm],
        y_train    = y[trm],
        pos_label  = pos_label,
    )
    prior = cal.rolling_prior(logits, t, window=window, method=prior_method)
    return {
        "preds_baseline": logits.argmax(-1),
        "preds_temp":     cal.predict(logits, t, target_prior=None,  use_threshold=False),
        "preds_prior":    cal.predict(logits, t, target_prior=prior, use_threshold=False),
        "preds_thresh":   cal.predict(logits, t, target_prior=None,  use_threshold=True),
        "preds_tpc_tta":  cal.predict(logits, t, target_prior=prior, use_threshold=True),
        "calibrator":     cal,
    }
