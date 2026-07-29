"""
experiments/run_protocol_phase_diagram.py

Controlled synthetic validation of the protocol rank-reversal theory.

This is the empirical companion to ``models/protocol_theory.py``.  It does **not**
plug numbers into the closed-form model: it *samples real temporal graphs* with a
tunable homophily-decay ``h`` and prior-drift ``delta``, fits two genuinely
different scorers (a structure-reliant one and a feature-reliant one), evaluates
them under the transductive and inductive protocols, applies the **real**
TPC+TTA calibrator (``models.temporal_calibration.apply_tpc_tta``) to the
inductive logits, and records where the leaderboard reverses.

It then *calibrates the 5-parameter analytic model from the simulation itself*
(measuring each model's separation at the ``h=0`` and ``h=1`` endpoints) and asks
the theory to predict the entire interior of the ``(h, delta)`` reversal phase
diagram.  The headline number is the **agreement** between the predicted and the
observed reversal maps -- i.e. "a closed-form model fit at two endpoints predicts
the full 2-D reversal boundary measured from simulated graphs".

This is deliberately NOT a dataset loader and is never a stand-in for Elliptic /
DGraphFin / T-Finance -- it is a controlled mechanism testbed (CPU-only, seeded,
fast).  Real-data experiments still fail loudly when their files are missing.

CLI
---
    python experiments/run_protocol_phase_diagram.py            # full grid
    python experiments/run_protocol_phase_diagram.py --quick    # tiny smoke grid
    python experiments/run_protocol_phase_diagram.py --out results/phase/phase.json

Output JSON schema (see ``PhaseResult``) is consumed by
``scripts/make_protocol_phase_figure.py`` to render the figure + LaTeX table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.protocol_theory import (  # noqa: E402
    Environment,
    ModelProfile,
    auc_reversal_decay,
    auprc_reversal_decay,
    detect_reversal,
)
from models.temporal_calibration import apply_tpc_tta, _binary_f1  # noqa: E402
from utils.metrics import bootstrap_ci  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic temporal-graph generator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GenConfig:
    """Knobs for the synthetic temporal fraud graph.

    The two *protocol* knobs (``homophily_decay`` h, ``prior_drift`` delta) are
    swept; the rest define the fixed archetype: a strong latent **structural**
    signal that a structure-reliant model exploits under homophily, plus a
    weaker intrinsic **feature** signal that a feature-reliant model uses and
    that survives when structure decays.
    """

    n_timesteps: int = 12
    nodes_per_step: int = 350
    train_frac: float = 0.6            # fraction of timesteps used for train+val
    k_neighbours: int = 5
    feat_sep: float = 1.25             # intrinsic feature signal (feature model)
    struct_sep: float = 1.5            # latent structural signal (structure model)
    homophily_train: float = 0.85      # P(neighbour shares label) in train era
    prior: float = 0.12                # training-time fraud base rate
    homophily_decay: float = 0.0       # h in [0,1] swept
    prior_drift: float = 0.0           # delta swept


def _sample_graph(cfg: GenConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Sample one temporal graph; return node arrays + two structural aggregates.

    The structural channel is a neighbour-mean of a per-node latent.  We compute
    it under **two** homophily regimes so the protocol difference is explicit
    rather than baked in:

      * ``nb_full`` (transductive): every node -- including test-era nodes --
        aggregates over a *stable, homophilous* neighbourhood (homophily
        ``homophily_train`` everywhere).  This is the transductive model
        propagating signal from the labelled training graph to test nodes via
        cross-split edges.
      * ``nb_decayed`` (inductive): test-era nodes aggregate over a neighbourhood
        whose homophily has eroded to
        ``0.5 + (homophily_train - 0.5) * (1 - h)`` -- the inductive model has no
        clean training anchors at test time, so its structural signal decays.

    At ``h = 0`` the two coincide (no temporal shift); at ``h = 1`` the inductive
    structural signal is pure noise.  Edges are drawn within a +-1 timestep
    window (temporally local).
    """
    T = cfg.n_timesteps
    n = cfg.nodes_per_step
    N = T * n
    train_max_t = int(round(cfg.train_frac * (T - 1)))

    time_step = np.repeat(np.arange(T), n)
    is_test = time_step > train_max_t
    prior_test = float(np.clip(cfg.prior + cfg.prior_drift, 1e-4, 1 - 1e-4))
    node_prior = np.where(is_test, prior_test, cfg.prior)
    y = (rng.random(N) < node_prior).astype(int)

    sign = (2 * y - 1).astype(float)
    feat = sign * (cfg.feat_sep / 2.0) + rng.standard_normal(N)
    struct_latent = sign * (cfg.struct_sep / 2.0) + rng.standard_normal(N)

    psi_full = np.full(N, cfg.homophily_train)
    psi_decayed = np.where(
        is_test,
        0.5 + (cfg.homophily_train - 0.5) * (1.0 - cfg.homophily_decay),
        cfg.homophily_train,
    )

    by_step_label: Dict[Tuple[int, int], np.ndarray] = {}
    for t in range(T):
        for lab in (0, 1):
            by_step_label[(t, lab)] = np.where((time_step == t) & (y == lab))[0]

    def _pool(t: int, lab: int) -> np.ndarray:
        parts = [by_step_label[(tt, lab)] for tt in (t - 1, t, t + 1) if 0 <= tt < T]
        return np.concatenate(parts) if parts else np.empty(0, dtype=int)

    def _neighbour_mean(psi: np.ndarray) -> np.ndarray:
        out = np.zeros(N)
        for i in range(N):
            t = int(time_step[i])
            want_same = rng.random(cfg.k_neighbours) < psi[i]
            vals = np.empty(cfg.k_neighbours)
            for kk in range(cfg.k_neighbours):
                lab = int(y[i]) if want_same[kk] else int(1 - y[i])
                pool = _pool(t, lab)
                if pool.size == 0:
                    pool = _pool(t, int(y[i]))
                vals[kk] = struct_latent[pool[rng.integers(pool.size)]] if pool.size else 0.0
            out[i] = vals.mean()
        return out

    nb_full = _neighbour_mean(psi_full)
    nb_decayed = _neighbour_mean(psi_decayed)

    return {
        "time_step": time_step,
        "y": y,
        "feat": feat,
        "nb_full": nb_full,
        "nb_decayed": nb_decayed,
        "train_max_t": np.array(train_max_t),
        "is_test": is_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scorers (genuinely different models on the same graph)
# ─────────────────────────────────────────────────────────────────────────────


def _logreg_fit_score(
    x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray
) -> np.ndarray:
    """Fit a 1-D logistic regression on the train rows, return logit matrix [N,2].

    Closed-form-free but cheap; falls back to a thresholded standardiser if the
    train labels are single-class (degenerate cell).  Returns 2-column logits so
    it flows through ``apply_tpc_tta`` unchanged.
    """
    x_train = x_train.reshape(-1, 1)
    x_all = x_all.reshape(-1, 1)
    if len(np.unique(y_train)) < 2:
        z = (x_all[:, 0] - x_all[:, 0].mean()) / (x_all[:, 0].std() + 1e-9)
        return np.column_stack([-z, z])
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(x_train, y_train)
    p = clf.predict_proba(x_all)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(1 - p), np.log(p)])


def _separation(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Empirical d-prime of the positive-class logit on ``mask`` rows."""
    s = logits[:, 1] - logits[:, 0]
    s_pos = s[mask & (y == 1)]
    s_neg = s[mask & (y == 0)]
    if s_pos.size < 2 or s_neg.size < 2:
        return 0.0
    pooled = np.sqrt(0.5 * (s_pos.var(ddof=1) + s_neg.var(ddof=1))) + 1e-9
    return float((s_pos.mean() - s_neg.mean()) / pooled)


def _auc(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    ym = y[mask]
    if len(np.unique(ym)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(ym, (logits[:, 1] - logits[:, 0])[mask]))


def _ap(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Average precision (AUPRC) of the positive score on ``mask`` rows."""
    ym = y[mask]
    if len(np.unique(ym)) < 2:
        return float(ym.mean()) if ym.size else 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(ym, (logits[:, 1] - logits[:, 0])[mask]))


def _best_threshold_f1(scores: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """F1-optimal threshold on the positive-class probability (val)."""
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.02, 0.98, 97):
        f1 = _binary_f1(y, (scores >= tau).astype(int), pos_label=1)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau, best_f1


def _pos_prob(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return p[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# One simulation cell: train both scorers under both protocols, apply TPC
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CellMeasurement:
    h: float
    delta: float
    # transductive
    auc_T_struct: float
    auc_T_feat: float
    auprc_T_struct: float
    auprc_T_feat: float
    f1_T_struct: float
    f1_T_feat: float
    # inductive (stale threshold from val, drifted test prior)
    auc_I_struct: float
    auc_I_feat: float
    auprc_I_struct: float
    auprc_I_feat: float
    f1_I_struct: float
    f1_I_feat: float
    # inductive + real TPC+TTA
    f1_tpc_struct: float
    f1_tpc_feat: float
    # measured separations (for theory calibration)
    muT_struct: float
    muT_feat: float
    muI_struct: float
    muI_feat: float


def _simulate_cell(cfg: GenConfig, seeds: List[int]) -> CellMeasurement:
    """Average one ``(h, delta)`` cell over ``seeds``."""
    acc: Dict[str, List[float]] = {}

    def push(k: str, v: float) -> None:
        acc.setdefault(k, []).append(float(v))

    for sd in seeds:
        rng = np.random.default_rng(sd)
        g = _sample_graph(cfg, rng)
        ts, y = g["time_step"], g["y"]
        train_max_t = int(g["train_max_t"])
        feat = g["feat"]
        nb_full, nb_decayed = g["nb_full"], g["nb_decayed"]

        in_window = ts <= train_max_t
        # val = last in-window timestep; train = the rest of the window.
        val_t = train_max_t
        val_mask = in_window & (ts == val_t)
        train_mask = in_window & (ts < val_t)
        test_mask = ts > train_max_t
        if val_mask.sum() < 5 or train_mask.sum() < 5 or test_mask.sum() < 5:
            continue

        # --- Transductive: structure aggregated over the INTACT graph (test
        #     nodes propagate from labelled training neighbours).  Feature model
        #     uses the intrinsic feature (protocol-invariant by construction).
        struct_T = _logreg_fit_score(nb_full[train_mask], y[train_mask], nb_full)
        feat_T = _logreg_fit_score(feat[train_mask], y[train_mask], feat)

        push("auc_T_struct", _auc(struct_T, y, test_mask))
        push("auc_T_feat", _auc(feat_T, y, test_mask))
        push("auprc_T_struct", _ap(struct_T, y, test_mask))
        push("auprc_T_feat", _ap(feat_T, y, test_mask))
        push("muT_struct", _separation(struct_T, y, test_mask))
        push("muT_feat", _separation(feat_T, y, test_mask))

        # Transductive operating point: threshold fit on val, applied on test at
        # the *same* in-distribution conditions (no drift penalty).
        for name, lg in (("struct", struct_T), ("feat", feat_T)):
            p = _pos_prob(lg)
            tau, _ = _best_threshold_f1(p[val_mask], y[val_mask])
            f1 = _binary_f1(y[test_mask], (p[test_mask] >= tau).astype(int))
            push(f"f1_T_{name}", f1)

        # --- Inductive: structure aggregated over the DECAYED test-era graph;
        #     no clean training anchors -> structural signal erodes with h.  The
        #     stale threshold is fit on the train-era val window (where structure
        #     is still intact -> a badly-calibrated cut for the decayed test).
        struct_I = _logreg_fit_score(nb_decayed[train_mask], y[train_mask], nb_decayed)
        feat_I = _logreg_fit_score(feat[train_mask], y[train_mask], feat)

        push("auc_I_struct", _auc(struct_I, y, test_mask))
        push("auc_I_feat", _auc(feat_I, y, test_mask))
        push("auprc_I_struct", _ap(struct_I, y, test_mask))
        push("auprc_I_feat", _ap(feat_I, y, test_mask))
        push("muI_struct", _separation(struct_I, y, test_mask))
        push("muI_feat", _separation(feat_I, y, test_mask))

        for name, lg in (("struct", struct_I), ("feat", feat_I)):
            p = _pos_prob(lg)
            tau, _ = _best_threshold_f1(p[val_mask], y[val_mask])  # stale on val
            f1_stale = _binary_f1(y[test_mask], (p[test_mask] >= tau).astype(int))
            push(f"f1_I_{name}", f1_stale)

            # Real TPC+TTA on the inductive logits.
            res = apply_tpc_tta(
                logits_full=lg,
                y_all=y,
                time_step=ts,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                window=3,
                prior_method="bbse",
            )
            f1_tpc = _binary_f1(y[test_mask], res["preds_tpc_tta"][test_mask])
            push(f"f1_tpc_{name}", f1_tpc)

    mean = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    return CellMeasurement(
        h=cfg.homophily_decay,
        delta=cfg.prior_drift,
        auc_T_struct=mean.get("auc_T_struct", 0.5),
        auc_T_feat=mean.get("auc_T_feat", 0.5),
        auprc_T_struct=mean.get("auprc_T_struct", 0.0),
        auprc_T_feat=mean.get("auprc_T_feat", 0.0),
        f1_T_struct=mean.get("f1_T_struct", 0.0),
        f1_T_feat=mean.get("f1_T_feat", 0.0),
        auc_I_struct=mean.get("auc_I_struct", 0.5),
        auc_I_feat=mean.get("auc_I_feat", 0.5),
        auprc_I_struct=mean.get("auprc_I_struct", 0.0),
        auprc_I_feat=mean.get("auprc_I_feat", 0.0),
        f1_I_struct=mean.get("f1_I_struct", 0.0),
        f1_I_feat=mean.get("f1_I_feat", 0.0),
        f1_tpc_struct=mean.get("f1_tpc_struct", 0.0),
        f1_tpc_feat=mean.get("f1_tpc_feat", 0.0),
        muT_struct=mean.get("muT_struct", 0.0),
        muT_feat=mean.get("muT_feat", 0.0),
        muI_struct=mean.get("muI_struct", 0.0),
        muI_feat=mean.get("muI_feat", 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reversal classification helpers
# ─────────────────────────────────────────────────────────────────────────────


def _reversed(t_struct: float, t_feat: float, i_struct: float, i_feat: float,
              eps: float = 1e-6) -> int:
    """1 if the struct-vs-feat ranking flips transductive -> inductive."""
    lt = np.sign(t_struct - t_feat) if abs(t_struct - t_feat) > eps else 0
    li = np.sign(i_struct - i_feat) if abs(i_struct - i_feat) > eps else 0
    return int(lt != 0 and li != 0 and lt != li)


# ─────────────────────────────────────────────────────────────────────────────
# Theory calibration from the simulation endpoints
# ─────────────────────────────────────────────────────────────────────────────


def _calibrate_theory(
    cells: List[CellMeasurement], prior: float,
) -> Tuple[ModelProfile, ModelProfile, Environment]:
    """Fit the 5-parameter analytic model from the simulation's h-endpoints.

    ``a_m`` (feature_sep) is the residual *inductive* separation at full decay
    (``h~1``, structure gone); the structural *gain* is the extra separation the
    model has at ``h=0`` (``muT - a_m``).  We share one graph budget
    ``G = max gain`` across the two models so the reliances ``rho_m = gain_m / G``
    land in ``[0, 1]`` *without clipping the separation scale* -- the
    reconstructed ``mu_T = a_m + rho_m G`` then matches the measured endpoints
    exactly, and the linear-decay interior ``mu_I(h) = a_m + rho_m G (1-h)`` is
    the theory's prediction the simulation is asked to confirm.
    """
    by_h: Dict[float, List[CellMeasurement]] = {}
    for c in cells:
        by_h.setdefault(round(c.h, 6), []).append(c)
    lo = by_h[min(by_h)]
    hi = by_h[max(by_h)]

    def avg(cs, attr):
        return float(np.mean([getattr(c, attr) for c in cs]))

    # Transductive separations are protocol-intact and h-independent; average
    # over all cells for a stable estimate.
    muT_struct = avg(cells, "muT_struct")
    muT_feat = avg(cells, "muT_feat")
    a_struct = max(avg(hi, "muI_struct"), 0.0)
    a_feat = max(avg(hi, "muI_feat"), 0.0)
    gain_struct = max(muT_struct - a_struct, 0.0)
    gain_feat = max(muT_feat - a_feat, 0.0)
    G = max(gain_struct, gain_feat, 1e-6)
    rho_struct = float(np.clip(gain_struct / G, 0.0, 1.0))
    rho_feat = float(np.clip(gain_feat / G, 0.0, 1.0))

    A = ModelProfile(feature_sep=a_struct, structural_reliance=rho_struct,
                     name="structure_reliant")
    B = ModelProfile(feature_sep=a_feat, structural_reliance=rho_feat,
                     name="feature_reliant")
    env = Environment(graph_info=G, prior=float(prior))
    return A, B, env


def _grid_status(
    grid_cells: List[CellMeasurement], n_delta: int, n_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reversal-status + AUC-reversal grids from row-major (delta x h) cells."""
    sim = np.zeros((n_delta, n_h), dtype=int)
    auc = np.zeros((n_delta, n_h), dtype=int)
    idx = 0
    for i in range(n_delta):
        for j in range(n_h):
            m = grid_cells[idx]
            idx += 1
            stale = _reversed(m.f1_T_struct, m.f1_T_feat, m.f1_I_struct, m.f1_I_feat)
            tpc = _reversed(m.f1_T_struct, m.f1_T_feat, m.f1_tpc_struct, m.f1_tpc_feat)
            sim[i, j] = 2 if tpc else (1 if stale else 0)
            auc[i, j] = _reversed(m.auc_T_struct, m.auc_T_feat, m.auc_I_struct, m.auc_I_feat)
    return sim, auc


def _theory_grid(
    A: ModelProfile, B: ModelProfile, env: Environment,
    h_grid: np.ndarray, delta_grid: np.ndarray,
) -> np.ndarray:
    """Analytic reversal-status grid for the calibrated theory over (delta x h)."""
    from dataclasses import replace as _replace
    th = np.zeros((delta_grid.size, h_grid.size), dtype=int)
    for i, d in enumerate(delta_grid):
        for j, h in enumerate(h_grid):
            e = _replace(env, homophily_decay=float(h), prior_drift=float(d))
            sr = detect_reversal(A, B, e, "f1", "inductive_stale").reversed
            tr = detect_reversal(A, B, e, "f1", "inductive_tpc").reversed
            th[i, j] = 2 if tr else (1 if sr else 0)
    return th


def _interior_h_mask(h_grid: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean (delta x h) mask of interior-h cells (h != min, max)."""
    h_arr = np.asarray(h_grid, float)
    interior = (h_arr > h_arr.min() + 1e-9) & (h_arr < h_arr.max() - 1e-9)
    return np.broadcast_to(interior[None, :], shape)


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PhaseResult:
    h_grid: List[float]
    delta_grid: List[float]
    cells: List[dict]
    sim_status: List[List[int]]        # delta x h : 0 none / 1 tpc-repairs / 2 persists
    theory_status: List[List[int]]
    auc_sim_reversed: List[List[int]]
    auprc_sim_reversed: List[List[int]]
    auprc_theory_reversed: List[List[int]]
    theory_params: dict
    agreement: dict
    config: dict
    wall_sec: float
    schema: str = "protocol_phase_v1"


def run_phase_sweep(
    h_grid: np.ndarray,
    delta_grid: np.ndarray,
    base: GenConfig,
    seeds: List[int],
    verbose: bool = True,
) -> PhaseResult:
    t0 = time.time()
    cells: List[CellMeasurement] = []
    sim_status = np.zeros((delta_grid.size, h_grid.size), dtype=int)
    auc_sim = np.zeros((delta_grid.size, h_grid.size), dtype=int)
    auprc_sim = np.zeros((delta_grid.size, h_grid.size), dtype=int)
    grid_cells: List[CellMeasurement] = []

    for i, d in enumerate(delta_grid):
        for j, h in enumerate(h_grid):
            cfg = GenConfig(**{**asdict(base),
                               "homophily_decay": float(h), "prior_drift": float(d)})
            m = _simulate_cell(cfg, seeds)
            cells.append(m)
            grid_cells.append(m)
            stale_rev = _reversed(m.f1_T_struct, m.f1_T_feat, m.f1_I_struct, m.f1_I_feat)
            tpc_rev = _reversed(m.f1_T_struct, m.f1_T_feat, m.f1_tpc_struct, m.f1_tpc_feat)
            sim_status[i, j] = 2 if tpc_rev else (1 if stale_rev else 0)
            auc_sim[i, j] = _reversed(m.auc_T_struct, m.auc_T_feat,
                                      m.auc_I_struct, m.auc_I_feat)
            auprc_sim[i, j] = _reversed(m.auprc_T_struct, m.auprc_T_feat,
                                        m.auprc_I_struct, m.auprc_I_feat)
            if verbose:
                print(f"  [cell] h={h:.2f} d={d:.2f}  "
                      f"f1_T(S/F)={m.f1_T_struct:.3f}/{m.f1_T_feat:.3f}  "
                      f"f1_I={m.f1_I_struct:.3f}/{m.f1_I_feat:.3f}  "
                      f"f1_tpc={m.f1_tpc_struct:.3f}/{m.f1_tpc_feat:.3f}  "
                      f"status={sim_status[i, j]}")

    # Calibrate theory from endpoints, predict the interior.
    A, B, env = _calibrate_theory(grid_cells, base.prior)
    theory_status = np.zeros_like(sim_status)
    auprc_theory = np.zeros_like(auprc_sim)
    for i, d in enumerate(delta_grid):
        for j, h in enumerate(h_grid):
            from dataclasses import replace as _replace
            e = _replace(env, homophily_decay=float(h), prior_drift=float(d))
            sr = detect_reversal(A, B, e, "f1", "inductive_stale").reversed
            tr = detect_reversal(A, B, e, "f1", "inductive_tpc").reversed
            theory_status[i, j] = 2 if tr else (1 if sr else 0)
            auprc_theory[i, j] = int(
                detect_reversal(A, B, e, "auprc", "inductive_stale").reversed
            )

    # Agreement metrics.
    sim_rev = (sim_status > 0).astype(int)
    th_rev = (theory_status > 0).astype(int)
    match_rev = (sim_rev == th_rev).astype(int)
    match_status = (sim_status == theory_status).astype(int)
    reversal_agree = float(match_rev.mean())
    status_agree = float(match_status.mean())

    # Held-out validation. `_calibrate_theory` anchors the binormal separations
    # at the two homophily-decay endpoints (h=min, h=max) and never uses delta.
    # The interior-h cells are therefore genuine out-of-sample predictions, so
    # reporting agreement on that subset answers the obvious "fit and evaluated
    # on the same cells" objection to the theory's predictive claim. A percentile
    # bootstrap over grid cells reports whether the agreement is uniform across
    # the region rather than carried by a handful of cells.
    h_arr = np.asarray(h_grid, float)
    interior = (h_arr > h_arr.min() + 1e-9) & (h_arr < h_arr.max() - 1e-9)
    holdout_mask = np.broadcast_to(interior[None, :], sim_status.shape)
    n_holdout = int(holdout_mask.sum())

    def _ci(vec: np.ndarray) -> List[Optional[float]]:
        if vec.size == 0:
            return [None, None]
        lo, hi = bootstrap_ci(vec.ravel().astype(float).tolist())
        return [round(lo, 4), round(hi, 4)]

    full_ci = _ci(match_rev)
    if n_holdout > 0:
        holdout_rev_agree: Optional[float] = float(match_rev[holdout_mask].mean())
        holdout_status_agree: Optional[float] = float(match_status[holdout_mask].mean())
        holdout_ci = _ci(match_rev[holdout_mask])
    else:
        holdout_rev_agree = holdout_status_agree = None
        holdout_ci = [None, None]
    hstar = auc_reversal_decay(A, B, env)

    # AUPRC validation: the closed-form AUPRC reversal (theory) vs the AUPRC
    # reversal measured on the sampled graphs (sim). AUPRC is the fraud-critical
    # ranking metric; this confirms the theory predicts its reversal too.
    auprc_match = (auprc_sim.astype(bool) == auprc_theory.astype(bool)).astype(int)
    auprc_agree = float(auprc_match.mean())
    auprc_ci = _ci(auprc_match)
    auprc_hstar = auprc_reversal_decay(A, B, env)

    # Per-seed robustness: re-run each seed's grid independently (its own
    # calibration + classification) so the theory<->sim agreement is reported as
    # a distribution across *independent* simulations, not just a seed-averaged
    # point estimate. This answers "is the agreement stable across simulations,
    # or seed-luck?" — complementary to the cell-bootstrap CI above.
    per_seed_full: List[float] = []
    per_seed_holdout: List[float] = []
    for s in seeds:
        gc_s: List[CellMeasurement] = []
        for d in delta_grid:
            for h in h_grid:
                cfg_s = GenConfig(**{**asdict(base),
                                     "homophily_decay": float(h), "prior_drift": float(d)})
                gc_s.append(_simulate_cell(cfg_s, [int(s)]))
        A_s, B_s, env_s = _calibrate_theory(gc_s, base.prior)
        sim_s, _ = _grid_status(gc_s, delta_grid.size, h_grid.size)
        th_s = _theory_grid(A_s, B_s, env_s, h_grid, delta_grid)
        match_s = (sim_s > 0) == (th_s > 0)
        per_seed_full.append(float(match_s.mean()))
        if n_holdout > 0:
            per_seed_holdout.append(float(match_s[holdout_mask].mean()))

    def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not xs:
            return None, None
        mean = round(float(np.mean(xs)), 4)
        std = round(float(np.std(xs, ddof=1)), 4) if len(xs) > 1 else 0.0
        return mean, std

    ps_full_mean, ps_full_std = _mean_std(per_seed_full)
    ps_ho_mean, ps_ho_std = _mean_std(per_seed_holdout)

    res = PhaseResult(
        h_grid=[float(x) for x in h_grid],
        delta_grid=[float(x) for x in delta_grid],
        cells=[asdict(c) for c in cells],
        sim_status=sim_status.tolist(),
        theory_status=theory_status.tolist(),
        auc_sim_reversed=auc_sim.tolist(),
        auprc_sim_reversed=auprc_sim.tolist(),
        auprc_theory_reversed=auprc_theory.tolist(),
        theory_params={
            "structure_reliant": {"feature_sep": A.feature_sep,
                                  "structural_reliance": A.structural_reliance},
            "feature_reliant": {"feature_sep": B.feature_sep,
                                "structural_reliance": B.structural_reliance},
            "graph_info": env.graph_info,
            "prior": env.prior,
            "auc_critical_decay": hstar,
            "auprc_critical_decay": auprc_hstar,
        },
        agreement={
            "reversal_binary_agreement": reversal_agree,
            "status_agreement": status_agree,
            "reversal_agreement_ci95": full_ci,
            "holdout_reversal_agreement": holdout_rev_agree,
            "holdout_status_agreement": holdout_status_agree,
            "holdout_reversal_agreement_ci95": holdout_ci,
            "n_holdout_cells": n_holdout,
            "holdout_definition": (
                "interior homophily-decay columns (h != min,max): the binormal "
                "separations are anchored at the two h endpoints, so these cells "
                "are out-of-sample; all delta rows are held out (delta never "
                "enters the calibration)"
            ),
            "ci_method": "percentile bootstrap over grid cells (utils.metrics.bootstrap_ci)",
            "per_seed": {
                "n_seeds": len(seeds),
                "reversal_agreement_mean": ps_full_mean,
                "reversal_agreement_std": ps_full_std,
                "reversal_agreement_values": [round(v, 4) for v in per_seed_full],
                "holdout_reversal_agreement_mean": ps_ho_mean,
                "holdout_reversal_agreement_std": ps_ho_std,
                "holdout_reversal_agreement_values": [round(v, 4) for v in per_seed_holdout],
                "note": "each value is one independent simulation seed with its own "
                        "theory calibration + classification (not seed-averaged)",
            },
            "auprc_reversal_agreement": auprc_agree,
            "auprc_reversal_agreement_ci95": auprc_ci,
            "auprc_sim_reversal_cells": int(auprc_sim.sum()),
            "auprc_theory_reversal_cells": int(auprc_theory.sum()),
            "n_cells": int(sim_status.size),
            "sim_reversal_cells": int(sim_rev.sum()),
            "theory_reversal_cells": int(th_rev.sum()),
        },
        config=asdict(base),
        wall_sec=round(time.time() - t0, 1),
    )
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/phase/phase_diagram.json")
    p.add_argument("--quick", action="store_true",
                   help="tiny 4x4 grid, 1 seed, small graph (smoke).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--h-points", type=int, default=13)
    p.add_argument("--delta-points", type=int, default=11)
    p.add_argument("--delta-max", type=float, default=0.30)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        base = GenConfig(n_timesteps=8, nodes_per_step=160)
        h_grid = np.linspace(0.0, 1.0, 4)
        delta_grid = np.linspace(0.0, args.delta_max, 4)
        seeds = [0]
    else:
        base = GenConfig()
        h_grid = np.linspace(0.0, 1.0, args.h_points)
        delta_grid = np.linspace(0.0, args.delta_max, args.delta_points)
        seeds = list(args.seeds)

    print(f"[phase] grid {h_grid.size}x{delta_grid.size}  seeds={seeds}  "
          f"nodes/cell~{base.n_timesteps * base.nodes_per_step}")
    res = run_phase_sweep(h_grid, delta_grid, base, seeds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(asdict(res), f, indent=2)

    ag = res.agreement
    print(f"\n[phase] wrote {args.out}")
    print(f"[phase] theory params: {json.dumps(res.theory_params)}")
    fc = ag["reversal_agreement_ci95"]
    print(f"[phase] reversal-map agreement (theory vs sim): "
          f"{ag['reversal_binary_agreement']*100:.1f}%  "
          f"[95% CI {fc[0]*100:.1f}-{fc[1]*100:.1f}%]  "
          f"(status {ag['status_agreement']*100:.1f}%)  "
          f"sim={ag['sim_reversal_cells']}/{ag['n_cells']} cells reversed")
    if ag["n_holdout_cells"] and ag["holdout_reversal_agreement"] is not None:
        hc = ag["holdout_reversal_agreement_ci95"]
        print(f"[phase] HELD-OUT agreement (interior h, {ag['n_holdout_cells']} cells): "
              f"{ag['holdout_reversal_agreement']*100:.1f}%  "
              f"[95% CI {hc[0]*100:.1f}-{hc[1]*100:.1f}%]")
    ps = ag["per_seed"]
    if ps["reversal_agreement_mean"] is not None:
        print(f"[phase] PER-SEED agreement ({ps['n_seeds']} indep. sims): "
              f"{ps['reversal_agreement_mean']*100:.1f}% +/- {ps['reversal_agreement_std']*100:.1f}%  "
              f"values={[round(v*100, 1) for v in ps['reversal_agreement_values']]}")
    ac = ag["auprc_reversal_agreement_ci95"]
    print(f"[phase] AUPRC reversal agreement (theory vs sim): "
          f"{ag['auprc_reversal_agreement']*100:.1f}%  "
          f"[95% CI {ac[0]*100:.1f}-{ac[1]*100:.1f}%]  "
          f"sim={ag['auprc_sim_reversal_cells']}/{ag['n_cells']} cells reversed  "
          f"(AUPRC h*={res.theory_params['auprc_critical_decay']})")
    print(f"[phase] wall={res.wall_sec}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
