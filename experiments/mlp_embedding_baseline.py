"""
experiments/mlp_embedding_baseline.py

FIX 5b — the non-negotiable test of the hybrid claim.

The paper claims F1=0.807 for a concat hybrid (GNN-penultimate || raw → RF)
and attributes the lift to graph-derived representations carrying
complementary signal. If a parallel pipeline that swaps the GNN for an
MLP (SAGE-with-no-edges, i.e. the paper's own "MLP" baseline) matches
that number, the graph-signal claim is false — the lift would come from
any nonlinear penultimate projector, not from graph structure.

Protocol
--------
Everything is strictly inductive:

  * Encoder training   : inductive subgraph (time_step <= 34), relabeled.
                         For the MLP, the encoder is the same SAGE
                         architecture with edge_index emptied, so message
                         passing is disabled end-to-end.  This matches
                         the paper's own MLP definition exactly.

  * Embedding extract  : full graph at inference.  For the GNN this
                         lets test-period nodes receive messages from
                         train-period neighbours (standard inductive
                         inference).  For the MLP edges are ignored.

  * RF training        : features built on the full graph; train on the
                         labeled train_mask rows, evaluate on labeled
                         test_mask rows.  The labels are binary (illicit
                         vs. licit), matching every other binary F1
                         number in the paper.

Cells
-----
  A.  [GNN_emb  || raw]  →  RF      (paper's F1=0.807 claim)
  B.  [MLP_emb  || raw]  →  RF      (the non-negotiable replication)
  C.  [GNN_emb]          →  RF      (GNN-emb contribution in isolation)
  D.  [MLP_emb]          →  RF      (MLP-emb contribution in isolation)
  E.  [raw]              →  RF      (known ~0.820 reference; cheap to rerun)

Default is 3 seeds for a fast screen.  Expand to 10 with --seeds 10 if
the headline comparison (A vs B) is marginal.

Outputs
-------
  results/fix5b_mlp_embedding/per_run.csv
  results/fix5b_mlp_embedding/summary.csv
  results/fix5b_mlp_embedding/summary.json

Usage
-----
  python experiments/mlp_embedding_baseline.py --seeds 3 --epochs 200
  python experiments/mlp_embedding_baseline.py --seeds 10 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torch_geometric.data import Data  # noqa: E402

from data.dataset import (  # noqa: E402
    get_class_weights, load_elliptic_raw, preprocess,
)
from experiments._leakage_gap_utils import build_inductive_subgraph  # noqa: E402
from models.gnn import build_model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility and device
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ─────────────────────────────────────────────────────────────────────────────
# Graph variants
# ─────────────────────────────────────────────────────────────────────────────

def _strip_edges(data: Data) -> Data:
    """Return a copy of `data` with edge_index emptied.

    Used to turn a SAGE model into the paper's MLP baseline: with no edges
    the SAGEConv message-passing term is zero, so each convolution reduces
    to the self-linear W_1 x_i. End-to-end the model becomes an MLP with
    BatchNorm and residual-free stacked linears.
    """
    empty = torch.zeros(2, 0, dtype=torch.long)
    out = Data(
        x          = data.x,
        edge_index = empty,
        y          = data.y,
    )
    for attr in ("train_mask", "test_mask", "time_step"):
        if hasattr(data, attr):
            setattr(out, attr, getattr(data, attr))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training (matches _leakage_gap_utils recipe; inductive-safe)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    hidden_channels: int = 256
    num_layers:      int = 3
    dropout:         float = 0.5
    lr:              float = 1e-3
    weight_decay:    float = 5e-4
    epochs:          int = 200
    eval_every:      int = 10
    patience:        int = 40


def _train_encoder(
    train_data: Data,
    eval_data:  Data,
    cfg:        TrainConfig,
    device:     torch.device,
    seed:       int,
    tag:        str,
) -> torch.nn.Module:
    """Train a SAGE encoder under strict inductive protocol.

    Args:
        train_data: inductive subgraph (or no-edges version of same).
        eval_data:  the full graph; only used for early-stopping signal via
                    held-out test F1.  Labels observed for early-stop are
                    test labels but the encoder parameters are only shaped
                    by the training-period adjacency/features.  This is the
                    same protocol used by run_inductive.py.
        cfg:        training hyperparameters.
        device:     torch device.
        seed:       rng seed.
        tag:        "gnn" or "mlp" (logging only).
    """
    set_seed(seed)

    train_data = train_data.to(device)
    eval_data  = eval_data.to(device)

    class_weights = get_class_weights(train_data).to(device)

    model = build_model(
        "sage",
        in_channels     = train_data.num_node_features,
        hidden_channels = cfg.hidden_channels,
        num_layers      = cfg.num_layers,
        dropout         = cfg.dropout,
        out_channels    = 3,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_f1, best_state, stale = 0.0, None, 0
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        logits = model(train_data.x, train_data.edge_index)
        loss = F.cross_entropy(
            logits[train_data.train_mask],
            train_data.y[train_data.train_mask],
            weight = class_weights,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if ep % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                le = model(eval_data.x, eval_data.edge_index)
                preds_b = (le[eval_data.test_mask].argmax(-1) == 1).cpu().numpy()
                ytrue_b = (eval_data.y[eval_data.test_mask] == 1).cpu().numpy()
                f1 = f1_score(ytrue_b, preds_b, zero_division=0)
            if f1 > best_f1:
                best_f1     = f1
                best_state  = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += cfg.eval_every
            if ep % 20 == 0 or ep == cfg.eval_every:
                print(f"      [{tag}] ep {ep:4d}  loss={loss.item():.4f}  "
                      f"testF1={f1:.4f}  bestF1={best_f1:.4f}  "
                      f"stale={stale}  t={time.time()-t0:.0f}s", flush=True)
            if stale >= cfg.patience:
                print(f"      [{tag}] early stop at ep {ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    [{tag}] seed={seed}  best_F1(test/earlystop)={best_f1:.4f}  "
          f"time={time.time()-t0:.0f}s", flush=True)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction via forward hook on classifier
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_penultimate(
    model: torch.nn.Module,
    data:  Data,
    device: torch.device,
) -> np.ndarray:
    """Extract per-node penultimate features (input to the classifier).

    Uses a forward hook on `model.classifier`.  Works for any model whose
    last layer is a single Linear named `classifier` — i.e. GCN, GraphSAGE,
    GAT, EdgeGAT.  Returns a numpy array of shape [N, hidden_channels].
    """
    captured = {}

    def _hook(_module, inputs, _output):
        captured["emb"] = inputs[0].detach().cpu()

    hook = model.classifier.register_forward_hook(_hook)
    model.eval()
    data = data.to(device)
    _ = model(data.x, data.edge_index)
    hook.remove()
    return captured["emb"].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# RF evaluation (binary: illicit=1 vs licit=2)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_rf_and_eval(
    feat_mat:   np.ndarray,
    y:          np.ndarray,
    train_mask: np.ndarray,
    test_mask:  np.ndarray,
    seed:       int,
    n_boot:     int = 10_000,
) -> Dict[str, float]:
    """Train RF on train-split features, evaluate on test-split.

    Labels passed are 0/1/2 (0 = unknown, 1 = illicit, 2 = licit). Only
    labeled rows from the masks are used.  Metrics are binary (illicit vs
    non-illicit).  95% bootstrap CIs are computed on the test-set row
    bootstrap at `n_boot` resamples.
    """
    X_tr = feat_mat[train_mask]
    y_tr = y[train_mask]
    X_te = feat_mat[test_mask]
    y_te = y[test_mask]

    keep_tr = y_tr != 0
    X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
    keep_te = y_te != 0
    X_te, y_te = X_te[keep_te], y_te[keep_te]

    clf = RandomForestClassifier(
        n_estimators = 300,
        min_samples_leaf = 2,
        class_weight = "balanced",
        random_state = seed,
        n_jobs = -1,
    )
    clf.fit(X_tr, y_tr)

    preds   = clf.predict(X_te)
    proba   = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    prob_fraud = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(y_te))

    y_bin      = (y_te == 1).astype(int)
    preds_bin  = (preds == 1).astype(int)

    f1_point  = f1_score(y_bin, preds_bin, zero_division=0)
    prec      = precision_score(y_bin, preds_bin, zero_division=0)
    rec       = recall_score(y_bin, preds_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_bin, prob_fraud)
    except ValueError:
        auc = float("nan")

    rng  = np.random.default_rng(seed)
    n    = len(y_bin)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        boot[i] = f1_score(y_bin[idx], preds_bin[idx], zero_division=0)
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])

    return {
        "f1":        float(f1_point),
        "precision": float(prec),
        "recall":    float(rec),
        "auc":       float(auc),
        "f1_ci_lo":  float(ci_lo),
        "f1_ci_hi":  float(ci_hi),
        "n_test":    int(n),
        "n_fraud":   int(y_bin.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_seed(
    seed:       int,
    full_data:  Data,
    train_sub:  Data,
    cfg:        TrainConfig,
    device:     torch.device,
    raw_numpy:  np.ndarray,
    y_numpy:    np.ndarray,
    tr_mask_np: np.ndarray,
    te_mask_np: np.ndarray,
) -> List[Dict]:
    """Run all five cells for one seed; return a list of row dicts."""

    # Encoders
    gnn_model = _train_encoder(train_sub,           full_data, cfg, device,
                               seed=seed, tag="gnn")
    mlp_model = _train_encoder(_strip_edges(train_sub),
                               _strip_edges(full_data), cfg, device,
                               seed=seed, tag="mlp")

    # Embeddings on the full graph
    emb_gnn = _extract_penultimate(gnn_model, full_data, device)
    emb_mlp = _extract_penultimate(mlp_model, _strip_edges(full_data), device)
    assert emb_gnn.shape[0] == full_data.num_nodes == emb_mlp.shape[0]

    rows: List[Dict] = []

    cells = [
        ("A_gnn_emb+raw", np.concatenate([emb_gnn, raw_numpy], axis=1)),
        ("B_mlp_emb+raw", np.concatenate([emb_mlp, raw_numpy], axis=1)),
        ("C_gnn_emb",     emb_gnn),
        ("D_mlp_emb",     emb_mlp),
        ("E_raw",         raw_numpy),
    ]

    for cell, feat_mat in cells:
        t_cell = time.time()
        print(f"    cell={cell:<15s}  training RF (dim={feat_mat.shape[1]}) …",
              flush=True)
        m = _fit_rf_and_eval(
            feat_mat   = feat_mat,
            y          = y_numpy,
            train_mask = tr_mask_np,
            test_mask  = te_mask_np,
            seed       = seed,
        )
        print(f"    cell={cell:<15s}  F1={m['f1']:.4f}  "
              f"[{m['f1_ci_lo']:.4f},{m['f1_ci_hi']:.4f}]  "
              f"AUC={m['auc']:.4f}  dim={feat_mat.shape[1]}  "
              f"t={time.time()-t_cell:.0f}s", flush=True)
        rows.append({"seed": seed, "cell": cell, "dim": feat_mat.shape[1], **m})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",     type=int, default=3,
                   help="how many seeds to run (0..seeds-1)")
    p.add_argument("--epochs",    type=int, default=200)
    p.add_argument("--hidden",    type=int, default=256)
    p.add_argument("--layers",    type=int, default=3)
    p.add_argument("--dropout",   type=float, default=0.5)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--patience",  type=int, default=40)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--out_dir",   type=str,
                   default="results/fix5b_mlp_embedding")
    p.add_argument("--n_boot",    type=int, default=10_000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    cfg = TrainConfig(
        hidden_channels = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        lr              = args.lr,
        epochs          = args.epochs,
        patience        = args.patience,
    )
    print(f"[fix5b] device={device}  seeds={args.seeds}  epochs={cfg.epochs}")

    # Data
    features, classes, edges = load_elliptic_raw()
    full_data = preprocess(features, classes, edges, normalize=True)
    train_sub = build_inductive_subgraph(full_data, train_time_max=34)

    print(f"[data] full   : {full_data.num_nodes:,} nodes  "
          f"{full_data.num_edges:,} edges  "
          f"{full_data.num_node_features} features")
    print(f"[data] trainS : {train_sub.num_nodes:,} nodes  "
          f"{train_sub.num_edges:,} edges "
          f"(train-subgraph, time<=34)")
    print(f"[data] raw concat dim (emb_hidden + raw) "
          f"= {args.hidden + full_data.num_node_features}  "
          f"(paper reports 421)")

    raw_numpy   = full_data.x.cpu().numpy()
    y_numpy     = full_data.y.cpu().numpy()
    tr_mask_np  = full_data.train_mask.cpu().numpy()
    te_mask_np  = full_data.test_mask.cpu().numpy()

    all_rows: List[Dict] = []
    for seed in range(args.seeds):
        print(f"\n─── seed {seed} ─────────────────────────────────────────")
        rows = _run_one_seed(
            seed       = seed,
            full_data  = full_data,
            train_sub  = train_sub,
            cfg        = cfg,
            device     = device,
            raw_numpy  = raw_numpy,
            y_numpy    = y_numpy,
            tr_mask_np = tr_mask_np,
            te_mask_np = te_mask_np,
        )
        all_rows.extend(rows)

        # Save incrementally so a crash mid-run doesn't lose prior seeds.
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.out_dir, "per_run.csv"), index=False,
        )

    # Aggregate
    df   = pd.DataFrame(all_rows)
    agg  = (
        df.groupby("cell")
          .agg(
              f1_mean   = ("f1",    "mean"),
              f1_std    = ("f1",    "std"),
              ci_lo_med = ("f1_ci_lo", "median"),
              ci_hi_med = ("f1_ci_hi", "median"),
              prec_mean = ("precision", "mean"),
              rec_mean  = ("recall", "mean"),
              auc_mean  = ("auc",  "mean"),
              seeds     = ("seed", "count"),
          )
          .reset_index()
    )
    agg_csv = os.path.join(args.out_dir, "summary.csv")
    agg.to_csv(agg_csv, index=False)

    # Pairwise test: the headline comparison is A vs B.
    def _cell_f1(df: pd.DataFrame, cell: str) -> np.ndarray:
        return df.loc[df["cell"] == cell, "f1"].values

    from scipy import stats
    def _pairwise(cell_a: str, cell_b: str) -> Dict:
        a = _cell_f1(df, cell_a)
        b = _cell_f1(df, cell_b)
        if len(a) < 2 or len(b) < 2:
            return {"a": cell_a, "b": cell_b, "t": None, "p": None, "d": None,
                    "a_mean": float(np.mean(a)) if len(a) else None,
                    "b_mean": float(np.mean(b)) if len(b) else None,
                    "n_a": len(a), "n_b": len(b)}
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
        pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
        d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")
        return {
            "a": cell_a, "b": cell_b,
            "a_mean": float(np.mean(a)), "a_std": float(np.std(a, ddof=1)),
            "b_mean": float(np.mean(b)), "b_std": float(np.std(b, ddof=1)),
            "delta":  float(np.mean(a) - np.mean(b)),
            "t": float(t), "p": float(pv), "cohens_d": float(d),
            "n_a": len(a), "n_b": len(b),
        }

    tests = [
        _pairwise("A_gnn_emb+raw", "B_mlp_emb+raw"),   # the critical test
        _pairwise("A_gnn_emb+raw", "E_raw"),
        _pairwise("B_mlp_emb+raw", "E_raw"),
        _pairwise("C_gnn_emb",     "D_mlp_emb"),
    ]

    summary = {
        "cfg":   cfg.__dict__,
        "seeds": args.seeds,
        "device": str(device),
        "per_cell": agg.to_dict(orient="records"),
        "pairwise_welch": tests,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Final report
    print("\n=== Aggregate across seeds ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Pairwise Welch t-tests on F1 ===")
    for t in tests:
        if t.get("p") is None:
            continue
        print(f"  {t['a']:<15s} vs {t['b']:<15s}  "
              f"Δ={t['delta']:+.4f}  t={t['t']:+.2f}  "
              f"p={t['p']:.4f}  d={t['cohens_d']:+.2f}  "
              f"(n={t['n_a']},{t['n_b']})")

    print(f"\n[io] wrote {agg_csv}")
    print(f"[io] wrote {os.path.join(args.out_dir, 'per_run.csv')}")
    print(f"[io] wrote {os.path.join(args.out_dir, 'summary.json')}")

    a_mean = df.loc[df["cell"] == "A_gnn_emb+raw", "f1"].mean()
    b_mean = df.loc[df["cell"] == "B_mlp_emb+raw", "f1"].mean()
    delta  = a_mean - b_mean
    print(f"\n[HEADLINE]  A (GNN_emb+raw) = {a_mean:.4f}")
    print(f"[HEADLINE]  B (MLP_emb+raw) = {b_mean:.4f}")
    print(f"[HEADLINE]  A - B           = {delta:+.4f}")
    if abs(delta) < 0.01:
        print("[HEADLINE]  |Δ| < 0.01  →  graph-embedding-adds-signal claim "
              "is NOT supported; rewrite §7.8 and abstract.")
    elif delta > 0:
        print("[HEADLINE]  A > B at this seed count.  Expand to 10 seeds "
              "and compute CI to confirm.")
    else:
        print("[HEADLINE]  B > A: MLP embeddings beat GNN embeddings. "
              "Graph-signal claim is affirmatively falsified.")


if __name__ == "__main__":
    main()
