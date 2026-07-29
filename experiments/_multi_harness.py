"""
experiments/_multi_harness.py

Shared training / evaluation utilities for multi-dataset, multi-model
experiments. Used by every ``run_multi_*`` script so the protocol, seeding,
optimiser and metrics are held constant across runs.

Design choices
--------------
* **Always reports three numbers per run**: transductive F1, inductive F1,
  TPC+TTA F1 (our proposed fix applied on top of the inductive logits).
  This is the experimental core of the reviewer's ask — "show that the
  ranking reversal happens under modern models, and that TPC+TTA fixes
  it" — so we compute all three in one pass rather than letting each
  script reinvent the harness.

* **Temporal-aware inductive subgraph**: for each dataset we restrict the
  training graph to edges whose both endpoints have
  ``time_step <= train_max``, mirroring the existing Elliptic recipe. The
  inductive / transductive split is the *only* thing that differs
  between the two runs — same model instance, same seed, same optimiser
  schedule, same epochs.

* **Temporal models (``SnapshotTGN``)** have ``set_time_step`` called on
  the sub-graph that training sees and on the full graph that eval sees.
  The harness handles this automatically via the registry's category tag.

* **Single results schema** across all scripts:
      {
        "dataset", "model", "seed",
        "transductive": {"f1", "precision", "recall", "auc"},
        "inductive":    {"f1", "precision", "recall", "auc"},
        "tpc_tta":      {variant: {"f1", "precision", "recall"} for variant in
                         ("raw", "temp", "prior", "thresh", "tpc_tta")},
        "n_params", "train_edges", "eval_edges", "wall_sec"
      }
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.datasets import FraudDataset, load_dataset                        # noqa: E402
from data.datasets.base import describe_dataset, get_class_weights          # noqa: E402
from models.registry import build_model, count_parameters                   # noqa: E402
from models.temporal_calibration import apply_tpc_tta, _binary_f1           # noqa: E402
from utils.metrics import compute_metrics                                   # noqa: E402
from experiments.run_metadata import build_run_metadata                     # noqa: E402
from experiments.result_audit import write_json_atomic                      # noqa: E402
from utils.reproducibility import set_global_determinism                    # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    dataset:         str   = "elliptic"
    model:           str   = "sage"
    scaler_mode:     str   = "train_only"
    hidden_channels: int   = 256
    num_layers:      int   = 3
    dropout:         float = 0.5

    lr:              float = 1e-3
    weight_decay:    float = 5e-4
    epochs:          int   = 200
    eval_every:      int   = 10
    patience:        int   = 40

    seed:            int   = 42
    device:          str   = "auto"

    # TPC/TTA
    tpc_window:      int   = 3

    # Outputs
    results_dir:     str   = "results/multi"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility + device
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed all RNGs and request deterministic kernels (see utils.reproducibility)."""
    set_global_determinism(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_inductive_view(data: FraudDataset, train_max: int) -> FraudDataset:
    """Relabelled subgraph with only nodes whose ``time_step <= train_max``.

    Differs from the existing Elliptic helper only in that it preserves the
    multi-dataset schema (``val_mask`` is carried through so early stopping
    can still use the in-window validation slice).
    """
    keep_nodes = data.time_step <= train_max
    keep_idx   = torch.where(keep_nodes)[0]

    sub_ei, _ = subgraph(
        keep_idx,
        data.edge_index,
        relabel_nodes = True,
        num_nodes     = data.num_nodes,
    )
    sub = Data(
        x          = data.x[keep_idx],
        y          = data.y[keep_idx],
        edge_index = sub_ei,
    )
    sub.time_step  = data.time_step[keep_idx]
    sub.train_mask = data.train_mask[keep_idx]
    sub.val_mask   = data.val_mask[keep_idx]
    sub.test_mask  = torch.zeros_like(sub.train_mask)
    sub.name       = data.name
    sub.meta       = data.meta
    sub.pos_label  = data.pos_label
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Forward-pass helpers aware of model category
# ─────────────────────────────────────────────────────────────────────────────

def _forward(built, data: Data) -> torch.Tensor:
    model = built.model
    if built.category == "temporal":
        model.set_time_step(data.time_step.to(data.x.device))
    edge_attr = getattr(data, "edge_attr", None)
    return model(data.x, data.edge_index, edge_attr)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_ce(logits, y, mask, weight):
    return F.cross_entropy(logits[mask], y[mask], weight=weight)


def _train(
    built,
    train_data: Data,
    eval_data:  Data,
    cfg:        RunConfig,
    setting:    str,
) -> Tuple[torch.Tensor, Dict]:
    """Train the wrapped model; return full-graph logits and timing info."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_data = train_data.to(device)
    eval_data  = eval_data.to(device)
    built.model.to(device)
    built.model.reset_parameters()

    class_weights = get_class_weights(train_data).to(device)

    optimizer = torch.optim.AdamW(
        built.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )

    best_val, best_state, patience_ctr, t0 = 0.0, None, 0, time.time()
    for ep in range(1, cfg.epochs + 1):
        built.model.train()
        logits = _forward(built, train_data)
        loss   = _weighted_ce(logits, train_data.y, train_data.train_mask,
                              class_weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(built.model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % cfg.eval_every == 0 or ep == 1:
            built.model.eval()
            with torch.no_grad():
                vm = eval_data.val_mask
                if vm.sum().item() == 0:
                    # Fall back to train loss improvement — some datasets
                    # might end up with no val nodes after filtering.
                    if loss.item() < -best_val + 1e-6:
                        best_val = -loss.item()
                        best_state = {k: v.detach().clone()
                                      for k, v in built.model.state_dict().items()}
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                else:
                    eval_logits = _forward(built, eval_data)
                    m = compute_metrics(eval_logits, eval_data.y, vm)
                    if m["f1"] > best_val:
                        best_val = m["f1"]
                        best_state = {k: v.detach().clone()
                                      for k, v in built.model.state_dict().items()}
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break

    if best_state is not None:
        built.model.load_state_dict(best_state)
    built.model.eval()
    with torch.no_grad():
        eval_logits = _forward(built, eval_data)

    return eval_logits.detach(), {
        "best_val_f1": round(float(best_val), 4),
        "wall_sec":    round(time.time() - t0, 1),
        "n_epochs":    ep,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TPC+TTA evaluation on top of inductive logits
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_from_preds(preds: np.ndarray, y: np.ndarray, mask: np.ndarray,
                        pos_label: int = 1) -> Dict[str, float]:
    y_m  = y[mask]
    p_m  = preds[mask]
    f1   = _binary_f1(y_m, p_m, pos_label=pos_label)
    tp = int(((p_m == pos_label) & (y_m == pos_label)).sum())
    fp = int(((p_m == pos_label) & (y_m != pos_label)).sum())
    fn = int(((p_m != pos_label) & (y_m == pos_label)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    return {"f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)}


def _apply_tpc(
    logits:    torch.Tensor,
    data:      FraudDataset,
    cfg:       RunConfig,
) -> Dict[str, Dict[str, float]]:
    """Run TPC+TTA on the inductive logits and report the full mechanism ablation.

    Variants reported (matching the feature-only study run_tpc_tta_eval.py):
        raw        : argmax on raw logits (inductive baseline, no fix)
        temp       : temperature-scaled argmax (calibration only)
        prior      : temperature + temporal prior correction (no threshold)
        thresh     : temperature + F1-optimal threshold (no prior)
        tpc_tta    : full proposed method (temperature + prior + threshold)
    """
    res = apply_tpc_tta(
        logits_full = logits,
        y_all       = data.y,
        time_step   = data.time_step,
        train_mask  = data.train_mask,
        val_mask    = data.val_mask,
        test_mask   = data.test_mask,
        pos_label   = int(data.pos_label),
        window      = cfg.tpc_window,
    )
    test = data.test_mask.cpu().numpy().astype(bool)
    y    = data.y.cpu().numpy()
    pl   = int(data.pos_label)

    return {
        "raw":      _metrics_from_preds(res["preds_baseline"], y, test, pl),
        "temp":     _metrics_from_preds(res["preds_temp"],     y, test, pl),
        "prior":    _metrics_from_preds(res["preds_prior"],    y, test, pl),
        "thresh":   _metrics_from_preds(res["preds_thresh"],   y, test, pl),
        "tpc_tta":  _metrics_from_preds(res["preds_tpc_tta"],  y, test, pl),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_single(cfg: RunConfig) -> Dict:
    """Load dataset, train under both protocols, apply TPC+TTA, return result dict."""
    data = load_dataset(cfg.dataset, scaler_mode=cfg.scaler_mode)
    print(describe_dataset(data))

    train_max = int(data.meta.get("split_train_max",
                                  # fall back to the implicit boundary: the
                                  # largest bucket in train_mask
                                  int(data.time_step[data.train_mask].max().item())))

    # --- Transductive run: train on full graph, eval on full graph ---
    built_t = build_model(
        cfg.model,
        in_channels     = data.num_node_features,
        hidden_channels = cfg.hidden_channels,
        num_layers      = cfg.num_layers,
        dropout         = cfg.dropout,
        out_channels    = 3,
    )
    logits_t, info_t = _train(built_t, data, data, cfg, setting="transductive")
    m_trans = compute_metrics(logits_t, data.y, data.test_mask)

    # --- Inductive run: train on subgraph, eval on full graph ---
    sub = build_inductive_view(data, train_max=train_max)
    built_i = build_model(
        cfg.model,
        in_channels     = data.num_node_features,
        hidden_channels = cfg.hidden_channels,
        num_layers      = cfg.num_layers,
        dropout         = cfg.dropout,
        out_channels    = 3,
    )
    logits_i, info_i = _train(built_i, sub, data, cfg, setting="inductive")
    m_ind = compute_metrics(logits_i, data.y, data.test_mask)

    # --- TPC+TTA on top of inductive logits ---
    tpc = _apply_tpc(logits_i, data, cfg)

    result = {
        "result_schema_version": "multi_v2",
        "dataset":      cfg.dataset,
        "model":        cfg.model,
        "seed":         cfg.seed,
        "hyperparams":  asdict(cfg),
        "protocol": {
            "transductive": "train and evaluate with full graph adjacency",
            "inductive": "train on nodes/edges with time_step <= train_max; evaluate on full graph",
            "model_selection": "validation mask when present; train-loss fallback only if val is empty",
            "tpc_tta": "temperature, temporal prior correction, and thresholding on inductive logits",
        },
        "split":        data.meta.get("split"),
        "scaler_mode":  data.meta.get("scaler_mode", cfg.scaler_mode),
        "n_params":     count_parameters(built_i.model),
        "transductive": {k: m_trans[k] for k in ("f1", "precision", "recall", "auc")},
        "inductive":    {k: m_ind[k]   for k in ("f1", "precision", "recall", "auc")},
        "tpc_tta":      tpc,
        "wall_sec": {
            "transductive": info_t["wall_sec"],
            "inductive":    info_i["wall_sec"],
        },
        "run_metadata": build_run_metadata("run_multi_dataset", asdict(cfg)),
    }
    return result


def run_grid(
    datasets: Iterable[str],
    models:   Iterable[str],
    seeds:    Iterable[int],
    base_cfg: RunConfig,
    out_dir:  str,
) -> List[Dict]:
    """Run every (dataset, model, seed) triple and persist results.

    Each result is written as an individual JSON file to make resuming a
    partial sweep trivial (no locking, no append-to-master-file race).
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for ds in datasets:
        for mdl in models:
            for sd in seeds:
                cfg = RunConfig(**{**asdict(base_cfg),
                                   "dataset": ds, "model": mdl, "seed": sd})
                tag = f"{ds}__{mdl}__seed{sd}.json"
                path = os.path.join(out_dir, tag)
                if os.path.exists(path):
                    print(f"[skip] {tag} (exists)")
                    with open(path) as f: results.append(json.load(f))
                    continue
                print(f"\n[run] {tag}")
                try:
                    r = run_single(cfg)
                except Exception as exc:
                    print(f"[err] {tag}: {exc}")
                    r = {
                        "result_schema_version": "multi_v2",
                        "dataset": ds,
                        "model": mdl,
                        "seed": sd,
                        "hyperparams": asdict(cfg),
                        "error": str(exc),
                        "run_metadata": build_run_metadata("run_multi_dataset", asdict(cfg)),
                    }
                write_json_atomic(path, r)
                results.append(r)
    return results
