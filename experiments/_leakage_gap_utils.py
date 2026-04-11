"""
experiments/_leakage_gap_utils.py

Shared helpers for the Transductive vs. Inductive Evaluation Gap experiment.

Both `run_transductive.py` and `run_inductive.py` import from this module so
that the two settings differ **only** in the adjacency used for message
passing at training time:

  * Transductive : full adjacency (train + test nodes/edges visible)
  * Inductive    : train-period adjacency only (both endpoints in steps 1-34)

Everything else — model, seed, optimiser, schedule, loss, evaluation —
is held constant.  No new models, no new features.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Make sure we can import the project modules when invoked as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import load_elliptic_raw, preprocess, get_class_weights  # noqa: E402
from models.gnn import build_model, count_parameters  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakageGapConfig:
    """Hyperparameters shared by both transductive and inductive runs.

    Kept identical on purpose — the experiment isolates the effect of the
    adjacency matrix, not of model capacity or training recipe.
    """
    model_name: str       = "sage"
    hidden_channels: int  = 256
    num_layers: int       = 3
    dropout: float        = 0.5

    lr: float             = 1e-3
    weight_decay: float   = 5e-4
    epochs: int           = 200
    eval_every: int       = 10
    patience: int         = 40

    seed: int             = 42
    device: str           = "auto"   # "auto" | "cpu" | "cuda" | "mps"


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
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
# Adjacency masking — the ONLY place where the two settings differ
# ─────────────────────────────────────────────────────────────────────────────

def build_inductive_edge_index(
    edge_index: torch.Tensor,
    time_step:  torch.Tensor,
    train_time_max: int = 34,
) -> torch.Tensor:
    """Keep only edges whose **both** endpoints are in the training period.

    Under this masking, GraphSAGE message passing during training cannot see
    any test-period node features — a strict inductive setup.  At evaluation
    time the caller should swap back to the full adjacency so that test-period
    neighbourhoods are available for inference.
    """
    src, dst = edge_index
    keep = (time_step[src] <= train_time_max) & (time_step[dst] <= train_time_max)
    return edge_index[:, keep]


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_ce(logits: torch.Tensor,
                 y: torch.Tensor,
                 mask: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[mask], y[mask], weight=weight)


def _eval(model, data, train_edge_index, eval_edge_index, mask) -> Dict[str, float]:
    """Evaluate with the adjacency that should be visible at inference."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, eval_edge_index)
    return compute_metrics(logits, data.y, mask)


def train_and_evaluate(
    data,
    train_edge_index: torch.Tensor,
    eval_edge_index:  torch.Tensor,
    cfg: LeakageGapConfig,
    setting_name: str,
) -> Dict:
    """Train GraphSAGE with a given training adjacency and evaluate on test.

    Args:
        data:              PyG Data (full graph, all masks populated).
        train_edge_index:  adjacency used for message passing during training.
        eval_edge_index:   adjacency used for evaluation.
        cfg:               shared hyperparameters.
        setting_name:      "transductive" or "inductive", used for logging.

    Returns:
        Dict with best test metrics, per-timestep F1 breakdown, and timing.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    data              = data.to(device)
    train_edge_index  = train_edge_index.to(device)
    eval_edge_index   = eval_edge_index.to(device)

    class_weights = get_class_weights(data).to(device)

    model = build_model(
        cfg.model_name,
        in_channels     = data.num_node_features,
        hidden_channels = cfg.hidden_channels,
        num_layers      = cfg.num_layers,
        dropout         = cfg.dropout,
        out_channels    = 3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    n_params = count_parameters(model)
    print(f"[{setting_name}] model={cfg.model_name} "
          f"params={n_params:,} device={device}")

    best_f1      = 0.0
    best_metrics: Dict[str, float] = {}
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        logits = model(data.x, train_edge_index)
        loss   = _weighted_ce(logits, data.y, data.train_mask, class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            te = _eval(model, data, train_edge_index, eval_edge_index,
                       data.test_mask)
            if te["f1"] > best_f1:
                best_f1      = te["f1"]
                best_metrics = te.copy()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch == 1 or epoch % (cfg.eval_every * 5) == 0:
                print(f"  ep {epoch:4d} | loss {loss.item():.4f} | "
                      f"testF1 {te['f1']:.4f} | bestF1 {best_f1:.4f} | "
                      f"{time.time()-t0:.0f}s")

            if patience_ctr >= cfg.patience:
                print(f"  early stop at epoch {epoch}")
                break

    # Per-timestep breakdown on test window
    per_step = _per_timestep_breakdown(model, data, eval_edge_index)

    return {
        "setting":       setting_name,
        "model":         cfg.model_name,
        "hyperparams":   asdict(cfg),
        "n_parameters":  n_params,
        "best_metrics":  best_metrics,
        "per_timestep":  per_step,
        "wall_time_sec": round(time.time() - t0, 1),
    }


@torch.no_grad()
def _per_timestep_breakdown(model, data, eval_edge_index) -> Dict[int, Dict[str, float]]:
    model.eval()
    logits = model(data.x, eval_edge_index)
    out: Dict[int, Dict[str, float]] = {}
    for t in range(35, 50):
        mask = data.test_mask & (data.time_step == t)
        if mask.sum().item() < 5:
            continue
        out[int(t)] = compute_metrics(logits, data.y, mask)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_elliptic():
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Result IO
# ─────────────────────────────────────────────────────────────────────────────

def save_results(result: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[io] wrote {path}")


def load_results(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
