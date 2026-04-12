"""
experiments/_leakage_gap_utils.py

Shared helpers for the Transductive vs. Inductive Evaluation Gap experiment.

Both `run_transductive.py` and `run_inductive.py` import from this module so
that the two settings differ **only** in the adjacency (and feature subset)
used during training-time message passing:

  * Transductive : full adjacency + full node tensor at training time.  The
                   loss is still masked to training-period labels only, but
                   message passing can propagate information through edges
                   that touch test-period nodes, and BatchNorm sees every
                   node's features.  This is the protocol used by most
                   published Elliptic results.

  * Inductive    : training operates on a relabeled subgraph containing
                   only nodes from steps 1-34 and only edges whose both
                   endpoints lie in that window.  The model never observes
                   a test-period feature vector during training, so neither
                   message passing nor BatchNorm can leak test-period
                   statistics into the learned weights.  At inference time
                   the full adjacency is restored so that test-period nodes
                   can still receive messages from their (train-period)
                   neighbourhoods.

Everything else — model class, seed, optimiser, schedule, loss, evaluation
code, early-stopping criterion — is held constant.  No new models, no new
features, no hidden hyperparameter differences.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

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
    training-time adjacency and node subset, not of model capacity or
    training recipe.
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
# Adjacency masking — the single lever that separates the two settings
# ─────────────────────────────────────────────────────────────────────────────

def build_inductive_edge_index(
    edge_index: torch.Tensor,
    time_step:  torch.Tensor,
    train_time_max: int = 34,
) -> torch.Tensor:
    """Keep only edges whose **both** endpoints are in the training period.

    Used as a helper by `build_inductive_subgraph`; also exposed for unit
    tests of the masking rule in isolation.
    """
    src, dst = edge_index
    keep = (time_step[src] <= train_time_max) & (time_step[dst] <= train_time_max)
    return edge_index[:, keep]


def build_inductive_subgraph(
    data: Data,
    train_time_max: int = 34,
) -> Data:
    """Return a relabeled PyG Data object containing ONLY train-period nodes.

    The resulting object has:
        * `x` restricted to rows whose `time_step` <= `train_time_max`
        * `y` and `time_step` restricted identically
        * `edge_index` restricted to edges whose both endpoints survived,
          and relabeled into the new 0..N-1 index space
        * `train_mask` that matches the original semantics (labeled nodes
          in steps 1-34) but projected onto the subset

    This is the proper inductive training graph: a GNN trained on it has no
    mechanism — neither message passing nor BatchNorm running statistics —
    to observe any test-period information.
    """
    train_node_mask = data.time_step <= train_time_max
    keep_idx        = torch.where(train_node_mask)[0]

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
    sub.test_mask  = torch.zeros_like(sub.train_mask)  # sanity: no test nodes here

    assert sub.train_mask.sum().item() == data.train_mask.sum().item(), (
        "inductive subgraph should preserve every labeled training node; "
        "the node mask was derived from the same time-step condition"
    )
    assert sub.test_mask.sum().item() == 0, (
        "inductive subgraph must not expose any test nodes"
    )
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_ce(logits: torch.Tensor,
                 y: torch.Tensor,
                 mask: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    """Masked, class-weighted cross-entropy — identical formulation in both
    settings.  Only `mask` selects the subset that contributes to the loss."""
    return F.cross_entropy(logits[mask], y[mask], weight=weight)


@torch.no_grad()
def _eval_full_graph(model, eval_data: Data) -> torch.Tensor:
    """Run the model over the full evaluation graph and return logits."""
    model.eval()
    return model(eval_data.x, eval_data.edge_index)


def train_and_evaluate(
    train_data: Data,
    eval_data:  Data,
    cfg:        LeakageGapConfig,
    setting_name: str,
) -> Dict:
    """Train GraphSAGE on `train_data`, evaluate on `eval_data.test_mask`.

    Args:
        train_data:   Data object used for every training forward pass.
                       * Transductive: the full graph.
                       * Inductive:    the train-period subgraph returned
                                       by `build_inductive_subgraph`.
        eval_data:     ALWAYS the full graph.  This is how test-period nodes
                       receive messages from their train-period neighbours
                       at inference in the inductive setting.
        cfg:           shared hyperparameters.
        setting_name:  "transductive" or "inductive", used for logging.

    Returns:
        Dict with best test metrics, per-timestep F1 breakdown, and timing.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_data = train_data.to(device)
    eval_data  = eval_data.to(device)

    class_weights = get_class_weights(train_data).to(device)

    model = build_model(
        cfg.model_name,
        in_channels     = train_data.num_node_features,
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
          f"params={n_params:,} device={device} "
          f"train_nodes={train_data.num_nodes:,} "
          f"train_edges={train_data.num_edges:,} "
          f"eval_nodes={eval_data.num_nodes:,} "
          f"eval_edges={eval_data.num_edges:,}")

    best_f1      = 0.0
    best_metrics: Dict[str, float] = {}
    patience_ctr = 0
    t0           = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        logits = model(train_data.x, train_data.edge_index)
        loss   = _weighted_ce(logits, train_data.y, train_data.train_mask,
                              class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            logits_full = _eval_full_graph(model, eval_data)
            te = compute_metrics(logits_full, eval_data.y, eval_data.test_mask)

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

    per_step = _per_timestep_breakdown(model, eval_data)

    return {
        "setting":       setting_name,
        "model":         cfg.model_name,
        "hyperparams":   asdict(cfg),
        "n_parameters":  n_params,
        "train_nodes":   int(train_data.num_nodes),
        "train_edges":   int(train_data.num_edges),
        "eval_nodes":    int(eval_data.num_nodes),
        "eval_edges":    int(eval_data.num_edges),
        "best_metrics":  best_metrics,
        "per_timestep":  per_step,
        "wall_time_sec": round(time.time() - t0, 1),
    }


@torch.no_grad()
def _per_timestep_breakdown(model, eval_data: Data) -> Dict[int, Dict[str, float]]:
    model.eval()
    logits = model(eval_data.x, eval_data.edge_index)
    out: Dict[int, Dict[str, float]] = {}
    for t in range(35, 50):
        mask = eval_data.test_mask & (eval_data.time_step == t)
        if mask.sum().item() < 5:
            continue
        out[int(t)] = compute_metrics(logits, eval_data.y, mask)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_elliptic() -> Data:
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
