"""
utils/trainer.py
Training loop for all model types (Levels 1-4).

Supports:
  - Static GNNs  (GCN, GraphSAGE, GAT, EdgeGAT)
  - Temporal GNN (EvolveGCN) — snapshot-based training
  - Adaptive strategy switching via DriftDetector (Level 4)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Optional, List
from utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_static(
    model:     nn.Module,
    data:      Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler=None,
) -> float:
    model.train()
    data      = data.to(device)
    criterion = criterion.to(device)

    use_amp = (scaler is not None) and (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
        if hasattr(criterion, "forward") and "mask" in criterion.forward.__code__.co_varnames:
            loss = criterion(logits, data.y, data.train_mask)
        else:
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return loss.item()


def train_epoch_evolve(
    model:     nn.Module,
    snapshots: list,
    data:      Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.train()
    data      = data.to(device)
    criterion = criterion.to(device)
    snaps_dev = [(x.to(device), ei.to(device)) for x, ei in snapshots]

    logits = model(snaps_dev)

    if hasattr(criterion, "forward") and "mask" in criterion.forward.__code__.co_varnames:
        loss = criterion(logits, data.y, data.train_mask)
    else:
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    data:        Data,
    mask:        torch.Tensor,
    device:      torch.device,
    snapshots:   Optional[list] = None,
) -> Dict[str, float]:
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    if snapshots is not None:
        snaps_dev = [(x.to(device), ei.to(device)) for x, ei in snapshots]
        logits    = model(snaps_dev)
    else:
        logits = model(data.x, data.edge_index,
                       getattr(data, "edge_attr", None))

    return compute_metrics(logits, data.y, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Full training run — static models
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:       nn.Module,
    data:        Data,
    criterion:   nn.Module,
    config:      dict,
    device:      torch.device,
    wandb_run=None,
    verbose:     bool = True,
    drift_detector=None,
    fallback_criterion=None,
) -> Dict:
    """
    Full training loop with early stopping on test F1.

    Config keys: lr, weight_decay, epochs, patience, eval_every

    drift_detector: DriftDetector instance (L4 adaptive strategy)
    fallback_criterion: criterion to switch to when drift detected
    """
    lr           = config.get("lr",           1e-3)
    weight_decay = config.get("weight_decay", 5e-4)
    epochs       = config.get("epochs",       300)
    patience     = config.get("patience",     30)
    eval_every   = config.get("eval_every",   10)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model     = model.to(device)
    criterion = criterion.to(device)

    # AMP GradScaler — only active on CUDA
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1      = 0.0
    best_metrics = {}
    patience_ctr = 0
    history      = {"loss": [], "train_f1": [], "test_f1": []}
    strategy_switches = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch_static(model, data, optimizer, criterion, device,
                                  scaler=scaler)
        scheduler.step()
        if hasattr(criterion, "step_epoch"):
            criterion.step_epoch()

        if epoch % eval_every == 0:
            tr = evaluate(model, data, data.train_mask, device)
            te = evaluate(model, data, data.test_mask,  device)

            history["loss"].append(loss)
            history["train_f1"].append(tr["f1"])
            history["test_f1"].append(te["f1"])

            # Drift detection (L4 adaptive)
            if drift_detector is not None and fallback_criterion is not None:
                if drift_detector.update(te["f1"]):
                    if verbose:
                        print(f"  [Drift detected @ epoch {epoch}] "
                              f"switching strategy → weighted CE")
                    criterion = fallback_criterion.to(device)
                    strategy_switches.append(epoch)

            if wandb_run:
                wandb_run.log({
                    "epoch":        epoch,
                    "loss":         loss,
                    "train/f1":     tr["f1"],
                    "test/f1":      te["f1"],
                    "test/prec":    te["precision"],
                    "test/recall":  te["recall"],
                    "test/auc":     te["auc"],
                })

            if te["f1"] > best_f1:
                best_f1      = te["f1"]
                best_metrics = te.copy()
                patience_ctr = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_ctr += 1

            if verbose and epoch % (eval_every * 5) == 0:
                print(
                    f"Ep {epoch:4d} | Loss {loss:.4f} | "
                    f"TrainF1 {tr['f1']:.4f} | "
                    f"TestF1 {te['f1']:.4f} | "
                    f"P {te['precision']:.4f} | R {te['recall']:.4f} | "
                    f"{time.time()-t0:.0f}s"
                )

            if patience_ctr >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return {
        "best_metrics":      best_metrics,
        "history":           history,
        "epochs_run":        epoch,
        "strategy_switches": strategy_switches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full training run — EvolveGCN
# ─────────────────────────────────────────────────────────────────────────────

def train_evolve(
    model:     nn.Module,
    data:      Data,
    criterion: nn.Module,
    config:    dict,
    device:    torch.device,
    verbose:   bool = True,
    wandb_run=None,
) -> Dict:
    """
    Training loop for EvolveGCN.
    Builds time-ordered training snapshots (t=1..34) each epoch.
    """
    from utils.temporal import build_snapshots

    lr           = config.get("lr",           5e-4)
    weight_decay = config.get("weight_decay", 5e-4)
    epochs       = config.get("epochs",       200)
    patience     = config.get("patience",     25)
    eval_every   = config.get("eval_every",   10)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model     = model.to(device)
    criterion = criterion.to(device)

    train_snaps = build_snapshots(data, time_range=range(1, 35))
    test_snaps  = build_snapshots(data, time_range=range(35, 50))

    if verbose:
        print(
            f"Built {len(train_snaps)} train snapshots and "
            f"{len(test_snaps)} test snapshots"
        )
        print(
            f"Running {epochs} epochs, eval every {eval_every} epochs",
            flush=True,
        )

    best_f1      = 0.0
    best_metrics = {}
    patience_ctr = 0
    history      = {"loss": [], "train_f1": [], "test_f1": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch_evolve(
            model, train_snaps, data, optimizer, criterion, device
        )
        scheduler.step()

        if verbose and epoch == 1:
            print(f"Finished epoch {epoch}", flush=True)

        if epoch % eval_every == 0:
            tr = evaluate(model, data, data.train_mask, device, train_snaps)
            te = evaluate(model, data, data.test_mask,  device, test_snaps)

            history["loss"].append(loss)
            history["train_f1"].append(tr["f1"])
            history["test_f1"].append(te["f1"])

            if wandb_run:
                wandb_run.log({
                    "epoch":    epoch,
                    "loss":     loss,
                    "test/f1":  te["f1"],
                    "test/auc": te["auc"],
                })

            if te["f1"] > best_f1:
                best_f1      = te["f1"]
                best_metrics = te.copy()
                patience_ctr = 0
                torch.save(model.state_dict(), "best_evolve_model.pt")
            else:
                patience_ctr += 1

            if verbose:
                print(
                    f"Ep {epoch:4d} | Loss {loss:.4f} | "
                    f"TrainF1 {tr['f1']:.4f} | TestF1 {te['f1']:.4f} | "
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )

            if patience_ctr >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return {"best_metrics": best_metrics, "history": history, "epochs_run": epoch}
