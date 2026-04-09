"""
utils/trainer_minibatch.py
Mini-batch training using RandomNodeLoader — no pyg-lib or torch-sparse needed.

RandomNodeLoader randomly samples subsets of nodes each batch and runs
GNN on the full graph but only computes loss on the sampled nodes.
This breaks the full-graph memorisation problem without needing C++ extensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from typing import Dict, Optional
from utils.metrics import compute_metrics


def make_loaders(data: Data, batch_size: int = 512, num_neighbors: list = None):
    """
    RandomNodeLoader: splits nodes into batches, runs full graph per batch.
    Works on M4 with no C++ extensions.
    """
    train_loader = RandomNodeLoader(
        data,
        num_parts  = max(1, data.train_mask.sum().item() // batch_size),
        shuffle    = True,
        num_workers= 0,
    )
    test_loader = RandomNodeLoader(
        data,
        num_parts  = max(1, data.test_mask.sum().item() // batch_size),
        shuffle    = False,
        num_workers= 0,
    )
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss  = 0
    total_nodes = 0

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        # Only compute loss on labeled train nodes in this batch
        mask    = batch.train_mask & (batch.y != 0)
        if mask.sum() == 0:
            continue

        if hasattr(criterion, 'forward') and \
                'mask' in criterion.forward.__code__.co_varnames:
            loss = criterion(logits, batch.y, mask)
        else:
            loss = F.cross_entropy(logits[mask], batch.y[mask])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item() * mask.sum().item()
        total_nodes += mask.sum().item()

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, use_test: bool = True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        mask = (batch.test_mask if use_test else batch.train_mask) & (batch.y != 0)
        if mask.sum() == 0:
            continue

        probs  = torch.softmax(logits[mask], dim=-1)[:, 1]
        preds  = logits[mask].argmax(dim=-1)
        labels = batch.y[mask]

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    if not all_preds:
        return {"f1": 0, "precision": 0, "recall": 0, "accuracy": 0, "auc": 0}

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    from sklearn.metrics import (f1_score, precision_score,
                                  recall_score, roc_auc_score)
    preds_bin  = (all_preds  == 1).astype(int)
    labels_bin = (all_labels == 1).astype(int)

    f1        = f1_score(labels_bin,        preds_bin, zero_division=0)
    precision = precision_score(labels_bin, preds_bin, zero_division=0)
    recall    = recall_score(labels_bin,    preds_bin, zero_division=0)
    acc       = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(labels_bin, all_probs)
    except Exception:
        auc = float("nan")

    return {
        "f1":        round(float(f1),        4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "accuracy":  round(float(acc),       4),
        "auc":       round(float(auc),       4),
    }


def train(
    model:     nn.Module,
    data:      Data,
    criterion: nn.Module,
    config:    dict,
    device:    torch.device,
    wandb_run  = None,
    verbose:   bool = True,
) -> Dict:
    lr            = config.get("lr",            1e-3)
    weight_decay  = config.get("weight_decay",  5e-4)
    epochs        = config.get("epochs",        300)
    patience      = config.get("patience",      30)
    eval_every    = config.get("eval_every",    5)
    batch_size    = config.get("batch_size",    512)

    train_loader, test_loader = make_loaders(data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model     = model.to(device)
    criterion = criterion.to(device)

    best_f1      = 0.0
    best_metrics = {}
    patience_ctr = 0
    history      = {"loss": [], "train_f1": [], "test_f1": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if hasattr(criterion, "step_epoch"):
            criterion.step_epoch()

        if epoch % eval_every == 0:
            tr = evaluate_loader(model, train_loader, device, use_test=False)
            te = evaluate_loader(model, test_loader,  device, use_test=True)

            history["loss"].append(loss)
            history["train_f1"].append(tr["f1"])
            history["test_f1"].append(te["f1"])

            if wandb_run:
                wandb_run.log({
                    "epoch":       epoch,
                    "loss":        loss,
                    "train/f1":    tr["f1"],
                    "test/f1":     te["f1"],
                    "test/prec":   te["precision"],
                    "test/recall": te["recall"],
                    "test/auc":    te["auc"],
                })

            if te["f1"] > best_f1:
                best_f1      = te["f1"]
                best_metrics = te.copy()
                patience_ctr = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_ctr += 1

            if verbose and epoch % (eval_every * 4) == 0:
                print(
                    f"Ep {epoch:4d} | Loss {loss:.4f} | "
                    f"TrainF1 {tr['f1']:.4f} | "
                    f"TestF1 {te['f1']:.4f} | "
                    f"P {te['precision']:.4f} | "
                    f"R {te['recall']:.4f} | "
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
        "strategy_switches": [],
    }