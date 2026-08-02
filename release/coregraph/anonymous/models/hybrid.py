"""
models/hybrid.py
GNN-embedding → classical-classifier hybrid pipeline.

Flow
----
  1. Load (or train) a GNN model.
  2. Extract per-node embeddings via a forward hook on the final linear layer.
  3. Fit XGBoost / RandomForest on the train-split embeddings.
  4. Evaluate on the test-split embeddings.

This file is intentionally stateless: it exposes functional helpers rather
than a class so the caller controls the training lifecycle.
"""

import sys
import os
import json
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from typing import Tuple, Dict, Optional

# XGBoost segfaults when loaded in the same process as PyTorch on macOS
# (both link OpenMP independently).  Run XGBoost in a clean subprocess.
_XGB_SCRIPT = """
import sys, json, numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, accuracy_score)
path = sys.argv[1]
d    = np.load(path)
X_tr, y_tr, X_te, y_te = d['X_tr'], d['y_tr'], d['X_te'], d['y_te']
lm     = {1: 1, 2: 0}
y_tr_x = np.array([lm[int(v)] for v in y_tr], dtype=np.int32)
y_te_x = np.array([lm[int(v)] for v in y_te], dtype=np.int32)
sw     = (y_tr_x == 0).sum() / max((y_tr_x == 1).sum(), 1)
m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw,
    eval_metric='logloss', tree_method='hist', device='cpu',
    n_jobs=1, verbosity=0, random_state=42)
m.fit(X_tr, y_tr_x)
p    = m.predict(X_te)
prob = m.predict_proba(X_te)[:, 1]
try:    auc = float(roc_auc_score(y_te_x, prob))
except: auc = float('nan')
print(json.dumps(dict(
    f1=float(f1_score(y_te_x,p,zero_division=0)),
    precision=float(precision_score(y_te_x,p,zero_division=0)),
    recall=float(recall_score(y_te_x,p,zero_division=0)),
    auc=auc, accuracy=float(accuracy_score(y_te_x,p)),
)))
"""

def _run_xgb_subprocess(X_tr, y_tr, X_te, y_te, timeout=120) -> Dict:
    npz = tempfile.mktemp(suffix=".npz")
    try:
        np.savez(npz, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
        res = subprocess.run([sys.executable, "-c", _XGB_SCRIPT, npz],
                             capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-400:])
        return json.loads(res.stdout.strip())
    finally:
        if os.path.exists(npz):
            os.unlink(npz)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction (works with any GNN that ends with self.classifier)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model:  nn.Module,
    data:   Data,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract the penultimate (pre-classifier) representations via a forward hook.

    Works for GCN, GraphSAGE, GAT, EdgeGAT — any model whose last layer is
    `self.classifier` (a single nn.Linear).

    Returns
    -------
    Tensor of shape [num_nodes, hidden_dim]
    """
    captured = {}

    def _hook(module, inp, out):
        captured["emb"] = inp[0].detach().cpu()

    hook = model.classifier.register_forward_hook(_hook)
    model.eval()
    data = data.to(device)
    _ = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    hook.remove()

    return captured["emb"]   # [N, hidden]


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn-metric helper (binary: illicit=1 vs licit=2)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_sklearn(clf, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    preds = clf.predict(X)
    proba = clf.predict_proba(X)

    # Map 3-class labels (1=illicit,2=licit) to binary
    y_bin    = (y_true == 1).astype(int)
    pred_bin = (preds  == 1).astype(int)

    # Fraud class probability — column index for label 1
    classes = list(clf.classes_)
    if 1 in classes:
        prob_fraud = proba[:, classes.index(1)]
    else:
        prob_fraud = np.zeros(len(y_true))

    try:
        auc = roc_auc_score(y_bin, prob_fraud)
    except ValueError:
        auc = float("nan")

    return {
        "f1":        round(float(f1_score(y_bin, pred_bin, zero_division=0)), 4),
        "precision": round(float(precision_score(y_bin, pred_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_bin, pred_bin, zero_division=0)), 4),
        "auc":       round(float(auc), 4),
        "accuracy":  round(float((preds == y_true).mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval_hybrid(
    model:      nn.Module,
    data:       Data,
    device:     torch.device,
    clf_type:   str  = "xgboost",   # xgboost | rf
    concat_raw: bool = False,        # whether to append raw features to embeddings
) -> Dict[str, float]:
    """
    Extract GNN embeddings, train a classical classifier on train nodes,
    evaluate on test nodes.

    Returns metrics dict for the test set.
    """
    emb = extract_embeddings(model, data, device)   # [N, h]
    feat = emb.numpy()

    if concat_raw:
        raw = data.x.cpu().numpy()
        feat = np.concatenate([feat, raw], axis=1)

    y = data.y.numpy()

    X_train = feat[data.train_mask.numpy()]
    y_train = y[data.train_mask.numpy()]
    X_test  = feat[data.test_mask.numpy()]
    y_test  = y[data.test_mask.numpy()]

    if clf_type == "xgboost":
        # XGBoost must run in subprocess to avoid torch+OpenMP conflict on macOS
        metrics = _run_xgb_subprocess(X_train, y_train, X_test, y_test)
        return {k: round(float(v), 4) for k, v in metrics.items()}, None

    clf = _build_clf(clf_type)
    clf.fit(X_train, y_train)
    metrics = _eval_sklearn(clf, X_test, y_test)
    return metrics, clf


def _build_clf(clf_type: str):
    if clf_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"Unknown clf_type: {clf_type}")
