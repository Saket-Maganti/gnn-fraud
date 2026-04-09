"""
experiments/baselines.py
Classical ML baselines on raw node features (no graph structure).

Models
------
  Logistic Regression   — linear, interpretable
  Random Forest         — ensemble, handles non-linearity
  XGBoost               — boosted trees, strong baseline

Temporal split is respected: train on steps 1-34, test on 35-49.
Results are appended to results/metrics.csv.
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
)
from data.dataset import load_elliptic_raw, preprocess

# ── XGBoost subprocess (avoids torch+OpenMP segfault on macOS) ───────────────
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
    fi=m.feature_importances_.tolist(),
)))
"""

def _run_xgb_subprocess(X_tr, y_tr, X_te, y_te, timeout=120):
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval(clf, X_test, y_test, name):
    preds = clf.predict(X_test)
    classes = list(clf.classes_)
    proba   = clf.predict_proba(X_test)

    y_bin    = (y_test == 1).astype(int)
    pred_bin = (preds  == 1).astype(int)

    if 1 in classes:
        prob_fraud = proba[:, classes.index(1)]
    else:
        prob_fraud = np.zeros(len(y_test))

    try:
        auc = roc_auc_score(y_bin, prob_fraud)
    except ValueError:
        auc = float("nan")

    return {
        "model":     name,
        "graph_type": "none",
        "f1":        round(float(f1_score(y_bin, pred_bin, zero_division=0)), 4),
        "precision": round(float(precision_score(y_bin, pred_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_bin, pred_bin, zero_division=0)), 4),
        "auc":       round(float(auc), 4),
        "accuracy":  round(float(accuracy_score(y_test, preds)), 4),
        "notes":     "no_graph",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(results_dir: str = "results") -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CLASSICAL ML BASELINES")
    print("=" * 60)

    # Load data
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)

    y       = data.y.numpy()
    X       = data.x.numpy()
    ts      = data.time_step.numpy()

    # Labeled nodes only, temporal split
    labeled   = y != 0
    train_idx = np.where(labeled & (ts <= 34))[0]
    test_idx  = np.where(labeled & (ts  > 34))[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"  Train: {len(X_train):,} nodes  |  Test: {len(X_test):,} nodes")
    n_ill = (y_train == 1).sum()
    n_lic = (y_train == 2).sum()
    print(f"  Class ratio (train): {n_lic/max(n_ill,1):.1f}:1 licit:illicit")

    rows = []

    # ── Logistic Regression ────────────────────────────────────────────────
    print("\n[1/3] Logistic Regression …", end=" ", flush=True)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
    )
    lr.fit(X_train, y_train)
    row = _eval(lr, X_test, y_test, "logistic_regression")
    rows.append(row)
    print(f"F1={row['f1']:.4f}  AUC={row['auc']:.4f}")

    # ── Random Forest ──────────────────────────────────────────────────────
    print("[2/3] Random Forest …", end=" ", flush=True)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    row = _eval(rf, X_test, y_test, "random_forest")
    rows.append(row)
    print(f"F1={row['f1']:.4f}  AUC={row['auc']:.4f}")

    # ── XGBoost (subprocess — avoids torch+OpenMP segfault on macOS) ─────────
    print("[3/3] XGBoost (subprocess) …", end=" ", flush=True)
    try:
        import time as _t; t0 = _t.time()
        out = _run_xgb_subprocess(X_train, y_train, X_test, y_test)
        row = {
            "model": "xgboost", "graph_type": "none",
            "f1":       round(out["f1"],        4),
            "precision":round(out["precision"], 4),
            "recall":   round(out["recall"],    4),
            "auc":      round(out["auc"],       4),
            "accuracy": round(out["accuracy"],  4),
            "notes": "no_graph",
        }
        rows.append(row)
        print(f"F1={row['f1']:.4f}  AUC={row['auc']:.4f}  ({_t.time()-t0:.1f}s)")
        # XGBoost feature importance
        xgb_fi = pd.DataFrame({
            "feature_idx": np.arange(len(out["fi"])),
            "importance":  out["fi"],
        }).sort_values("importance", ascending=False)
        xgb_fi.to_csv(os.path.join(results_dir, "xgb_feature_importance.csv"), index=False)
    except Exception as e:
        print(f"ERROR: {e}")

    # ── Feature importance (RF) ────────────────────────────────────────────
    imp_path = os.path.join(results_dir, "rf_feature_importance.csv")
    fi = pd.DataFrame({
        "feature_idx": np.arange(X_train.shape[1]),
        "importance":  rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(imp_path, index=False)
    print(f"\n  RF feature importances → {imp_path}")

    # ── Save ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Remove rows for these models so we don't duplicate
        existing = existing[~existing["model"].isin(df["model"])]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_baselines()
