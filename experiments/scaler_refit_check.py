"""
experiments/scaler_refit_check.py

FIX 8a: refit StandardScaler on train-only rows (time_step <= 34) and
re-run the feature-only Random Forest across 10 seeds to bound the
feature-normalisation leakage empirically.

Baseline for comparison: paper's current 0.821 ± 0.003 F1 with scaler
fit on the full labelled population.

Outputs
-------
  results/fix8a_scaler_refit/
    per_seed.csv       one row per seed with F1 / precision / recall / AUC
    summary.json       mean ± std, delta vs. paper's 0.821 baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.dataset import load_elliptic_raw


def build_features(train_only_scaler: bool) -> dict:
    """Return dict with x, y_bin, train_mask, test_mask."""
    features, classes, _edges = load_elliptic_raw()

    time_step = features.iloc[:, 1].values.astype(int)
    feat_mat  = features.iloc[:, 2:].values.astype(np.float32)

    # Labels
    node_ids  = features.iloc[:, 0].values
    id2label  = dict(zip(classes["txId"], classes["class"].astype(str)))
    label_map = {"unknown": 0, "1": 1, "2": 2}
    y = np.array(
        [label_map.get(id2label.get(txid, "unknown"), 0) for txid in node_ids],
        dtype=np.int64,
    )

    labeled    = y != 0
    train_mask = labeled & (time_step <= 34)
    test_mask  = labeled & (time_step > 34)

    if train_only_scaler:
        scaler = StandardScaler().fit(feat_mat[train_mask])
    else:
        scaler = StandardScaler().fit(feat_mat)
    feat_mat = scaler.transform(feat_mat)

    # Binary: illicit (class 1) vs licit (class 2). Drop unknowns for RF.
    y_bin = (y == 1).astype(int)

    return {
        "x":          feat_mat,
        "y_bin":      y_bin,
        "train_mask": train_mask,
        "test_mask":  test_mask,
    }


def run_rf(d: dict, seed: int) -> dict:
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    X_tr = d["x"][d["train_mask"]]
    y_tr = d["y_bin"][d["train_mask"]]
    X_te = d["x"][d["test_mask"]]
    y_te = d["y_bin"][d["test_mask"]]

    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)
    classes = list(rf.classes_)
    prob_fraud = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(y_te))
    preds = (prob_fraud >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_te, prob_fraud))
    except ValueError:
        auc = float("nan")

    return {
        "seed":      seed,
        "f1":        float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall":    float(recall_score(y_te, preds, zero_division=0)),
        "auc":       auc,
    }


def main(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # We run both conditions so we can report the delta directly.
    print("[fix8a] loading data + building features ...")
    cond_train_only = build_features(train_only_scaler=True)
    cond_full       = build_features(train_only_scaler=False)

    rows_to, rows_full = [], []
    t0 = time.time()
    for seed in range(args.seeds):
        r_to = run_rf(cond_train_only, seed)
        rows_to.append({"scaler": "train_only", **r_to})
        r_fu = run_rf(cond_full, seed)
        rows_full.append({"scaler": "full_population", **r_fu})
        elapsed = time.time() - t0
        print(
            f"  seed={seed}  train_only F1={r_to['f1']:.4f}  "
            f"full F1={r_fu['f1']:.4f}  "
            f"Δ={r_to['f1']-r_fu['f1']:+.4f}  "
            f"({elapsed/60:.1f} min)"
        )

    rows = rows_to + rows_full
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_seed.csv"), index=False)

    def agg(sub: pd.DataFrame) -> dict:
        return {
            "f1_mean":   float(sub["f1"].mean()),
            "f1_std":    float(sub["f1"].std(ddof=1)),
            "prec_mean": float(sub["precision"].mean()),
            "rec_mean":  float(sub["recall"].mean()),
            "auc_mean":  float(sub["auc"].mean()),
            "seeds":     len(sub),
        }

    summary = {
        "train_only":      agg(df[df["scaler"] == "train_only"]),
        "full_population": agg(df[df["scaler"] == "full_population"]),
    }
    summary["delta_f1_train_only_minus_full"] = (
        summary["train_only"]["f1_mean"]
        - summary["full_population"]["f1_mean"]
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FIX 8a summary ===")
    for name, s in summary.items():
        if isinstance(s, dict):
            print(f"  {name}: F1={s['f1_mean']:.4f} ± {s['f1_std']:.4f}  (n={s['seeds']})")
    print(f"  delta (train_only - full): {summary['delta_f1_train_only_minus_full']:+.4f}")
    print(f"\n[io] wrote {os.path.join(out_dir, 'per_seed.csv')}")
    print(f"[io] wrote {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   type=int, default=10)
    p.add_argument("--out_dir", type=str,
                   default="results/fix8a_scaler_refit")
    main(p.parse_args())
