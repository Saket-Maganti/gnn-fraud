"""
experiments/run_insights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Master insight-generation pipeline.  Runs all 7 parts in sequence and
produces three CSV files that drive the paper's experimental section.

Outputs
-------
  results/metrics.csv          – GNN (per graph type) + classical + hybrid
  results/ablation_results.csv – remove-edges / remove-features ablations
  results/graph_stats.csv      – topology statistics per graph variant

Usage
-----
  # Full pipeline (~60-90 min on CPU)
  python experiments/run_insights.py

  # Individual parts
  python experiments/run_insights.py --part graph_stats
  python experiments/run_insights.py --part baselines
  python experiments/run_insights.py --part gnn_variants
  python experiments/run_insights.py --part hybrid
  python experiments/run_insights.py --part ablation
  python experiments/run_insights.py --part explain

  # Fast mode (fewer epochs, quick sanity check, ~15-20 min)
  python experiments/run_insights.py --fast
"""

import os
import sys
import argparse
import time
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from data.graph_builder import build_graph, graph_summary
from models.gnn import build_model, count_parameters
from models.hybrid import extract_embeddings, train_and_eval_hybrid
from utils.imbalance import WeightedCELoss, apply_strategy
from utils.metrics import compute_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import networkx as nx

# ── XGBoost subprocess helper ─────────────────────────────────────────────────
# PyTorch and XGBoost both use OpenMP; on macOS they segfault when run in the
# same process.  We run XGBoost in a clean subprocess with data passed via npz.
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
    f1        = float(f1_score(y_te_x, p, zero_division=0)),
    precision = float(precision_score(y_te_x, p, zero_division=0)),
    recall    = float(recall_score(y_te_x, p, zero_division=0)),
    auc       = auc,
    accuracy  = float(accuracy_score(y_te_x, p)),
    fi        = m.feature_importances_.tolist(),
)))
"""

def _run_xgb_subprocess(X_tr, y_tr, X_te, y_te, timeout=120):
    """Run XGBoost in a subprocess to avoid torch+OpenMP conflict on macOS."""
    npz = tempfile.mktemp(suffix=".npz")
    try:
        np.savez(npz, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)
        res = subprocess.run(
            [sys.executable, "-c", _XGB_SCRIPT, npz],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-400:])
        return json.loads(res.stdout.strip())
    finally:
        if os.path.exists(npz):
            os.unlink(npz)

RESULTS_DIR = "results"
GRAPH_VARIANTS = ["original", "similarity", "feature_knn", "temporal", "augmented"]


# ─────────────────────────────────────────────────────────────────────────────
# Compact GNN training loop (reusable)
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(
    data,
    model_name: str   = "sage",
    epochs:     int   = 300,
    patience:   int   = 30,
    device:     torch.device = torch.device("cpu"),
    seed:       int   = 42,
) -> tuple:
    """
    Train GraphSAGE (or any model) with WeightedCELoss on the given data object.
    Returns (trained_model, test_metrics_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cw    = get_class_weights(data)
    aug_data, criterion = apply_strategy(data, "weighted", cw)

    model = build_model(
        model_name,
        in_channels    = aug_data.num_node_features,
        hidden_channels= 256,
        num_layers     = 3,
        dropout        = 0.5,
        out_channels   = 3,
    )
    model    = model.to(device)
    aug_data = aug_data.to(device)
    criterion= criterion.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state, no_improve = 0.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(aug_data.x, aug_data.edge_index,
                       getattr(aug_data, "edge_attr", None))
        loss   = criterion(logits, aug_data.y, aug_data.train_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if hasattr(criterion, "step_epoch"):
            criterion.step_epoch()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_eval = model(aug_data.x, aug_data.edge_index,
                                    getattr(aug_data, "edge_attr", None))
            metrics = compute_metrics(logits_eval, aug_data.y, aug_data.test_mask)
            f1 = metrics["f1"]
            if f1 > best_f1:
                best_f1   = f1
                best_state= {k: v.clone() for k, v in model.state_dict().items()}
                no_improve= 0
            else:
                no_improve += 1
            if no_improve >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    # Evaluate on ORIGINAL data (not aug_data which may have synthetic nodes)
    orig_dev = data.to(device)
    with torch.no_grad():
        logits_test = model(orig_dev.x, orig_dev.edge_index,
                            getattr(orig_dev, "edge_attr", None))
    final_metrics = compute_metrics(logits_test, orig_dev.y, orig_dev.test_mask)
    return model, final_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Part 5 – Graph statistics
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_stats(data, results_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PART 5 – GRAPH TOPOLOGY STATISTICS")
    print("=" * 60)

    rows = []
    for gtype in GRAPH_VARIANTS:
        print(f"  [{gtype}] …", end=" ", flush=True)
        gdata = build_graph(data, gtype)
        stats = graph_summary(gdata, name=gtype)
        # Sampled clustering (inline — avoids cross-module private import)
        try:
            rng = np.random.default_rng(42)
            n_s = min(3000, gdata.num_nodes)
            sampled = rng.choice(gdata.num_nodes, size=n_s, replace=False)
            src_np = gdata.edge_index[0].numpy()
            dst_np = gdata.edge_index[1].numpy()
            mask_c = np.isin(src_np, sampled) & np.isin(dst_np, sampled)
            G_nx = nx.Graph()
            G_nx.add_nodes_from(sampled)
            G_nx.add_edges_from(zip(src_np[mask_c].tolist(), dst_np[mask_c].tolist()))
            clust = round(float(nx.average_clustering(G_nx)), 6)
        except Exception:
            clust = float("nan")
        stats["avg_clustering_sampled"] = clust
        # Connected components (skip if too many edges)
        E = gdata.edge_index.shape[1]
        if E <= 800_000:
            src = gdata.edge_index[0].numpy()
            dst = gdata.edge_index[1].numpy()
            G   = nx.Graph()
            G.add_nodes_from(range(gdata.num_nodes))
            G.add_edges_from(zip(src.tolist(), dst.tolist()))
            stats["connected_components"] = nx.number_connected_components(G)
        else:
            stats["connected_components"] = -1
        print(f"E={stats['num_edges']:,}  avg_deg={stats['avg_degree']:.2f}  "
              f"clust={clust:.4f}")
        rows.append(stats)

    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, "graph_stats.csv")
    df.to_csv(out, index=False)
    print(f"\n  → {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 – Classical baselines
# ─────────────────────────────────────────────────────────────────────────────

def _sklearn_metrics(clf, X_test, y_test, name, graph_type="none") -> dict:
    preds  = clf.predict(X_test)
    proba  = clf.predict_proba(X_test)
    classes= list(clf.classes_)
    y_bin  = (y_test == 1).astype(int)
    pb     = (preds  == 1).astype(int)
    prob_f = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(y_test))
    try:
        auc = roc_auc_score(y_bin, prob_f)
    except ValueError:
        auc = float("nan")
    return {
        "model": name, "graph_type": graph_type,
        "f1":       round(float(f1_score(y_bin, pb, zero_division=0)), 4),
        "precision":round(float(precision_score(y_bin, pb, zero_division=0)), 4),
        "recall":   round(float(recall_score(y_bin, pb, zero_division=0)), 4),
        "auc":      round(float(auc), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "notes":    "no_graph",
    }


def run_baselines(data, results_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PART 3 – CLASSICAL BASELINES")
    print("=" * 60)

    X   = data.x.numpy()
    y   = data.y.numpy()
    ts  = data.time_step.numpy()

    labeled   = y != 0
    tr_idx    = np.where(labeled & (ts <= 34))[0]
    te_idx    = np.where(labeled & (ts  > 34))[0]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"  Train {len(X_tr):,} / Test {len(X_te):,}")

    rows = []

    # Logistic Regression
    print("  [1/3] Logistic Regression …", end=" ", flush=True)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            solver="lbfgs", random_state=42)
    lr.fit(X_tr, y_tr)
    row = _sklearn_metrics(lr, X_te, y_te, "logistic_regression")
    rows.append(row)
    print(f"F1={row['f1']:.4f}")

    # Random Forest
    print("  [2/3] Random Forest …", end=" ", flush=True)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=1)
    rf.fit(X_tr, y_tr)
    row = _sklearn_metrics(rf, X_te, y_te, "random_forest")
    rows.append(row)
    print(f"F1={row['f1']:.4f}")

    # Save RF feature importance
    fi_df = pd.DataFrame({
        "feature_idx": np.arange(X_tr.shape[1]),
        "importance":  rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(results_dir, "rf_feature_importance.csv"), index=False)

    # XGBoost — run in subprocess to avoid torch+OpenMP segfault on macOS
    print("  [3/3] XGBoost (subprocess) …", end=" ", flush=True)
    try:
        t0 = time.time()
        out = _run_xgb_subprocess(X_tr, y_tr, X_te, y_te)
        xgb_row = {
            "model": "xgboost", "graph_type": "none",
            "f1":       round(out["f1"],        4),
            "precision":round(out["precision"], 4),
            "recall":   round(out["recall"],    4),
            "auc":      round(out["auc"],       4),
            "accuracy": round(out["accuracy"],  4),
            "notes":    "no_graph",
        }
        rows.append(xgb_row)
        print(f"F1={xgb_row['f1']:.4f}  ({time.time()-t0:.1f}s)")
        # Feature importance
        xgb_fi = pd.DataFrame({
            "feature_idx": np.arange(len(out["fi"])),
            "importance":  out["fi"],
        }).sort_values("importance", ascending=False)
        xgb_fi.to_csv(os.path.join(results_dir, "xgb_feature_importance.csv"), index=False)
    except Exception as e:
        print(f"ERROR: {e}")
    return rows, rf, None


# ─────────────────────────────────────────────────────────────────────────────
# Part 1+2 – GNN on graph variants + hybrid
# ─────────────────────────────────────────────────────────────────────────────

def run_gnn_variants(data, epochs: int, device, results_dir: str) -> list:
    print("\n" + "=" * 60)
    print("PARTS 1+2 – GNN ON GRAPH VARIANTS + HYBRID")
    print("=" * 60)
    rows = []
    trained_models = {}

    for gtype in GRAPH_VARIANTS:
        print(f"\n  ── {gtype.upper()} ──")
        t0    = time.time()
        gdata = build_graph(data, gtype)
        model, metrics = train_gnn(gdata, model_name="sage",
                                   epochs=epochs, device=device)
        elapsed = time.time() - t0
        print(f"  GNN-SAGE  F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"AUC={metrics['auc']:.4f}  ({elapsed/60:.1f} min)")
        rows.append({
            "model": "gnn_sage", "graph_type": gtype,
            **metrics, "notes": "gnn_only",
        })
        if gtype == "original":
            trained_models["gnn"] = (model, gdata)

        # Save checkpoint
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"sage_{gtype}.pt"))

    # Hybrid: GNN embeddings → XGBoost + RF (on original graph only)
    print("\n  ── HYBRID (original graph) ──")
    gnn_model, gnn_data = trained_models["gnn"]
    for clf_type in ["xgboost", "rf"]:
        try:
            metrics_h, clf_h = train_and_eval_hybrid(
                gnn_model, gnn_data.cpu(), torch.device("cpu"), clf_type=clf_type
            )
            print(f"  GNN+{clf_type.upper():<8}  F1={metrics_h['f1']:.4f}  "
                  f"P={metrics_h['precision']:.4f}  R={metrics_h['recall']:.4f}  "
                  f"AUC={metrics_h['auc']:.4f}")
            rows.append({
                "model": f"gnn+{clf_type}", "graph_type": "original",
                **metrics_h, "notes": "hybrid",
            })
        except Exception as e:
            print(f"  GNN+{clf_type} FAILED: {e}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 – Ablation
# ─────────────────────────────────────────────────────────────────────────────

def _struct_features(data) -> torch.Tensor:
    """
    Create 5 structural (non-content) features per node:
      [degree_norm, log_degree, timestep_norm, in_deg_norm, out_deg_norm]
    Derived from the original graph and timestep only.
    """
    N  = data.num_nodes
    ei = data.edge_index

    # Undirected degree (use original edge_index)
    deg = torch.zeros(N)
    deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1]))
    deg_max = deg.max().clamp(min=1)
    deg_norm= deg / deg_max
    log_deg = (deg + 1).log()
    log_max = log_deg.max().clamp(min=1)
    log_deg = log_deg / log_max

    # Timestep normalised 0-1
    ts     = data.time_step.float()
    ts_min, ts_max = ts.min(), ts.max()
    ts_norm= (ts - ts_min) / (ts_max - ts_min + 1e-8)

    # In/out degree on directed graph (before to_undirected, use raw ei)
    in_deg = torch.zeros(N)
    in_deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1]))
    in_norm = in_deg / in_deg.max().clamp(min=1)

    out_deg = torch.zeros(N)
    out_deg.scatter_add_(0, ei[0], torch.ones(ei.shape[1]))
    out_norm = out_deg / out_deg.max().clamp(min=1)

    return torch.stack([deg_norm, log_deg, ts_norm, in_norm, out_norm], dim=1)


def run_ablation(data, epochs: int, device, results_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PART 4 – ABLATION EXPERIMENTS")
    print("=" * 60)

    rows = []

    # ── (a) Full model (reference) ─────────────────────────────────────────
    print("\n  [a] Full GNN (graph + features) …")
    _, m = train_gnn(data, model_name="sage", epochs=epochs, device=device)
    rows.append({"experiment": "full_model", "removed": "nothing", **m})
    print(f"  F1={m['f1']:.4f}")

    # ── (b) No edges (empty graph — pure feature MLP) ──────────────────────
    print("\n  [b] No edges (graph structure removed) …")
    no_edge_data = build_graph(data, "no_edges")
    _, m = train_gnn(no_edge_data, model_name="sage", epochs=epochs, device=device)
    rows.append({"experiment": "no_edges", "removed": "graph_structure", **m})
    print(f"  F1={m['f1']:.4f}")

    # ── (c) No node features (structural features only) ────────────────────
    print("\n  [c] No node features (structural features only) …")
    from copy import deepcopy
    struct_data       = deepcopy(data)
    struct_data.x     = _struct_features(data)
    struct_data.edge_attr = None
    _, m = train_gnn(struct_data, model_name="sage", epochs=epochs, device=device)
    rows.append({"experiment": "struct_only", "removed": "node_content_features", **m})
    print(f"  F1={m['f1']:.4f}")

    # ── (d) No edges AND no node features ─────────────────────────────────
    print("\n  [d] No edges + structural features only (minimal model) …")
    bare_data         = deepcopy(data)
    bare_data.x       = _struct_features(data)
    bare_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    bare_data.edge_attr  = None
    _, m = train_gnn(bare_data, model_name="sage", epochs=epochs, device=device)
    rows.append({"experiment": "no_edges_struct_only",
                 "removed": "graph_structure+node_content", **m})
    print(f"  F1={m['f1']:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(results_dir, "ablation_results.csv")
    df.to_csv(out, index=False)
    print(f"\n  → {out}")
    print(df[["experiment", "f1", "precision", "recall", "auc"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 6 – Explainability
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability(data, gnn_model, device, results_dir: str):
    """
    GNN gradient-based saliency (which input features drive fraud predictions).
    Saves: results/gnn_feature_saliency.csv  +  results/feature_importance_comparison.png
    """
    print("\n" + "=" * 60)
    print("PART 6 – EXPLAINABILITY")
    print("=" * 60)

    gnn_model.eval()
    data_dev = data.to(device)

    # Gradient of illicit-class logit w.r.t. input features
    x_grad  = data_dev.x.clone().detach().requires_grad_(True)
    logits  = gnn_model(x_grad, data_dev.edge_index,
                        getattr(data_dev, "edge_attr", None))

    # Focus on test nodes predicted as illicit
    fraud_mask = data_dev.test_mask & (data_dev.y == 1)
    illicit_score = logits[fraud_mask, 1].sum()
    illicit_score.backward()

    saliency = x_grad.grad[fraud_mask].abs().mean(dim=0).cpu().numpy()

    sal_df = pd.DataFrame({
        "feature_idx": np.arange(len(saliency)),
        "gnn_saliency": saliency,
    }).sort_values("gnn_saliency", ascending=False)
    sal_path = os.path.join(results_dir, "gnn_feature_saliency.csv")
    sal_df.to_csv(sal_path, index=False)
    print(f"  GNN saliency → {sal_path}")
    print(f"  Top-5 features by GNN gradient: "
          f"{sal_df['feature_idx'].head(5).tolist()}")

    # Load RF importance for comparison plot
    rf_path = os.path.join(results_dir, "rf_feature_importance.csv")
    if os.path.exists(rf_path):
        rf_fi = pd.read_csv(rf_path).sort_values("feature_idx").head(166)
        top20 = sal_df.head(20)["feature_idx"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].barh(range(20), sal_df.head(20)["gnn_saliency"].values[::-1])
        axes[0].set_yticks(range(20))
        axes[0].set_yticklabels([f"f{i}" for i in sal_df.head(20)["feature_idx"].values[::-1]])
        axes[0].set_title("GNN Gradient Saliency (Top 20)")
        axes[0].set_xlabel("Mean |gradient|")

        rf_top20 = rf_fi[rf_fi["feature_idx"].isin(top20)].sort_values("importance", ascending=True)
        axes[1].barh(range(len(rf_top20)), rf_top20["importance"].values)
        axes[1].set_yticks(range(len(rf_top20)))
        axes[1].set_yticklabels([f"f{i}" for i in rf_top20["feature_idx"].values])
        axes[1].set_title("RF Feature Importance (same features)")
        axes[1].set_xlabel("Gini importance")

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "feature_importance_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Comparison plot → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 7 – Consolidate metrics.csv
# ─────────────────────────────────────────────────────────────────────────────

def consolidate_metrics(baseline_rows, gnn_rows, results_dir: str) -> pd.DataFrame:
    all_rows = baseline_rows + gnn_rows
    df = pd.DataFrame(all_rows)
    cols = ["model", "graph_type", "f1", "precision", "recall", "auc", "accuracy", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df = df.sort_values(["graph_type", "f1"], ascending=[True, False])
    out = os.path.join(results_dir, "metrics.csv")
    df.to_csv(out, index=False)
    print(f"\n  Full metrics → {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics_df: pd.DataFrame, ablation_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\n── Metrics (sorted by F1) ──")
    show_cols = ["model", "graph_type", "f1", "precision", "recall", "auc"]
    print(metrics_df[show_cols].sort_values("f1", ascending=False).to_string(index=False))
    print("\n── Ablation ──")
    abl_cols = ["experiment", "f1", "precision", "recall", "auc"]
    print(ablation_df[abl_cols].to_string(index=False))
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all",
                        choices=["all", "graph_stats", "baselines",
                                 "gnn_variants", "hybrid", "ablation", "explain"])
    parser.add_argument("--fast", action="store_true",
                        help="Reduce epochs for quick validation (~15-20 min total)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    EPOCHS  = 100 if args.fast else 300
    PATIENCE= 15  if args.fast else 30
    device  = torch.device(args.device)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("GNN FRAUD DETECTION — INSIGHT PIPELINE")
    print(f"  mode   : {'FAST' if args.fast else 'FULL'}")
    print(f"  epochs : {EPOCHS}")
    print(f"  device : {device}")
    print(f"{'='*60}")

    t_total = time.time()

    # Load data once
    print("\nLoading Elliptic dataset …")
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)
    print(f"  Nodes={data.num_nodes:,}  Edges={data.edge_index.shape[1]:,}  "
          f"Features={data.num_node_features}")

    part = args.part
    gnn_rows = []
    baseline_rows_raw = []

    # ── Graph stats ─────────────────────────────────────────────────────────
    if part in ("all", "graph_stats"):
        run_graph_stats(data, RESULTS_DIR)

    # ── Baselines ───────────────────────────────────────────────────────────
    if part in ("all", "baselines"):
        result = run_baselines(data, RESULTS_DIR)
        if isinstance(result, tuple):
            baseline_rows_raw, rf_model, xgb_model = result
        else:
            baseline_rows_raw = result

    # ── GNN variants + hybrid ───────────────────────────────────────────────
    if part in ("all", "gnn_variants", "hybrid"):
        gnn_rows = run_gnn_variants(data, EPOCHS, device, RESULTS_DIR)

    # ── Ablation ────────────────────────────────────────────────────────────
    if part in ("all", "ablation"):
        run_ablation(data, EPOCHS, device, RESULTS_DIR)

    # ── Explainability ──────────────────────────────────────────────────────
    if part in ("all", "explain"):
        # Load or train a GNN for explainability
        ckpt = os.path.join(RESULTS_DIR, "checkpoints", "sage_original.pt")
        explain_model = build_model("sage", in_channels=data.num_node_features,
                                    hidden_channels=256, num_layers=3,
                                    dropout=0.5, out_channels=3)
        if os.path.exists(ckpt):
            explain_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"\n  Loaded checkpoint: {ckpt}")
        else:
            print("\n  No checkpoint found — training GNN for explainability …")
            explain_model, _ = train_gnn(data, model_name="sage",
                                          epochs=EPOCHS, device=device)
        run_explainability(data, explain_model, device, RESULTS_DIR)

    # ── Consolidate metrics.csv ──────────────────────────────────────────────
    if part == "all" and baseline_rows_raw and gnn_rows:
        df = consolidate_metrics(baseline_rows_raw, gnn_rows, RESULTS_DIR)
        abl = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
        print_summary(df, abl)

    elapsed = time.time() - t_total
    print(f"\n  Total wall time: {elapsed/60:.1f} min")
    print(f"  Outputs in:     {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
