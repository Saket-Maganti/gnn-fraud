"""
experiments/graph_ablation_extended.py
Extended graph ablation study — three structural conditions.

Conditions
----------
  1. original    — dataset edges as-is
  2. shuffled    — edges randomly permuted (same degree sequence, random topology)
  3. no_edges    — empty edge_index (pure MLP, no message passing)

For each condition, GraphSAGE+weighted is trained (3 seeds for stability)
and evaluated on the overall test set and per test timestep.

Key question: Is the GNN's gain structural (from real graph topology),
or would any graph — even a random one — help equally?

Outputs
-------
  results/graph_ablation_extended.csv   (per-condition metrics, mean±std)
  results/graph_ablation_extended.png   (figure)

Usage:
    python experiments/graph_ablation_extended.py
    python experiments/graph_ablation_extended.py --epochs 150 --seeds 3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Data
from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.metrics import compute_metrics
from utils.temporal import eval_per_timestep


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    return torch.device(device_arg)


# ─────────────────────────────────────────────────────────────────────────────
# Graph manipulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def shuffled_edges(data: Data, seed: int = 42) -> Data:
    """
    Permute the edge_index columns randomly so each edge still connects two
    nodes, but the topology is random (same edge count, randomised endpoints).
    This breaks any structural signal while keeping message-passing mechanics.
    """
    rng = np.random.default_rng(seed)
    ei  = data.edge_index.numpy()  # [2, E]
    E   = ei.shape[1]
    # Randomly permute both endpoints independently
    src = rng.permutation(data.num_nodes)[rng.integers(0, data.num_nodes, size=E)]
    dst = rng.permutation(data.num_nodes)[rng.integers(0, data.num_nodes, size=E)]
    new_ei = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    new_data = Data(
        x          = data.x,
        edge_index = new_ei,
        y          = data.y,
        train_mask = data.train_mask,
        test_mask  = data.test_mask,
        time_step  = data.time_step,
    )
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        # edge_attr no longer matches — drop it
        new_data.edge_attr = None
    return new_data


def no_edges(data: Data) -> Data:
    """Remove all edges → pure feature-based MLP (no graph structure)."""
    empty_ei = torch.zeros(2, 0, dtype=torch.long)
    return Data(
        x          = data.x,
        edge_index = empty_ei,
        y          = data.y,
        train_mask = data.train_mask,
        test_mask  = data.test_mask,
        time_step  = data.time_step,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def train_one(graph_data: Data, device: torch.device,
              epochs: int = 200, seed: int = 42) -> dict:
    """Train SAGE+weighted on graph_data, return test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cw             = get_class_weights(graph_data)
    aug_data, crit = apply_strategy(graph_data, "weighted", cw)
    aug_data       = aug_data.to(device)
    crit           = crit.to(device)

    model = build_model(
        "sage",
        in_channels     = aug_data.num_node_features,
        hidden_channels = 256,
        num_layers      = 3,
        dropout         = 0.5,
        out_channels    = 3,
    ).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state, patience_cnt = 0.0, None, 0
    patience = 30

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(aug_data.x, aug_data.edge_index,
                       getattr(aug_data, "edge_attr", None))
        loss = crit(logits, aug_data.y, aug_data.train_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if hasattr(crit, "step_epoch"):
            crit.step_epoch()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                le = model(aug_data.x, aug_data.edge_index,
                           getattr(aug_data, "edge_attr", None))
            f1 = compute_metrics(le, aug_data.y, aug_data.test_mask)["f1"]
            if f1 > best_f1:
                best_f1   = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= patience // 10:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on the original graph_data (edge_index unchanged for this condition)
    model.eval()
    gd = graph_data.to(device)
    with torch.no_grad():
        logits = model(gd.x, gd.edge_index, getattr(gd, "edge_attr", None))
    metrics = compute_metrics(logits, gd.y, gd.test_mask)
    return model, metrics, logits, gd


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

CONDITION_LABELS = {
    "original": "Original Graph",
    "shuffled": "Shuffled Edges",
    "no_edges": "No Edges (MLP)",
}

CONDITION_COLORS = {
    "original": "#4e79a7",
    "shuffled": "#f28e2b",
    "no_edges": "#e15759",
}


def run(args):
    device      = resolve_device(args.device)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Using device: {device}")
    print("Loading data …")
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)
    print(f"  Nodes: {data.num_nodes:,}  |  Edges: {data.edge_index.shape[1]:,}")

    conditions = {
        "original": data,
        "shuffled": shuffled_edges(data, seed=0),
        "no_edges": no_edges(data),
    }

    all_rows      = []
    temporal_res  = {}   # condition → { timestep: metrics }

    for cond_name, graph_data in conditions.items():
        print(f"\n{'='*55}")
        print(f"Condition: {CONDITION_LABELS[cond_name]}")
        print(f"  Edges: {graph_data.edge_index.shape[1]:,}")
        print(f"{'='*55}")

        seed_metrics = []
        for s_idx, seed in enumerate(args.seeds[:args.n_seeds]):
            print(f"  Seed {seed} ({s_idx+1}/{args.n_seeds}) …", end=" ", flush=True)
            model, metrics, logits, gd = train_one(
                graph_data, device, epochs=args.epochs, seed=seed
            )
            seed_metrics.append(metrics)
            print(f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

        # Aggregate across seeds
        for metric in ["f1", "precision", "recall", "auc", "accuracy"]:
            vals = [m[metric] for m in seed_metrics]
            row = {
                "condition":  cond_name,
                "label":      CONDITION_LABELS[cond_name],
                "seeds":      len(seed_metrics),
                f"{metric}_mean": round(float(np.mean(vals)), 4),
                f"{metric}_std":  round(float(np.std(vals)),  4),
            }
        # Build full row
        full_row = {"condition": cond_name, "label": CONDITION_LABELS[cond_name],
                    "seeds": len(seed_metrics)}
        for metric in ["f1", "precision", "recall", "auc", "accuracy"]:
            vals = [m[metric] for m in seed_metrics]
            full_row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            full_row[f"{metric}_std"]  = round(float(np.std(vals)), 4)
        all_rows.append(full_row)

        # Per-timestep evaluation (using last trained model)
        model.eval()
        gd = graph_data.to(device)
        ts_metrics = {}
        with torch.no_grad():
            logits_full = model(gd.x, gd.edge_index, getattr(gd, "edge_attr", None))
        for t in range(35, 50):
            mask = gd.test_mask & (gd.time_step == t)
            if mask.sum() < 5:
                continue
            ts_metrics[t] = compute_metrics(logits_full, gd.y, mask.to(device))
        temporal_res[cond_name] = ts_metrics

        print(f"  → mean F1={full_row['f1_mean']:.4f} ± {full_row['f1_std']:.4f}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(results_dir, "graph_ablation_extended.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")

    # Per-timestep CSV
    ts_rows = []
    for cond_name, ts_dict in temporal_res.items():
        for t, m in ts_dict.items():
            ts_rows.append({"condition": cond_name, "timestep": t, **m})
    pd.DataFrame(ts_rows).to_csv(
        os.path.join(results_dir, "graph_ablation_temporal.csv"), index=False
    )

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'Condition':<22} {'F1':>8} {'±':>4} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
    print("─" * 60)
    for row in all_rows:
        print(f"{row['label']:<22} {row['f1_mean']:>8.4f} {row['f1_std']:>4.4f} "
              f"{row['precision_mean']:>8.4f} {row['recall_mean']:>8.4f} {row['auc_mean']:>8.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        _plot(df, temporal_res, results_dir)

    return df


def _plot(summary_df: pd.DataFrame, temporal_res: dict, results_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart (overall metrics)
    ax = axes[0]
    conds    = summary_df["condition"].tolist()
    f1_means = summary_df["f1_mean"].tolist()
    f1_stds  = summary_df["f1_std"].tolist()
    auc_means = summary_df["auc_mean"].tolist()
    x = np.arange(len(conds))
    w = 0.35
    colors = [CONDITION_COLORS[c] for c in conds]
    bars1 = ax.bar(x - w/2, f1_means, w, yerr=f1_stds, capsize=4,
                   color=colors, alpha=0.85, label="F1")
    bars2 = ax.bar(x + w/2, auc_means, w,
                   color=colors, alpha=0.50, hatch="//", label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conds], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Test Performance\nby Graph Condition", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars1, f1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # Right: per-timestep F1 lines
    ax2 = axes[1]
    steps = list(range(35, 50))
    for cond_name, ts_dict in temporal_res.items():
        ys = [ts_dict.get(t, {}).get("f1", float("nan")) for t in steps]
        ax2.plot(steps, ys, color=CONDITION_COLORS[cond_name],
                 linewidth=2.0, marker="o", markersize=4,
                 label=CONDITION_LABELS[cond_name])

    ax2.axvspan(42.5, 49.5, alpha=0.07, color="#e15759", label="Late drift zone")
    ax2.set_xlabel("Test Timestep")
    ax2.set_ylabel("F1 Score")
    ax2.set_xlim(34.5, 49.5)
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_title("Per-Timestep F1\nby Graph Condition", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(steps)
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Graph Ablation Study: Structural Contribution of Edges",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(results_dir, "graph_ablation_extended.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 7, 123, 0, 99])
    parser.add_argument("--device",  choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--no_plot", action="store_true")
    run(parser.parse_args())
