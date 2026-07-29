"""
experiments/run_shuffle_ablation_multi.py

Cross-dataset shuffled-edges ablation.

The paper's flagship observation ("randomly shuffled edges beat the real
transaction graph") was only demonstrated on Elliptic in the original
submission. Reviewer #1's core ask was: does this survive on *other*
datasets and with *modern* models? This script answers that.

For each (dataset, model, seed) triple we train three variants:

    real      : the dataset's true edge_index
    shuffled  : a per-endpoint random permutation that preserves the
                degree distribution (Fisher–Yates on dst indices, src
                unchanged) — destroys all real topology but keeps
                message-passing volume identical
    none      : edge_index made empty so the GNN reduces to a per-node
                MLP applied independently

We compare **inductive** F1 across variants (the transductive numbers are
uninteresting under shuffling because the test-set edges are scrambled
out from under the evaluation graph as well). Anywhere ``shuffled > real``
is direct evidence that the real graph provides no useful temporal
signal for the classifier under inductive evaluation.

Output schema: ``results/shuffle_multi/{dataset}__{model}__seed{s}.json``
with keys ``real | shuffled | none`` each mapping to the usual metrics
dict.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.datasets import load_dataset                                   # noqa: E402
from data.datasets.base import describe_dataset                         # noqa: E402
from experiments._multi_harness import (                                # noqa: E402
    RunConfig, build_inductive_view, _forward, _train, set_seed,
)
from experiments.run_metadata import build_run_metadata                  # noqa: E402
from experiments.result_audit import (                                  # noqa: E402
    audit_sweep,
    summarize_rows,
    write_audit_reports,
    write_json_atomic,
)
from models.registry import build_model, count_parameters                # noqa: E402
from utils.metrics import compute_metrics                               # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Edge manipulations
# ─────────────────────────────────────────────────────────────────────────────

def shuffle_edges(edge_index: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute destination indices while leaving sources in place.

    Preserves the degree sequence of the *source* endpoints exactly and the
    degree sequence of destinations in distribution. The original topology
    is completely destroyed while the number of edges and the amount of
    message passing is unchanged — the cleanest "is the graph doing
    anything" counterfactual.
    """
    g = torch.Generator().manual_seed(seed)
    dst = edge_index[1]
    perm = torch.randperm(dst.numel(), generator=g)
    return torch.stack([edge_index[0], dst[perm]], dim=0)


def drop_all_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return edge_index.new_zeros((2, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Single-variant trainer
# ─────────────────────────────────────────────────────────────────────────────

def _train_variant(
    data_full,
    variant:    str,
    cfg:        RunConfig,
    train_max:  int,
) -> dict:
    """Produce (train_graph, eval_graph) for the variant and train.

    ``real``:      untouched edges on train subgraph + full graph for eval.
    ``shuffled``:  shuffle BOTH train subgraph edges and eval-graph edges
                   (two independent permutations, same seed source).
    ``none``:      drop edges on both.
    """
    full = data_full
    sub  = build_inductive_view(full, train_max=train_max)

    if variant == "shuffled":
        sub.edge_index  = shuffle_edges(sub.edge_index,  seed=cfg.seed * 7 + 1)
        full = copy.copy(full)
        full.edge_index = shuffle_edges(full.edge_index, seed=cfg.seed * 7 + 2)
    elif variant == "none":
        sub.edge_index  = drop_all_edges(sub.edge_index)
        full = copy.copy(full)
        full.edge_index = drop_all_edges(full.edge_index)
    elif variant != "real":
        raise ValueError(f"unknown variant: {variant}")

    built = build_model(
        cfg.model,
        in_channels     = full.num_node_features,
        hidden_channels = cfg.hidden_channels,
        num_layers      = cfg.num_layers,
        dropout         = cfg.dropout,
        out_channels    = 3,
    )
    logits, info = _train(built, sub, full, cfg, setting=f"ind-{variant}")
    m = compute_metrics(logits, full.y, full.test_mask)
    return {
        "f1":        m["f1"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "auc":       m["auc"],
        "wall_sec":  info["wall_sec"],
        "n_params":  count_parameters(built.model),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    datasets, models, seeds, base: RunConfig, out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ds in datasets:
        data = load_dataset(ds, scaler_mode=base.scaler_mode)
        print(describe_dataset(data))
        train_max = int(data.time_step[data.train_mask].max().item())

        for mdl in models:
            for sd in seeds:
                tag = f"{ds}__{mdl}__seed{sd}.json"
                path = os.path.join(out_dir, tag)
                if os.path.exists(path):
                    print(f"[skip] {tag}")
                    continue
                print(f"\n[run] {tag}")
                cfg = RunConfig(**{**asdict(base),
                                   "dataset": ds, "model": mdl, "seed": sd})
                out = {
                    "result_schema_version": "shuffle_v2",
                    "dataset": ds,
                    "model": mdl,
                    "seed": sd,
                    "hyperparams": asdict(cfg),
                    "protocol": {
                        "setting": "inductive",
                        "real": "true observed graph",
                        "shuffled": "destination permutation; same edge count, topology destroyed",
                        "none": "empty edge_index on train and evaluation graphs",
                    },
                    "scaler_mode": data.meta.get("scaler_mode", cfg.scaler_mode),
                    "run_metadata": build_run_metadata("run_shuffle_ablation_multi", asdict(cfg)),
                }
                try:
                    for variant in ("real", "shuffled", "none"):
                        out[variant] = _train_variant(
                            data, variant, cfg, train_max=train_max,
                        )
                except Exception as exc:
                    out["error"] = str(exc)
                    print(f"[err] {exc}")
                write_json_atomic(path, out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+",
                   default=["elliptic", "dgraphfin", "tfinance"])
    p.add_argument("--models",   nargs="+",
                   default=["sage", "graph_transformer", "gps", "pcgnn"])
    p.add_argument("--seeds",    nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs",   type=int,   default=200)
    p.add_argument("--hidden",   type=int,   default=256)
    p.add_argument("--layers",   type=int,   default=3)
    p.add_argument("--dropout",  type=float, default=0.5)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--patience", type=int,   default=40)
    p.add_argument("--device",   type=str,   default="auto")
    p.add_argument("--out",      type=str,   default="results/shuffle_multi")
    p.add_argument(
        "--scaler-mode",
        choices=["train_only", "full_population", "none"],
        default="train_only",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Write an artifact manifest for the selected grid and exit without loading data or training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = RunConfig(
        hidden_channels = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        lr              = args.lr,
        epochs          = args.epochs,
        patience        = args.patience,
        device          = args.device,
        scaler_mode     = args.scaler_mode,
        results_dir     = args.out,
    )
    print("=" * 60)
    print("CROSS-DATASET SHUFFLED-EDGES ABLATION")
    print("=" * 60)
    print(f"  datasets : {args.datasets}")
    print(f"  models   : {args.models}")
    print(f"  seeds    : {args.seeds}")
    print(f"  out_dir  : {args.out}")
    print("=" * 60)
    if args.plan_only:
        rows = audit_sweep(
            "shuffle",
            result_dir=args.out,
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
        )
        counts = summarize_rows(rows)
        print("  plan     : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        write_audit_reports(
            rows,
            csv_path="results/reports/shuffle_ablation_plan.csv",
            md_path="results/reports/shuffle_ablation_plan.md",
            title="Shuffle-ablation sweep plan",
        )
        print("[plan] wrote results/reports/shuffle_ablation_plan.md")
        print("[plan] no datasets loaded; no training launched")
        return
    run_sweep(args.datasets, args.models, args.seeds, base, args.out)


if __name__ == "__main__":
    main()
