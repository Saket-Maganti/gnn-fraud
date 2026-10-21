"""
experiments/run_multi_dataset.py

Main driver for the cross-dataset, cross-model, cross-seed sweep that
backs the revised paper's central claim:

    "Evaluation protocols can reverse model rankings under temporal shift,
     and TPC+TTA closes most of the resulting gap — across three fraud
     datasets and four modern GNN families."

By default the sweep covers:
    datasets : elliptic, dgraphfin, tfinance
    models   : sage, graph_transformer, gps, pcgnn, snapshot_tgn, gcn,
               evolvegcn   (legacy anchors kept for continuity with the
               original paper results)
    seeds    : 42, 43, 44, 45, 46  (5 seeds — extend to 10 for the final
               paper table; the default keeps the sweep tractable on a
               single GPU overnight)

Each result is written as an individual JSON under ``results/multi/``. Run
``experiments/aggregate_multi.py`` afterward to produce the summary table
and per-dataset plots.

Usage
-----
    python experiments/run_multi_dataset.py                   # full default sweep
    python experiments/run_multi_dataset.py --quick           # small smoke test
    python experiments/run_multi_dataset.py \
        --datasets elliptic tfinance \
        --models sage gps \
        --seeds 42 43 44 \
        --epochs 150 --device cuda

Resume semantics
----------------
The harness skips any ``(dataset, model, seed)`` whose result file
already exists — drop or rename individual files under ``results/multi/``
to force a re-run of just those triples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments._multi_harness import RunConfig, run_grid       # noqa: E402
from experiments.result_audit import (                           # noqa: E402
    audit_sweep,
    summarize_rows,
    write_audit_reports,
    write_json_atomic,
)


DEFAULT_DATASETS = ["elliptic", "dgraphfin", "tfinance"]
DEFAULT_MODELS   = [
    # modern baselines (reviewer ask)
    "graph_transformer", "gps", "pcgnn", "snapshot_tgn",
    # legacy anchors to reproduce the original paper's numbers
    "sage", "gcn",
]
DEFAULT_SEEDS    = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models",   nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs",   type=int,   default=200)
    p.add_argument("--hidden",   type=int,   default=256)
    p.add_argument("--layers",   type=int,   default=3)
    p.add_argument("--dropout",  type=float, default=0.5)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--patience", type=int,   default=40)
    p.add_argument("--device",   type=str,   default="auto")
    p.add_argument("--out",      type=str,   default="results/multi")
    p.add_argument(
        "--scaler-mode",
        choices=["train_only", "full_population", "none"],
        default="train_only",
        help="Feature scaling mode (default: train_only).",
    )
    p.add_argument("--quick",    action="store_true",
                   help="Smoke-test: 2 datasets, 2 models, 1 seed, 20 epochs.")
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Write an artifact manifest for the selected grid and exit without loading data or training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        datasets = ["elliptic", "tfinance"]
        models   = ["sage", "graph_transformer"]
        seeds    = [42]
        epochs   = 20
        patience = 10
    else:
        datasets = args.datasets
        models   = args.models
        seeds    = args.seeds
        epochs   = args.epochs
        patience = args.patience

    base = RunConfig(
        hidden_channels = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        lr              = args.lr,
        epochs          = epochs,
        patience        = patience,
        device          = args.device,
        scaler_mode     = args.scaler_mode,
        results_dir     = args.out,
    )

    print("=" * 60)
    print("MULTI-DATASET SWEEP")
    print("=" * 60)
    print(f"  datasets : {datasets}")
    print(f"  models   : {models}")
    print(f"  seeds    : {seeds}")
    print(f"  epochs   : {epochs}")
    print(f"  out_dir  : {args.out}")
    print("=" * 60)

    if args.plan_only:
        rows = audit_sweep(
            "multi",
            result_dir=args.out,
            datasets=datasets,
            models=models,
            seeds=seeds,
        )
        counts = summarize_rows(rows)
        print("  plan     : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        write_audit_reports(
            rows,
            csv_path="results/reports/multi_dataset_plan.csv",
            md_path="results/reports/multi_dataset_plan.md",
            title="Multi-dataset sweep plan",
        )
        print("[plan] wrote results/reports/multi_dataset_plan.md")
        print("[plan] no datasets loaded; no training launched")
        return

    results = run_grid(
        datasets = datasets,
        models   = models,
        seeds    = seeds,
        base_cfg = base,
        out_dir  = args.out,
    )

    summary_path = os.path.join(args.out, "_all_runs.json")
    write_json_atomic(summary_path, results)
    print(f"\n[io] wrote combined summary: {summary_path}")
    print(f"[io] individual result files in: {args.out}/")


if __name__ == "__main__":
    main()
