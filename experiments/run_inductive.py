"""
experiments/run_inductive.py

Inductive evaluation of GraphSAGE on the Elliptic Bitcoin Dataset.

In the inductive setting the training adjacency is restricted to edges whose
**both** endpoints fall in the training window (time steps 1-34).  The model
therefore never propagates information through test-period nodes during
training.  At inference time the full adjacency is restored so that test
nodes can still receive messages from their (train-period) neighbourhoods.

Paired runner: experiments/run_transductive.py
Shared config: experiments/_leakage_gap_utils.LeakageGapConfig
Results file:  results/inductive_results.json
"""

import argparse
import os
import sys

# Allow running as `python experiments/run_inductive.py` from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments._leakage_gap_utils import (  # noqa: E402
    LeakageGapConfig,
    build_inductive_edge_index,
    load_elliptic,
    save_results,
    train_and_evaluate,
)


def parse_args() -> LeakageGapConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",   type=int,   default=200)
    p.add_argument("--hidden",   type=int,   default=256)
    p.add_argument("--layers",   type=int,   default=3)
    p.add_argument("--dropout",  type=float, default=0.5)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--device",   type=str,   default="auto")
    p.add_argument("--patience", type=int,   default=40)
    p.add_argument("--out",      type=str,
                   default="results/inductive_results.json")
    args = p.parse_args()
    cfg = LeakageGapConfig(
        model_name      = "sage",
        hidden_channels = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        lr              = args.lr,
        epochs          = args.epochs,
        patience        = args.patience,
        seed            = args.seed,
        device          = args.device,
    )
    cfg._out_path = args.out  # type: ignore[attr-defined]
    return cfg


def main() -> None:
    cfg = parse_args()
    print("=" * 60)
    print("INDUCTIVE GraphSAGE  (train adjacency restricted to steps 1-34)")
    print("=" * 60)

    data = load_elliptic()

    train_edge_index = build_inductive_edge_index(
        data.edge_index, data.time_step, train_time_max=34
    )
    eval_edge_index  = data.edge_index  # full graph for inference

    kept = train_edge_index.size(1)
    full = data.edge_index.size(1)
    print(f"[inductive] kept {kept:,}/{full:,} edges for training "
          f"({100 * kept / max(full,1):.1f}%)")

    result = train_and_evaluate(
        data             = data,
        train_edge_index = train_edge_index,
        eval_edge_index  = eval_edge_index,
        cfg              = cfg,
        setting_name     = "inductive",
    )

    print("\nBest test metrics (inductive):")
    for k, v in result["best_metrics"].items():
        print(f"  {k:<10s} {v:.4f}")

    save_results(result, cfg._out_path)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
