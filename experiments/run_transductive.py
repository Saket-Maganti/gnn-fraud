"""
experiments/run_transductive.py

Transductive evaluation of GraphSAGE on the Elliptic Bitcoin Dataset.

In the transductive setting the **full** graph (every node's features plus
every edge, including those that touch test-period nodes) is visible during
training.  Message passing can propagate information across edges that
connect training nodes to test-period nodes, and BatchNorm sees every
node's features; only the loss remains masked to the training label set.
This matches the protocol used by most published Elliptic GNN results.

Paired runner: experiments/run_inductive.py
Shared config: experiments/_leakage_gap_utils.LeakageGapConfig
Results file:  results/transductive_results.json
"""

import argparse
import os
import sys

# Allow running as `python experiments/run_transductive.py` from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments._leakage_gap_utils import (  # noqa: E402
    LeakageGapConfig,
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
                   default="results/transductive_results.json")
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
    print("TRANSDUCTIVE GraphSAGE  (full graph visible at training time)")
    print("=" * 60)

    data = load_elliptic()

    # Transductive: train AND eval on the exact same full-graph Data object.
    # Message passing during training therefore sees every edge, including
    # edges that touch test-period nodes.  The loss remains masked to
    # `data.train_mask`, which is how every published Elliptic GNN trains.
    result = train_and_evaluate(
        train_data   = data,
        eval_data    = data,
        cfg          = cfg,
        setting_name = "transductive",
    )

    print("\nBest test metrics (transductive):")
    for k, v in result["best_metrics"].items():
        print(f"  {k:<10s} {v:.4f}")

    save_results(result, cfg._out_path)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
