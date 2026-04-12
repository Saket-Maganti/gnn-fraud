"""
experiments/run_inductive.py

Inductive evaluation of GraphSAGE on the Elliptic Bitcoin Dataset.

In the inductive setting the model is trained on a relabeled **subgraph**
containing only nodes from time steps 1-34 and only edges whose both
endpoints lie in that window.  No test-period feature vector and no edge
that touches a test-period node is ever visible to the training forward
pass, so neither message passing nor BatchNorm running statistics can leak
test-period information into the learned weights.  At inference time the
full adjacency is restored so that test-period nodes can still receive
messages from their (train-period) neighbourhoods — this is the standard
textbook inductive-evaluation protocol.

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
    build_inductive_subgraph,
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
    print("INDUCTIVE GraphSAGE  (train subgraph = steps 1-34 only)")
    print("=" * 60)

    full_data = load_elliptic()
    train_sub = build_inductive_subgraph(full_data, train_time_max=34)

    print(f"[inductive] full graph : {full_data.num_nodes:,} nodes, "
          f"{full_data.num_edges:,} edges")
    print(f"[inductive] train subg : {train_sub.num_nodes:,} nodes, "
          f"{train_sub.num_edges:,} edges "
          f"({100 * train_sub.num_edges / max(full_data.num_edges, 1):.1f}%)")

    # Train on the subgraph, evaluate on the full graph.  This is proper
    # inductive generalisation: the model learns a decision boundary using
    # only training-period signal, then is asked to classify unseen
    # test-period nodes whose neighbourhoods may include train-period
    # counterparties.
    result = train_and_evaluate(
        train_data   = train_sub,
        eval_data    = full_data,
        cfg          = cfg,
        setting_name = "inductive",
    )

    print("\nBest test metrics (inductive):")
    for k, v in result["best_metrics"].items():
        print(f"  {k:<10s} {v:.4f}")

    save_results(result, cfg._out_path)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
