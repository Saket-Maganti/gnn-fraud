"""
experiments/temporal_analysis.py  — Level 3 (key publishable finding)

Evaluates every strategy's F1 on each individual test time step (35-49).
Generates the temporal drift plot — the finding that makes this paper publishable:
  "Graph augmentation maximises recall but degrades after step ~42 due to temporal drift"

This is Figure 2 in the paper.

Runtime on M4 Mac:
    Train 4 models (~40 min total) → evaluate per timestep (~2 min) → plot
    If you already have best_model.pt checkpoints: just run --eval_only

Usage:
    python experiments/temporal_analysis.py
    python experiments/temporal_analysis.py --models sage gat --epochs 150
    python experiments/temporal_analysis.py --eval_only  # use saved checkpoints
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.trainer import train
from utils.temporal import eval_per_timestep, summarise_temporal
from utils.visualise import plot_temporal_drift

os.makedirs("results", exist_ok=True)

CONFIG = {
    "lr": 1e-3, "weight_decay": 5e-4,
    "epochs": 300, "patience": 30, "eval_every": 10,
    "hidden": 256, "num_layers": 3, "dropout": 0.5,
}

# For temporal analysis we focus on the 2 most interesting combos per model
EXPERIMENTS = [
    ("sage", "baseline"),
    ("sage", "weighted"),
    ("sage", "graph_aug"),
    ("gat",  "baseline"),
    ("gat",  "weighted"),
    ("gat",  "graph_aug"),
]


def resolve_device(arg: str = "auto") -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def run(args):
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    print("Loading data...")
    features, classes, edges = load_elliptic_raw()
    data          = preprocess(features, classes, edges)
    class_weights = get_class_weights(data)

    config   = {**CONFIG, "epochs": args.epochs}
    all_temporal = {}

    experiments = [(m, s) for m, s in EXPERIMENTS
                   if m in args.models and s in args.strategies]

    for model_name, strategy in experiments:
        key   = f"{model_name}_{strategy}"
        ckpt  = f"results/{key}_model.pt"
        print(f"\n── {key} ──")

        model = build_model(
            model_name,
            in_channels     = data.num_node_features,
            hidden_channels = config["hidden"],
            num_layers      = config["num_layers"],
            dropout         = config["dropout"],
            out_channels    = 3,
        )

        if args.eval_only and os.path.exists(ckpt):
            print(f"  Loading checkpoint: {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            aug_data, criterion = apply_strategy(data, strategy, class_weights)
            print(f"  Training ({config['epochs']} epochs max)...")
            result = train(
                model=model, data=aug_data, criterion=criterion,
                config=config, device=device, verbose=True,
            )
            model.load_state_dict(torch.load("best_model.pt", map_location=device))
            torch.save(model.state_dict(), ckpt)

        model = model.to(device)
        print("  Evaluating per time step...")
        temporal = eval_per_timestep(model, data, device, model_type="static")
        all_temporal[key] = temporal

        print(summarise_temporal(temporal))

    # Save results
    temporal_serialisable = {
        k: {str(t): v for t, v in vs.items()}
        for k, vs in all_temporal.items()
    }
    with open("results/temporal_results.json", "w") as f:
        json.dump(temporal_serialisable, f, indent=2)

    # Generate the temporal drift plot (Figure 2 of paper)
    plot_temporal_drift(
        all_temporal,
        save_path="results/temporal_drift.png",
    )
    print("\nSaved: results/temporal_results.json, results/temporal_drift.png")
    return all_temporal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+",
                        default=["sage", "gat"],
                        choices=["gcn","sage","gat"])
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline","weighted","graph_aug"],
                        choices=["baseline","weighted","oversample","graph_aug"])
    parser.add_argument("--epochs",    type=int, default=CONFIG["epochs"])
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load saved checkpoints")
    parser.add_argument("--device",    choices=["auto","cpu","cuda","mps"],
                        default="auto")
    args = parser.parse_args()
    run(args)
