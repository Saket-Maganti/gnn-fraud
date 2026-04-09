"""
experiments/run_seeds.py  — Level 3
5-seed variance experiment runner.

Runs every model × strategy combination 5 times with different random seeds.
Reports mean ± std for F1, precision, recall, AUC.
This is the format required for a publishable comparison paper.

Runtime on M4 Mac (CPU):
    Static models : 3 models × 4 strategies × 5 seeds × ~10 min = ~10 hrs
    Recommend     : run overnight with nohup

Usage:
    python experiments/run_seeds.py
    python experiments/run_seeds.py --models sage gat --strategies weighted graph_aug
    python experiments/run_seeds.py --seeds 3 --epochs 150   # faster dev run
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights, print_stats
from models.gnn import build_model, count_parameters
from utils.imbalance import apply_strategy
from utils.trainer import train
from utils.metrics import aggregate_seed_metrics, format_metrics_table

os.makedirs("results", exist_ok=True)

MODELS     = ["gcn", "sage", "gat"]
STRATEGIES = ["baseline", "weighted", "oversample", "graph_aug"]
N_SEEDS    = 5

CONFIG = {
    "lr":           1e-3,
    "weight_decay": 5e-4,
    "epochs":       300,
    "patience":     30,
    "eval_every":   10,
    "hidden":       256,
    "num_layers":   3,
    "dropout":      0.5,
}


def run_single(model_name, strategy, data, class_weights, config, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    aug_data, criterion = apply_strategy(data, strategy, class_weights)

    model = build_model(
        model_name,
        in_channels      = data.num_node_features,
        hidden_channels  = config["hidden"],
        num_layers       = config["num_layers"],
        dropout          = config["dropout"],
        out_channels     = 3,
    )

    result = train(
        model=model, data=aug_data, criterion=criterion,
        config=config, device=device, verbose=False,
    )
    return result["best_metrics"]


def resolve_device(arg: str = "auto") -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def run_all(args):
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device : {device}")
    print(f"Seeds  : {args.seeds}")
    print(f"Epochs : {args.epochs}")

    print("\nLoading data...")
    features, classes, edges = load_elliptic_raw()
    data         = preprocess(features, classes, edges)
    class_weights = get_class_weights(data)
    print_stats(data)

    config = {**CONFIG, "epochs": args.epochs}

    all_rows    = []
    grand_start = time.time()

    for model_name in args.models:
        for strategy in args.strategies:
            run_key  = f"{model_name}_{strategy}"
            print(f"\n{'='*56}")
            print(f"  {run_key}  ({args.seeds} seeds)")
            print(f"{'='*56}")

            seed_metrics = []
            for seed in range(args.seeds):
                t0 = time.time()
                print(f"  Seed {seed}...", end=" ", flush=True)
                m = run_single(
                    model_name, strategy, data, class_weights,
                    config, device, seed
                )
                elapsed = time.time() - t0
                print(f"F1={m['f1']:.4f}  ({elapsed/60:.1f} min)")
                seed_metrics.append(m)

            agg = aggregate_seed_metrics(seed_metrics)
            print(f"\n{format_metrics_table(agg)}")

            row = {
                "model":    model_name,
                "strategy": strategy,
                "seeds":    args.seeds,
            }
            for metric, vals in agg.items():
                row[f"{metric}_mean"] = vals["mean"]
                row[f"{metric}_std"]  = vals["std"]
            all_rows.append(row)

            # Save incrementally
            pd.DataFrame(all_rows).to_csv("results/seed_results.csv", index=False)

    df       = pd.DataFrame(all_rows)
    elapsed  = (time.time() - grand_start) / 3600
    print(f"\n\nTotal runtime: {elapsed:.2f} hrs")
    print("\nFINAL RESULTS (F1 mean ± std):")
    pivot = df.pivot_table(
        index="strategy", columns="model",
        values=["f1_mean", "f1_std"], aggfunc="first"
    )
    print(pivot.round(4).to_string())

    with open("results/seed_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print("\nSaved: results/seed_results.csv, results/seed_results.json")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=MODELS,
                        choices=["gcn","sage","gat","edgegat","evolvegcn"])
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES,
                        choices=["baseline","weighted","oversample","graph_aug","adaptive"])
    parser.add_argument("--seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    args = parser.parse_args()
    run_all(args)
