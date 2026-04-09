"""
experiments/run_l4.py  — Level 4
Novel architecture + adaptive strategy experiments.

New in this file vs run_seeds.py:
  1. EvolveGCN as a 3rd architecture (temporal GNN)
  2. EdgeGAT — GAT with edge feature integration
  3. Adaptive strategy — DriftDetector auto-switches strategy mid-training
  4. Ablation: ±edge features, ±adaptive switching (8 ablation configs)

Runtime on M4 Mac (CPU):
    EvolveGCN        : ~20 min per run (fewer epochs, snapshot overhead)
    EdgeGAT          : ~15 min per run (edge feature NNConv adds ~30% overhead)
    Adaptive strategy: same as graph_aug (~12 min) + drift eval overhead (~1 min)
    Full L4 suite    : ~3-4 hrs

Usage:
    python experiments/run_l4.py                    # all L4 experiments
    python experiments/run_l4.py --exp evolvegcn    # just EvolveGCN
    python experiments/run_l4.py --exp edgegat      # just EdgeGAT
    python experiments/run_l4.py --exp adaptive     # just adaptive strategy
    python experiments/run_l4.py --exp ablation     # ablation study
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
from utils.imbalance import apply_strategy, DriftDetector, WeightedCELoss
from utils.trainer import train, train_evolve
from utils.metrics import aggregate_seed_metrics, format_metrics_table
from utils.temporal import eval_per_timestep, summarise_temporal

os.makedirs("results/l4", exist_ok=True)

BASE_CONFIG = {
    "lr": 1e-3, "weight_decay": 5e-4,
    "epochs": 300, "patience": 30, "eval_every": 10,
    "hidden": 256, "num_layers": 3, "dropout": 0.5,
}
EVOLVE_CONFIG = {
    "lr": 5e-4, "weight_decay": 5e-4,
    "epochs": 200, "patience": 25, "eval_every": 10,
    "hidden": 128, "num_layers": 2, "dropout": 0.4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: EvolveGCN
# ─────────────────────────────────────────────────────────────────────────────

def run_evolvegcn(data, class_weights, device, args):
    """
    Train EvolveGCN-O on temporal snapshots.
    Strategies: weighted CE (no augmentation — augmented nodes lack time labels)
    Seeds: 3 (slower per run)
    """
    print("\n" + "="*56)
    print("  EXPERIMENT 1: EvolveGCN-O")
    print("="*56)
    print("Architecture: GRU-evolved GCN weights, 2 layers, h=128")
    print(f"Runtime est : ~20 min/run × 3 seeds = ~60 min")

    from utils.trainer import train_evolve

    config = {**EVOLVE_CONFIG, "epochs": args.epochs_evolve}
    seed_metrics = []

    for seed in range(3):
        torch.manual_seed(seed)
        print(f"\n  Seed {seed}...", flush=True)
        t0 = time.time()

        model = build_model(
            "evolvegcn",
            in_channels     = data.num_node_features,
            hidden_channels = config["hidden"],
            num_layers      = config["num_layers"],
            dropout         = config["dropout"],
            out_channels    = 3,
        )
        print(f"  Parameters: {count_parameters(model):,}")

        _, criterion = apply_strategy(data, "weighted", class_weights)
        result = train_evolve(
            model=model, data=data, criterion=criterion,
            config=config, device=device, verbose=True,
        )
        seed_metrics.append(result["best_metrics"])
        elapsed = time.time() - t0
        print(f"  F1={result['best_metrics']['f1']:.4f}  ({elapsed/60:.1f} min)")

    agg = aggregate_seed_metrics(seed_metrics)
    print(f"\nEvolveGCN results (3 seeds):\n{format_metrics_table(agg)}")

    row = {"model": "evolvegcn", "strategy": "weighted"}
    for m, v in agg.items():
        row[f"{m}_mean"] = v["mean"]
        row[f"{m}_std"]  = v["std"]

    with open("results/l4/evolvegcn_results.json", "w") as f:
        json.dump({"agg": agg, "seeds": seed_metrics}, f, indent=2)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: EdgeGAT (edge features)
# ─────────────────────────────────────────────────────────────────────────────

def run_edgegat(data, class_weights, device, args):
    """
    Train EdgeGAT with 6-dim edge features (derived from node features).
    Compare vs standard GAT to quantify edge feature contribution.
    """
    print("\n" + "="*56)
    print("  EXPERIMENT 2: EdgeGAT (edge features)")
    print("="*56)
    print("Edge features: 6-dim (local/agg means + diffs)")
    print(f"Runtime est : ~15 min/run × 2 configs × 3 seeds = ~90 min")

    # Reload data with edge features
    print("  Reloading data with edge features...")
    from data.dataset import load_elliptic_raw, preprocess
    features, classes, edges = load_elliptic_raw()
    data_ef = preprocess(features, classes, edges, use_edge_features=True)
    cw_ef   = get_class_weights(data_ef)
    print(f"  Edge attr shape: {data_ef.edge_attr.shape}")

    config = {**BASE_CONFIG, "epochs": args.epochs}
    results = []

    for strategy in ["weighted", "graph_aug"]:
        print(f"\n  EdgeGAT + {strategy}")
        seed_metrics = []
        for seed in range(3):
            torch.manual_seed(seed)
            t0 = time.time()

            aug_data, criterion = apply_strategy(data_ef, strategy, cw_ef)
            model = build_model(
                "edgegat",
                in_channels     = data_ef.num_node_features,
                hidden_channels = config["hidden"],
                num_layers      = config["num_layers"],
                dropout         = config["dropout"],
                edge_dim        = 6,
                out_channels    = 3,
            )
            result = train(
                model=model, data=aug_data, criterion=criterion,
                config=config, device=device, verbose=False,
            )
            seed_metrics.append(result["best_metrics"])
            print(f"    Seed {seed}: F1={result['best_metrics']['f1']:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)")

        agg = aggregate_seed_metrics(seed_metrics)
        print(f"  {format_metrics_table(agg)}")

        row = {"model": "edgegat", "strategy": strategy}
        for m, v in agg.items():
            row[f"{m}_mean"] = v["mean"]
            row[f"{m}_std"]  = v["std"]
        results.append(row)

    with open("results/l4/edgegat_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Adaptive strategy
# ─────────────────────────────────────────────────────────────────────────────

def run_adaptive(data, class_weights, device, args):
    """
    Adaptive strategy: start with graph_aug, switch to weighted CE
    when DriftDetector fires. Compare vs fixed strategies.

    Key result to look for: adaptive >= graph_aug on recall in early
    time steps, but doesn't degrade as badly on later time steps.
    """
    print("\n" + "="*56)
    print("  EXPERIMENT 3: Adaptive Strategy Switching")
    print("="*56)
    print("Detector: window=5, threshold=0.04 F1 drop")
    print(f"Runtime est : ~12 min/run × 2 models × 3 seeds = ~72 min")

    config = {**BASE_CONFIG, "epochs": args.epochs}

    from utils.imbalance import WeightedCELoss
    results = []

    for model_name in ["sage", "gat"]:
        print(f"\n  {model_name.upper()} + adaptive")
        seed_metrics = []
        switch_epochs = []

        for seed in range(3):
            torch.manual_seed(seed)
            t0 = time.time()

            # Start with graph_aug data
            aug_data, init_criterion = apply_strategy(
                data, "graph_aug", class_weights
            )
            # Fallback: weighted CE (activates when drift detected)
            fallback = WeightedCELoss(class_weights)

            detector = DriftDetector(window=5, threshold=0.04)

            model = build_model(
                model_name,
                in_channels     = data.num_node_features,
                hidden_channels = config["hidden"],
                num_layers      = config["num_layers"],
                dropout         = config["dropout"],
                out_channels    = 3,
            )

            result = train(
                model=model, data=aug_data, criterion=init_criterion,
                config=config, device=device, verbose=True,
                drift_detector=detector,
                fallback_criterion=fallback,
            )
            seed_metrics.append(result["best_metrics"])
            switch_epochs.append(result["strategy_switches"])
            print(f"  Seed {seed}: F1={result['best_metrics']['f1']:.4f} "
                  f"switches@{result['strategy_switches']} "
                  f"({(time.time()-t0)/60:.1f} min)")

            # Temporal analysis on best model
            model.load_state_dict(
                torch.load("best_model.pt", map_location=device)
            )
            temporal = eval_per_timestep(model, data, device)
            print(summarise_temporal(temporal))

        agg = aggregate_seed_metrics(seed_metrics)
        print(f"\n{format_metrics_table(agg)}")
        print(f"Strategy switched at epochs: {switch_epochs}")

        row = {"model": model_name, "strategy": "adaptive",
               "switch_epochs": str(switch_epochs)}
        for m, v in agg.items():
            row[f"{m}_mean"] = v["mean"]
            row[f"{m}_std"]  = v["std"]
        results.append(row)

    with open("results/l4/adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(data, class_weights, device, args):
    """
    Ablation: quantify contribution of each L4 component.
    Configs:
      GAT + weighted             (no edge feats, no adaptive)
      EdgeGAT + weighted         (+edge feats, no adaptive)
      GAT + adaptive             (no edge feats, +adaptive)
      EdgeGAT + adaptive         (+edge feats, +adaptive) ← full L4
    Single seed, shorter epochs for speed.
    """
    print("\n" + "="*56)
    print("  EXPERIMENT 4: Ablation Study")
    print("="*56)
    print("Runtime est : 4 configs × ~15 min = ~60 min")

    from data.dataset import load_elliptic_raw, preprocess
    features, classes, edges = load_elliptic_raw()
    data_ef = preprocess(features, classes, edges, use_edge_features=True)
    cw_ef   = get_class_weights(data_ef)

    ablation_config = {**BASE_CONFIG, "epochs": min(args.epochs, 150), "patience": 20}

    ablations = [
        ("gat",     "weighted",  data,    class_weights, False),
        ("edgegat", "weighted",  data_ef, cw_ef,         False),
        ("gat",     "adaptive",  data,    class_weights, True),
        ("edgegat", "adaptive",  data_ef, cw_ef,         True),
    ]

    results = []
    for model_name, strategy, d, cw, use_adaptive in ablations:
        label = f"{model_name}+{strategy}"
        print(f"\n  {label}")
        t0 = time.time()

        aug_data, criterion = apply_strategy(d, strategy, cw)
        model = build_model(
            model_name,
            in_channels     = d.num_node_features,
            hidden_channels = ablation_config["hidden"],
            num_layers      = ablation_config["num_layers"],
            dropout         = ablation_config["dropout"],
            edge_dim        = 6 if model_name == "edgegat" else None,
            out_channels    = 3,
        )

        dd  = DriftDetector(window=5, threshold=0.04) if use_adaptive else None
        fb  = WeightedCELoss(cw) if use_adaptive else None

        result = train(
            model=model, data=aug_data, criterion=criterion,
            config=ablation_config, device=device, verbose=False,
            drift_detector=dd, fallback_criterion=fb,
        )
        elapsed = time.time() - t0
        m = result["best_metrics"]
        print(f"  F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"({elapsed/60:.1f} min)")

        results.append({
            "config":        label,
            "edge_features": model_name == "edgegat",
            "adaptive":      use_adaptive,
            **m,
        })

    with open("results/l4/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import pandas as pd
    df = pd.DataFrame(results)
    print("\nAblation summary:")
    print(df[["config","edge_features","adaptive","f1","precision","recall"]].to_string(index=False))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(data):
    from data.dataset import get_class_weights as _gcw
    return _gcw(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+",
                        default=["evolvegcn","edgegat","adaptive","ablation"],
                        choices=["evolvegcn","edgegat","adaptive","ablation"])
    parser.add_argument("--epochs",        type=int, default=300)
    parser.add_argument("--epochs_evolve", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from data.dataset import (load_elliptic_raw, preprocess,
                               get_class_weights, print_stats)
    features, classes, edges = load_elliptic_raw()
    data          = preprocess(features, classes, edges)
    class_weights = get_class_weights(data)
    print_stats(data)

    all_results = []

    if "evolvegcn" in args.exp:
        r = run_evolvegcn(data, class_weights, device, args)
        all_results.append(r)

    if "edgegat" in args.exp:
        rs = run_edgegat(data, class_weights, device, args)
        all_results.extend(rs)

    if "adaptive" in args.exp:
        rs = run_adaptive(data, class_weights, device, args)
        all_results.extend(rs)

    if "ablation" in args.exp:
        rs = run_ablation(data, class_weights, device, args)
        all_results.extend(rs)

    import pandas as pd
    pd.DataFrame(all_results).to_csv("results/l4/l4_all_results.csv", index=False)
    print("\nAll L4 results saved to results/l4/")
