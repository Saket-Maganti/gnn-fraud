"""
train.py — mini-batch with RandomNodeLoader (M4 compatible)
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import (load_elliptic_raw, preprocess,
                           get_class_weights, print_stats)
from models.gnn import build_model, count_parameters
from utils.imbalance import apply_strategy
from utils.trainer_minibatch import train
from utils.visualise import plot_training_curves
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="sage",
                        choices=["gcn","sage","gat","edgegat","evolvegcn"])
    parser.add_argument("--strategy",   default="weighted",
                        choices=["baseline","weighted","focal",
                                 "oversample","graph_aug","adaptive"])
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--layers",     type=int,   default=3)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--patience",   type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--use_wandb",  action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Device   : {device}")
    print(f"Model    : {args.model.upper()}")
    print(f"Strategy : {args.strategy}")

    print("\nLoading Elliptic dataset...")
    features, classes, edges = load_elliptic_raw()
    data          = preprocess(features, classes, edges)
    print_stats(data)
    class_weights = get_class_weights(data)

    aug_data, criterion = apply_strategy(data, args.strategy, class_weights)

    model = build_model(
        args.model,
        in_channels     = data.num_node_features,
        hidden_channels = args.hidden,
        num_layers      = args.layers,
        dropout         = args.dropout,
        out_channels    = 3,
    )
    print(f"Parameters: {count_parameters(model):,}")

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="gnn-fraud-detection",
            name=f"{args.model}_{args.strategy}_mb",
            config=vars(args),
        )

    config = {
        "lr":            args.lr,
        "weight_decay":  5e-4,
        "epochs":        args.epochs,
        "patience":      args.patience,
        "eval_every":    5,
        "batch_size":    args.batch_size,
    }

    print(f"\nTraining (RandomNodeLoader, batch={args.batch_size})...")
    result = train(
        model=model, data=aug_data, criterion=criterion,
        config=config, device=device,
        wandb_run=wandb_run, verbose=True,
    )

    print("\n" + "="*52)
    print("BEST TEST METRICS:")
    for k, v in result["best_metrics"].items():
        print(f"  {k:<12}: {v:.4f}")

    # Classification report — load best checkpoint, run full-graph inference
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        model.eval()
        model = model.to(device)
        data_dev = data.to(device)

        with torch.no_grad():
            logits = model(data_dev.x, data_dev.edge_index)

        mask   = data_dev.test_mask
        preds  = logits[mask].argmax(-1).cpu().numpy()
        labels = data_dev.y[mask].cpu().numpy()

        # Filter out any unknowns
        known   = labels != 0
        preds   = preds[known]
        labels  = labels[known]

        print("\n── CLASSIFICATION REPORT (test set) ──")
        print(classification_report(
            labels, preds,
            labels=[1, 2],
            target_names=["Illicit (fraud)", "Licit (legit)"],
            digits=4,
        ))
        cm = confusion_matrix(labels, preds, labels=[1, 2])
        tn, fp, fn, tp = cm.ravel()
        print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    os.makedirs("results", exist_ok=True)
    plot_training_curves(
        result["history"],
        title=f"{args.model}_{args.strategy}",
        save_path=f"results/{args.model}_{args.strategy}_curves.png",
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()