"""
scripts/addition5_ablation.py
Addition 5: Ablation — GNN with vs without graph structure per time step.

Compares GNN and MLP at each test time step to show:
  - How much graph structure helps in early time steps vs late ones
  - Whether GNN advantage diminishes as fraud actors adapt (temporal drift)
  This is a specific, novel finding unique to this paper.

Codepath : scripts/addition5_ablation.py
Runtime  : ~8 min (trains MLP 3 seeds + loads GNN checkpoints)
Output   : results/additions/ablation_temporal.png
           results/additions/ablation_summary.csv
"""

import sys, os, torch, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from models.mlp_baseline import MLP
from utils.imbalance import apply_strategy
from utils.trainer_minibatch import train
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/additions", exist_ok=True)

CONFIG = dict(lr=1e-3, weight_decay=5e-4, epochs=300, patience=30,
              eval_every=5, batch_size=512)


def eval_timesteps(model, data, device):
    model.eval()
    data_d = data.to(device)
    with torch.no_grad():
        logits = model(data_d.x, data_d.edge_index)
    results = {}
    for t in range(35, 50):
        mask = data.test_mask & (data.time_step == t)
        if mask.sum().item() < 5:
            continue
        preds  = logits[mask].argmax(-1).cpu().numpy()
        labels = data.y[mask].cpu().numpy()
        pb = (preds==1).astype(int); lb = (labels==1).astype(int)
        results[t] = round(float(f1_score(lb, pb, zero_division=0)), 4)
    return results


def train_mlp_seeds(data, cw, seeds=[42, 123, 456]):
    all_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        aug_data, criterion = apply_strategy(data, "weighted", cw)
        model = MLP(in_channels=data.num_node_features,
                    hidden_channels=256, out_channels=3,
                    num_layers=3, dropout=0.5)
        t0 = time.time()
        result = train(model=model, data=aug_data, criterion=criterion,
                       config=CONFIG, device=torch.device("cpu"), verbose=False)
        model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
        ts_result = eval_timesteps(model, data, torch.device("cpu"))
        all_results.append(ts_result)
        print(f"  MLP seed {seed}: overall F1={result['best_metrics']['f1']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)")
    # Average across seeds
    steps = sorted(all_results[0].keys())
    return {t: round(np.mean([r.get(t, 0) for r in all_results]), 4)
            for t in steps}


def main():
    print("Loading data...")
    f, c, e = load_elliptic_raw()
    data   = preprocess(f, c, e)
    cw     = get_class_weights(data)
    device = torch.device("cpu")

    print("\n── Training MLP baseline (3 seeds) ──")
    mlp_ts = train_mlp_seeds(data, cw)

    print("\n── Loading GNN checkpoints ──")
    gnn_results = {}

    for model_name, strategy, label in [
        ("sage", "weighted",  "SAGE + weighted"),
        ("sage", "graph_aug", "SAGE + graph_aug"),
    ]:
        ckpt = f"results/{model_name}_{strategy}_model.pt"
        if not os.path.exists(ckpt):
            print(f"  Skipping {label} — run train.py first")
            continue
        model = build_model(model_name,
                            in_channels=data.num_node_features,
                            hidden_channels=256, num_layers=3,
                            dropout=0.5, out_channels=3)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        gnn_results[label] = eval_timesteps(model, data, device)
        print(f"  {label} loaded")

    steps = sorted(mlp_ts.keys())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation: Graph Structure Contribution per Time Step",
                 fontsize=13, fontweight="bold")

    ax1, ax2 = axes

    # Left: absolute F1 per timestep
    ax1.plot(steps, [mlp_ts[t] for t in steps],
             color="#6B7280", lw=2.5, marker="o", markersize=6,
             linestyle="--", label="MLP (no graph)")
    colors = ["#2563EB", "#059669"]
    for (label, ts_result), color in zip(gnn_results.items(), colors):
        ax1.plot(steps, [ts_result[t] for t in steps],
                 color=color, lw=2.5, marker="s", markersize=6, label=label)

    ax1.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.6,
                label="Step 42 (drift onset)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("F1 Score (Fraud Class)")
    ax1.set_title("Absolute F1 per Time Step")
    ax1.set_ylim(0.2, 1.0)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: GNN advantage over MLP per timestep
    for (label, ts_result), color in zip(gnn_results.items(), colors):
        advantage = [ts_result[t] - mlp_ts[t] for t in steps]
        ax2.plot(steps, advantage, color=color, lw=2.5,
                 marker="s", markersize=6, label=f"{label} − MLP")
        ax2.fill_between(steps, 0, advantage, color=color, alpha=0.15)

    ax2.axhline(0, color="black", lw=1, alpha=0.5)
    ax2.axvline(42, color="gray", linestyle=":", lw=1.2, alpha=0.6,
                label="Step 42 (drift onset)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("GNN Advantage (F1 delta vs MLP)")
    ax2.set_title("Graph Structure Advantage per Time Step\n"
                  "(positive = GNN beats MLP)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/additions/ablation_temporal.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Save summary CSV
    rows = []
    for t in steps:
        row = {"timestep": t, "MLP_f1": mlp_ts[t]}
        for label, ts_result in gnn_results.items():
            safe = label.replace(" ", "_").replace("+", "")
            row[f"{safe}_f1"]       = ts_result.get(t, 0)
            row[f"{safe}_advantage"] = round(ts_result.get(t,0) - mlp_ts[t], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/additions/ablation_summary.csv", index=False)

    # Key finding
    print("\n── ABLATION FINDING ──")
    print("GNN advantage over MLP per time step:")
    for (label, ts_result) in gnn_results.items():
        early_adv = np.mean([ts_result[t]-mlp_ts[t] for t in steps if t<=39])
        late_adv  = np.mean([ts_result[t]-mlp_ts[t] for t in steps if t>=44])
        print(f"  {label}: early advantage={early_adv:+.3f}  "
              f"late advantage={late_adv:+.3f}  "
              f"{'← graph helps more early' if early_adv > late_adv else ''}")

    print("\nSaved: results/additions/ablation_temporal.png")
    print("Saved: results/additions/ablation_summary.csv")


if __name__ == "__main__":
    main()
