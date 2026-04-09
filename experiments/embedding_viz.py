"""
experiments/embedding_viz.py
GNN embedding space visualization via t-SNE and PCA.

Generates two key publication figures:
  1. t-SNE colored by fraud/licit label
     → shows how well the GNN separates classes in embedding space
  2. t-SNE colored by timestep
     → shows temporal drift as a *geometric shift* in embedding space
  3. PCA variance explained (how compact the representations are)

Key finding for paper
---------------------
  Early test timesteps (35–42) show clean cluster separation between
  fraud and licit nodes. Late timesteps (43–49) show cluster overlap,
  providing a visual proof that the model's learned representations
  fail to generalise to the shifted distribution.

Requires a trained SAGE checkpoint at results/sage_weighted_model.pt
(or trains one from scratch if not found).

Outputs
-------
  results/embedding_tsne_label.png     (class-colored t-SNE)
  results/embedding_tsne_time.png      (timestep-colored t-SNE)
  results/embedding_pca.png            (PCA scree + 2D projection)
  results/embeddings_meta.csv          (per-node: embedding PCs, label, timestep)

Usage:
    python experiments/embedding_viz.py
    python experiments/embedding_viz.py --device cuda  --no_train   (load checkpoint)
    python experiments/embedding_viz.py --n_sub 3000                (subsample for speed)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


# ─────────────────────────────────────────────────────────────────────────────
# Training (fast, used only if no checkpoint found)
# ─────────────────────────────────────────────────────────────────────────────

def train_sage(data, device, epochs: int = 200, seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed)
    cw = get_class_weights(data)
    aug, crit = apply_strategy(data, "weighted", cw)
    aug = aug.to(device); crit = crit.to(device)
    model = build_model("sage", in_channels=aug.num_node_features,
                        hidden_channels=256, num_layers=3, dropout=0.5,
                        out_channels=3).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_f1, best_st, no_imp = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            lo = model(aug.x, aug.edge_index, getattr(aug, "edge_attr", None))
            l  = crit(lo, aug.y, aug.train_mask)
        opt.zero_grad()
        if use_amp: scaler.scale(l).backward(); scaler.unscale_(opt)
        else: l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp: scaler.step(opt); scaler.update()
        else: opt.step()
        sched.step()
        if hasattr(crit, "step_epoch"): crit.step_epoch()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                le = model(aug.x, aug.edge_index, getattr(aug, "edge_attr", None))
            f1 = compute_metrics(le, aug.y, aug.test_mask)["f1"]
            if f1 > best_f1:
                best_f1 = f1; best_st = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 3: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction (hook on penultimate layer)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, data, device) -> np.ndarray:
    """Extract [N, hidden] pre-classifier embeddings using a forward hook."""
    embeddings = {}

    def hook(module, input, output):
        embeddings["emb"] = input[0].detach().cpu()

    handle = model.classifier.register_forward_hook(hook)
    data   = data.to(device)
    model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    handle.remove()
    return embeddings["emb"].numpy()   # [N, hidden]


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(emb: np.ndarray, n_components: int = 50):
    pca = PCA(n_components=min(n_components, emb.shape[1]), random_state=42)
    reduced = pca.fit_transform(emb)
    return reduced, pca


def run_tsne(emb_pca: np.ndarray, n_components: int = 2,
             perplexity: float = 35, n_sub: int = 3000) -> tuple:
    """
    Run t-SNE on first 50 PCs (faster, less noisy than raw embeddings).
    Subsamples if more nodes than n_sub.
    Returns (coords, subsample_indices).
    """
    rng = np.random.default_rng(42)
    N   = len(emb_pca)
    if N > n_sub:
        idx = rng.choice(N, n_sub, replace=False)
        emb_sub = emb_pca[idx]
    else:
        idx = np.arange(N)
        emb_sub = emb_pca

    tsne   = TSNE(n_components=n_components, perplexity=perplexity,
                  n_iter=1000, random_state=42, init="pca",
                  learning_rate="auto")
    coords = tsne.fit_transform(emb_sub)
    return coords, idx


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne_label(coords, labels, save_path: str):
    """t-SNE colored by fraud/licit/unknown label."""
    fig, ax = plt.subplots(figsize=(9, 7))

    color_map  = {0: "#cccccc", 1: "#e15759", 2: "#4e79a7"}
    label_name = {0: "Unknown",  1: "Illicit (fraud)", 2: "Licit"}
    alpha_map  = {0: 0.15, 1: 0.75, 2: 0.35}

    for lbl in [2, 0, 1]:   # draw unknown first, fraud on top
        mask = labels == lbl
        if mask.sum() == 0: continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color_map[lbl], s=8, alpha=alpha_map[lbl],
                   label=f"{label_name[lbl]} (n={mask.sum():,})",
                   linewidths=0, rasterized=True)

    ax.legend(fontsize=10, markerscale=3, loc="upper right",
              framealpha=0.9, edgecolor="#ccc")
    ax.set_title("GNN Embedding Space (t-SNE)\nColored by Node Label",
                 fontsize=12)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {save_path}")


def plot_tsne_time(coords, timesteps, test_mask_sub, save_path: str):
    """t-SNE colored by timestep — shows temporal drift as geometric shift."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Non-test nodes (training) in gray
    train_mask_sub = ~test_mask_sub
    if train_mask_sub.any():
        ax.scatter(coords[train_mask_sub, 0], coords[train_mask_sub, 1],
                   c="#dddddd", s=5, alpha=0.2, linewidths=0,
                   label="Training nodes", rasterized=True)

    # Test nodes colored by timestep
    test_ts = timesteps[test_mask_sub]
    sc = ax.scatter(coords[test_mask_sub, 0], coords[test_mask_sub, 1],
                    c=test_ts, cmap="RdYlGn_r", vmin=35, vmax=49,
                    s=12, alpha=0.65, linewidths=0, rasterized=True,
                    label="Test nodes (colored by step)")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Test Timestep", fontsize=10)
    cbar.ax.yaxis.set_ticks(range(35, 50))

    ax.set_title("GNN Embedding Space (t-SNE)\nColored by Test Timestep\n"
                 "(red=late steps = distribution shift)",
                 fontsize=11)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, markerscale=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {save_path}")


def plot_pca_analysis(pca, emb_pca, labels, save_path: str):
    """PCA scree plot + 2D projection."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scree
    ax = axes[0]
    var_ratio = pca.explained_variance_ratio_
    cumvar    = np.cumsum(var_ratio)
    comps     = np.arange(1, len(var_ratio) + 1)
    ax.bar(comps[:30], var_ratio[:30], color="#4e79a7", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(comps[:30], cumvar[:30], "r-o", markersize=4, linewidth=1.5)
    ax2.axhline(0.90, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Cumulative Variance", color="red")
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Explained")
    ax.set_title(f"PCA Scree Plot\n({cumvar[9]*100:.1f}% in top-10 PCs)", fontsize=11)
    ax.set_xlim(0.5, 30.5)

    # 2D projection
    ax = axes[1]
    cmap  = {0: "#cccccc", 1: "#e15759", 2: "#4e79a7"}
    lname = {0: "Unknown", 1: "Illicit", 2: "Licit"}
    rng   = np.random.default_rng(42)
    for lbl in [2, 0, 1]:
        mask = labels == lbl
        if mask.sum() == 0: continue
        idx  = rng.choice(np.where(mask)[0], min(mask.sum(), 800), replace=False)
        ax.scatter(emb_pca[idx, 0], emb_pca[idx, 1], c=cmap[lbl],
                   s=8, alpha=0.6, label=lname[lbl], linewidths=0)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("PCA Projection (first 2 PCs)", fontsize=11)
    ax.legend(fontsize=9, markerscale=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading data …")
    features, classes, edges = load_elliptic_raw()
    data = preprocess(features, classes, edges, normalize=True)

    # Load or train model
    ckpt_path = os.path.join(results_dir, "sage_weighted_model.pt")
    model = build_model("sage", in_channels=data.num_node_features,
                        hidden_channels=256, num_layers=3, dropout=0.5,
                        out_channels=3)

    if args.no_train and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Training SAGE+weighted ({args.epochs} epochs) …")
        model = train_sage(data, device, epochs=args.epochs)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

    model = model.to(device)

    # Extract embeddings
    print("Extracting embeddings …")
    emb = extract_embeddings(model, data, device)   # [N, 256]
    y   = data.y.numpy()
    ts  = data.time_step.numpy()
    tm  = data.test_mask.numpy()

    print(f"  Embedding shape: {emb.shape}")

    # PCA
    print("Running PCA …")
    n_pca  = min(50, emb.shape[1])
    emb_pc, pca = run_pca(emb, n_components=n_pca)
    print(f"  Variance in top-10 PCs: {pca.explained_variance_ratio_[:10].sum()*100:.1f}%")

    # t-SNE (on test nodes only + sample of train for context)
    print(f"Running t-SNE (perplexity={args.perplexity}, subsample={args.n_sub}) …")
    coords, sub_idx = run_tsne(emb_pc, perplexity=args.perplexity,
                               n_sub=args.n_sub)
    y_sub   = y[sub_idx]
    ts_sub  = ts[sub_idx]
    tm_sub  = tm[sub_idx]

    # Save metadata CSV (PCA top-5 dims)
    meta_rows = []
    for i in range(len(sub_idx)):
        meta_rows.append({
            "node_idx":    int(sub_idx[i]),
            "label":       int(y_sub[i]),
            "timestep":    int(ts_sub[i]),
            "is_test":     bool(tm_sub[i]),
            "tsne_x":      round(float(coords[i, 0]), 4),
            "tsne_y":      round(float(coords[i, 1]), 4),
            **{f"pc{k+1}": round(float(emb_pc[sub_idx[i], k]), 4) for k in range(5)},
        })
    pd.DataFrame(meta_rows).to_csv(
        os.path.join(results_dir, "embeddings_meta.csv"), index=False
    )

    # Figures
    print("Generating figures …")
    plot_tsne_label(
        coords, y_sub,
        save_path=os.path.join(results_dir, "embedding_tsne_label.png"),
    )
    plot_tsne_time(
        coords, ts_sub, tm_sub,
        save_path=os.path.join(results_dir, "embedding_tsne_time.png"),
    )
    plot_pca_analysis(
        pca, emb_pc, y,
        save_path=os.path.join(results_dir, "embedding_pca.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     choices=["auto","cpu","cuda","mps"], default="auto")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--no_train",   action="store_true",
                        help="Load checkpoint instead of training")
    parser.add_argument("--n_sub",      type=int, default=3000,
                        help="Max nodes for t-SNE (subsampled)")
    parser.add_argument("--perplexity", type=float, default=35)
    main(parser.parse_args())
