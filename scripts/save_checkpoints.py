"""
scripts/save_checkpoints.py
Saves named checkpoints for all 4 main configs.
Must be run BEFORE additions 2,3,4,5 which load checkpoints.

Runtime: ~15 min total (4 configs × ~3.5 min each)
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.trainer_minibatch import train

os.makedirs("results", exist_ok=True)

CONFIGS = [
    ("sage", "weighted"),
    ("sage", "graph_aug"),
    ("gat",  "weighted"),
    ("gat",  "graph_aug"),
]
CONFIG = dict(lr=1e-3, weight_decay=5e-4, epochs=300, patience=30,
              eval_every=5, batch_size=512)

print("Loading data...")
f, c, e = load_elliptic_raw()
data = preprocess(f, c, e)
cw   = get_class_weights(data)

for model_name, strategy in CONFIGS:
    ckpt = f"results/{model_name}_{strategy}_model.pt"
    if os.path.exists(ckpt):
        print(f"  Skipping {model_name}+{strategy} — checkpoint exists")
        continue
    print(f"\n── {model_name} + {strategy} ──")
    torch.manual_seed(42)
    aug_data, criterion = apply_strategy(data, strategy, cw)
    model = build_model(model_name,
                        in_channels=data.num_node_features,
                        hidden_channels=128, num_layers=2,
                        dropout=0.5, out_channels=3)
    result = train(model=model, data=aug_data, criterion=criterion,
                   config=CONFIG, device=torch.device("cpu"), verbose=True)
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved: {ckpt}  F1={result['best_metrics']['f1']:.4f}")

print("\nAll checkpoints saved.")
