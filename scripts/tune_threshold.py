import sys, torch
sys.path.insert(0, '.')
from data.dataset import load_elliptic_raw, preprocess
from models.gnn import build_model
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

device = torch.device('cpu')
f, c, e = load_elliptic_raw()
data = preprocess(f, c, e)

model = build_model('sage', in_channels=data.num_node_features,
                    hidden_channels=256, num_layers=3,
                    dropout=0.3, out_channels=3)

import os
ckpt = 'best_model.pt'
if not os.path.exists(ckpt):
    print("No best_model.pt found — run train.py first")
    sys.exit(1)

model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
data = data.to(device)

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs  = torch.softmax(logits, dim=-1)[:, 1]

mask   = data.test_mask
p      = probs[mask].numpy()
y_true = (data.y[mask] == 1).numpy().astype(int)

print(f"Test illicit rate       : {y_true.mean()*100:.1f}%")
print(f"Mean predicted P(illicit): {p.mean()*100:.1f}%")
print()
print(f"{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 42)

best_f1, best_t = 0, 0.5
for t in np.arange(0.05, 0.70, 0.025):
    preds = (p >= t).astype(int)
    f1    = f1_score(y_true, preds, zero_division=0)
    prec  = precision_score(y_true, preds, zero_division=0)
    rec   = recall_score(y_true, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t
    marker = ' <-- best' if t == best_t else ''
    print(f"{t:>10.3f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f}{marker}")

print(f"\nBest threshold : {best_t:.3f}")
print(f"Best F1        : {best_f1:.4f}")
