"""
Proper training with validation split.
Val = time steps 30-34 (from training period)
Test = time steps 35-49 (never seen during training)
Early stopping on VAL F1, not test F1.
"""
import torch, os, sys, argparse
sys.path.insert(0, '.')
from data.dataset import load_elliptic_raw, preprocess, get_class_weights, print_stats
from models.gnn import build_model, count_parameters
from utils.imbalance import apply_strategy
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import torch.nn.functional as F
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="sage", choices=["gcn","sage","gat"])
parser.add_argument("--strategy", default="weighted",
                    choices=["baseline","weighted","focal","oversample","graph_aug"])
parser.add_argument("--hidden",   type=int,   default=128)
parser.add_argument("--layers",   type=int,   default=2)
parser.add_argument("--dropout",  type=float, default=0.5)
parser.add_argument("--lr",       type=float, default=1e-3)
parser.add_argument("--epochs",   type=int,   default=500)
parser.add_argument("--patience", type=int,   default=40)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cpu")

features, classes, edges = load_elliptic_raw()
data = preprocess(features, classes, edges)
print_stats(data)
cw = get_class_weights(data)

# Proper split — val from time steps 30-34, test from 35-49
labeled     = data.y != 0
val_mask    = labeled & (data.time_step >= 30) & (data.time_step <= 34)
train_mask  = labeled & (data.time_step < 30)
test_mask   = labeled & (data.time_step > 34)
data.train_mask = train_mask
data.val_mask   = val_mask
data.test_mask  = test_mask

print(f"Train: {train_mask.sum().item():,} nodes (steps 1-29)")
print(f"Val  : {val_mask.sum().item():,} nodes (steps 30-34)")
print(f"Test : {test_mask.sum().item():,} nodes (steps 35-49)")

aug_data, criterion = apply_strategy(data, args.strategy, cw)

model = build_model(args.model,
    in_channels=data.num_node_features,
    hidden_channels=args.hidden,
    num_layers=args.layers,
    dropout=args.dropout,
    out_channels=3)
print(f"Parameters: {count_parameters(model):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def get_metrics(logits, y, mask):
    p = logits[mask].argmax(-1).cpu().numpy()
    t = y[mask].cpu().numpy()
    pb = (p==1).astype(int); tb = (t==1).astype(int)
    return {
        "f1":   round(f1_score(tb, pb, zero_division=0), 4),
        "prec": round(precision_score(tb, pb, zero_division=0), 4),
        "rec":  round(recall_score(tb, pb, zero_division=0), 4),
    }

best_val_f1, best_epoch = 0, 0
patience_ctr = 0
aug_data = aug_data.to(device)
model = model.to(device)
criterion = criterion.to(device)
t0 = time.time()

for epoch in range(1, args.epochs + 1):
    model.train()
    logits = model(aug_data.x, aug_data.edge_index)
    mask = aug_data.train_mask & (aug_data.y != 0)
    if hasattr(criterion, 'forward') and \
            'mask' in criterion.forward.__code__.co_varnames:
        loss = criterion(logits, aug_data.y, mask)
    else:
        loss = F.cross_entropy(logits[mask], aug_data.y[mask])
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    if hasattr(criterion, "step_epoch"):
        criterion.step_epoch()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(aug_data.x, aug_data.edge_index)
        # Use original data masks for val/test (not augmented)
        data_dev = data.to(device)
        with torch.no_grad():
            logits_orig = model(data_dev.x, data_dev.edge_index)
        val_m  = get_metrics(logits_orig, data_dev.y, data_dev.val_mask)
        test_m = get_metrics(logits_orig, data_dev.y, data_dev.test_mask)

        if epoch % 50 == 0:
            print(f"Ep {epoch:4d} | Loss {loss:.4f} | "
                  f"ValF1 {val_m['f1']:.4f} | "
                  f"TestF1 {test_m['f1']:.4f} | "
                  f"P {test_m['prec']:.4f} R {test_m['rec']:.4f} | "
                  f"{time.time()-t0:.0f}s")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch  = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch} (best val epoch: {best_epoch})")
            break

# Final eval on test
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()
data_dev = data.to(device)
with torch.no_grad():
    logits = model(data_dev.x, data_dev.edge_index)

mask   = data_dev.test_mask
preds  = logits[mask].argmax(-1).cpu().numpy()
labels = data_dev.y[mask].cpu().numpy()
known  = labels != 0

print("\n" + "="*52)
print(f"BEST MODEL: epoch {best_epoch}, val F1 {best_val_f1:.4f}")
print("TEST RESULTS:")
print(classification_report(
    labels[known], preds[known],
    labels=[1,2],
    target_names=["Illicit","Licit"],
    digits=4))
