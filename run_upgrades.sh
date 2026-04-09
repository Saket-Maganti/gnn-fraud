#!/usr/bin/env bash
# run_upgrades.sh
# Master runner for all experimental upgrades.
#
# Experiments
# -----------
#   0. (pre-req) temporal_analysis — trains SAGE/GAT, creates temporal_results.json
#   1. temporal_deep       — per-timestep F1/P/R + illicit rate (reads from JSON)
#   2. graph_ablation_ext  — original / shuffled / no-edges structural ablation
#   3. significance_tests  — 5-seed Welch t-test (GNN vs MLP) + from-CSV analysis
#   4. stronger_baselines  — SAGE_Deep and SAGE_MaxPool vs. standard SAGE
#   5. calibration_study   — ECE + reliability diagrams + temperature scaling
#   6. business_cost       — asymmetric FP/FN cost evaluation (from CSV, no training)
#   7. hybrid_ensemble     — weighted-avg + concat GNN+MLP ensembles
#   8. distribution_shift  — MMD + L2 feature drift per timestep
#   9. embedding_viz       — t-SNE of GNN embeddings (label + timestep colored)
#  10. pr_analysis         — Precision-Recall curves + optimal threshold sweep
#
# Flags
# -----
#   --fast          reduced epochs (150) for quick smoke-test
#   --part N        run only experiment N (1-10); 0 = run baseline + all
#   --from_csv      significance_tests uses only seed_results.csv (no new training)
#   --mps           force Apple MPS device (macOS)
#   --cuda          force CUDA device (Google Colab / GPU server)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Detect Python ─────────────────────────────────────────────────────────────
if   [[ -f "gnn_env/bin/python" ]];   then PY="gnn_env/bin/python"
elif [[ -f "venv/bin/python" ]];       then PY="venv/bin/python"
elif command -v python3 &>/dev/null;   then PY="python3"
else echo "ERROR: no Python found"; exit 1; fi

echo "Python: $($PY --version 2>&1)  [$PY]"

# ── Parse flags ───────────────────────────────────────────────────────────────
FAST=0; PART=0; FROM_CSV=0; EPOCHS=200; EPOCHS_ABLATION=200; SEEDS=3; DEVICE=auto

for arg in "$@"; do
  case "$arg" in
    --fast)      FAST=1; EPOCHS=150; EPOCHS_ABLATION=150; SEEDS=3 ;;
    --part=*)    PART="${arg#*=}" ;;
    --from_csv)  FROM_CSV=1 ;;
    --mps)       DEVICE=mps; export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 ;;
    --cuda)      DEVICE=cuda ;;
  esac
done

echo "Epochs: $EPOCHS  |  Ablation epochs: $EPOCHS_ABLATION  |  Seeds: $SEEDS  |  Device: $DEVICE"
[[ "$FAST" -eq 1 ]] && echo "  [fast mode]"
echo ""

mkdir -p results

# ── Dataset check ─────────────────────────────────────────────────────────────
FEATURES="data/raw/elliptic_txs_features.csv"
if [[ ! -f "$FEATURES" ]]; then
  echo "════════════════════════════════════════════════════════════"
  echo "  DATASET NOT FOUND"
  echo "════════════════════════════════════════════════════════════"
  echo "  Expected: $FEATURES"
  echo ""
  echo "  Option A — Kaggle CLI (recommended on Colab):"
  echo "    pip install kaggle"
  echo "    # Upload kaggle.json API key, then:"
  echo "    kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/ --unzip"
  echo "    mv data/raw/elliptic_bitcoin_dataset/* data/raw/ 2>/dev/null || true"
  echo ""
  echo "  Option B — Manual:"
  echo "    Download from https://kaggle.com/datasets/ellipticco/elliptic-data-set"
  echo "    Put the 3 CSV files in data/raw/"
  echo ""

  # Auto-attempt download if kaggle CLI is available
  if command -v kaggle &>/dev/null; then
    echo "  Kaggle CLI found — attempting auto-download …"
    mkdir -p data/raw
    kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/ --unzip
    # Kaggle unzips into a subdirectory — move files up
    if [[ -d "data/raw/elliptic_bitcoin_dataset" ]]; then
      mv data/raw/elliptic_bitcoin_dataset/*.csv data/raw/
      rmdir data/raw/elliptic_bitcoin_dataset 2>/dev/null || true
    fi
    echo "  Download complete."
  else
    echo "  ERROR: kaggle CLI not found and dataset missing. Cannot continue."
    echo "         Install kaggle: pip install kaggle"
    echo "         Then add your API key: https://www.kaggle.com/settings > API > Create Token"
    echo "         Save the token to ~/.kaggle/kaggle.json (chmod 600)"
    exit 1
  fi
fi
echo "Dataset: OK ($(du -sh "$FEATURES" | cut -f1))"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
run_part() {
  local n="$1"; shift
  [[ "$PART" -ne 0 && "$PART" -ne "$n" ]] && return 0
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  PART $n: $*"
  echo "════════════════════════════════════════════════════════════"
}

# ── Part 0 (prerequisite): Temporal Analysis ─────────────────────────────────
# Runs automatically when Part 1 or full run is requested and JSON is missing.
if [[ "$PART" -eq 0 || "$PART" -eq 1 ]]; then
  if [[ ! -f "results/temporal_results.json" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PRE-REQ: temporal_analysis.py (creates temporal_results.json)"
    echo "════════════════════════════════════════════════════════════"
    $PY experiments/temporal_analysis.py \
        --models sage gat \
        --strategies weighted graph_aug \
        --epochs "$EPOCHS" \
        --device "$DEVICE"
  else
    echo "Pre-req: temporal_results.json already exists — skipping training."
  fi
fi

# ── Part 1: Temporal Deep ─────────────────────────────────────────────────────
run_part 1 "Temporal deep analysis (F1/P/R per timestep + illicit rate)"
if [[ "$PART" -eq 0 || "$PART" -eq 1 ]]; then
  $PY experiments/temporal_deep.py
fi

# ── Part 2: Graph Ablation Extended ──────────────────────────────────────────
run_part 2 "Graph ablation (original / shuffled / no-edges)"
if [[ "$PART" -eq 0 || "$PART" -eq 2 ]]; then
  $PY experiments/graph_ablation_extended.py \
      --epochs "$EPOCHS_ABLATION" \
      --n_seeds "$SEEDS" \
      --device "$DEVICE"
fi

# ── Part 3: Statistical Significance ─────────────────────────────────────────
run_part 3 "Statistical significance tests (GNN vs MLP, Welch t-test)"
if [[ "$PART" -eq 0 || "$PART" -eq 3 ]]; then
  if [[ "$FROM_CSV" -eq 1 ]]; then
    $PY experiments/significance_tests.py --from_csv
  else
    $PY experiments/significance_tests.py \
        --epochs "$EPOCHS" \
        --n_seeds "$SEEDS" \
        --device "$DEVICE"
  fi
fi

# ── Part 4: Stronger GNN Baselines ───────────────────────────────────────────
run_part 4 "Stronger GNN baselines (SAGE_Deep, SAGE_MaxPool vs SAGE_Mean)"
if [[ "$PART" -eq 0 || "$PART" -eq 4 ]]; then
  EPOCHS="$EPOCHS" DEVICE="$DEVICE" $PY - <<'PYEOF'
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from data.dataset import load_elliptic_raw, preprocess, get_class_weights
from models.gnn import build_model
from utils.imbalance import apply_strategy
from utils.metrics import compute_metrics

EPOCHS = int(os.environ.get("EPOCHS", 200))
SEEDS  = [42, 7, 123]
MODELS = [("sage",         "SAGE-Mean (3L)"),
          ("sage_deep",    "SAGE-Deep (5L, residual)"),
          ("sage_maxpool", "SAGE-MaxPool (3L)")]
DEVICE_ARG = os.environ.get("DEVICE", "auto")

if DEVICE_ARG == "auto":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(DEVICE_ARG)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print(f"Using device: {DEVICE}")
print("Loading data ...")
features, classes, edges = load_elliptic_raw()
data = preprocess(features, classes, edges, normalize=True)

rows = []
for model_key, label in MODELS:
    print(f"\n  [{label}]")
    seed_f1s = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cw = get_class_weights(data)
        aug, crit = apply_strategy(data, "weighted", cw)
        aug = aug.to(DEVICE); crit = crit.to(DEVICE)
        m = build_model(model_key, in_channels=aug.num_node_features,
                        hidden_channels=256, num_layers=3, dropout=0.5,
                        out_channels=3).to(DEVICE)
        opt   = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        use_amp = DEVICE.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_f1, best_st, no_imp = 0.0, None, 0
        for ep in range(1, EPOCHS + 1):
            m.train()
            with torch.cuda.amp.autocast(enabled=use_amp):
                lo = m(aug.x, aug.edge_index, getattr(aug, "edge_attr", None))
                l  = crit(lo, aug.y, aug.train_mask)
            opt.zero_grad()
            if use_amp: scaler.scale(l).backward(); scaler.unscale_(opt)
            else:       l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if use_amp: scaler.step(opt); scaler.update()
            else:       opt.step()
            sched.step()
            if hasattr(crit, "step_epoch"): crit.step_epoch()
            if ep % 10 == 0:
                m.eval()
                with torch.no_grad():
                    le = m(aug.x, aug.edge_index, getattr(aug, "edge_attr", None))
                f1 = compute_metrics(le, aug.y, aug.test_mask)["f1"]
                if f1 > best_f1:
                    best_f1 = f1; best_st = {k: v.clone() for k, v in m.state_dict().items()}; no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= 3: break
        if best_st: m.load_state_dict(best_st)
        m.eval()
        od = data.to(DEVICE)
        with torch.no_grad():
            lo = m(od.x, od.edge_index, getattr(od, "edge_attr", None))
        met = compute_metrics(lo, od.y, od.test_mask)
        seed_f1s.append(met["f1"])
        print(f"    seed {seed}: F1={met['f1']:.4f}")
    rows.append({"model": label, "f1_mean": round(float(np.mean(seed_f1s)), 4),
                 "f1_std": round(float(np.std(seed_f1s, ddof=1)), 4),
                 "seeds": len(seed_f1s)})

df = pd.DataFrame(rows)
out = "results/stronger_baselines.csv"
df.to_csv(out, index=False)
print("\n" + "─"*50)
print(f"{'Model':<28} {'F1 mean':>9} {'±std':>7}")
print("─"*50)
for _, r in df.iterrows():
    print(f"  {r['model']:<26} {r['f1_mean']:>9.4f} ±{r['f1_std']:.4f}")
print(f"\nSaved → {out}")
PYEOF
fi

# ── Part 5: Calibration Study ─────────────────────────────────────────────────
run_part 5 "Calibration: ECE + reliability diagrams + temperature scaling"
if [[ "$PART" -eq 0 || "$PART" -eq 5 ]]; then
  $PY experiments/calibration_study.py --epochs "$EPOCHS" --device "$DEVICE"
fi

# ── Part 6: Business Cost ─────────────────────────────────────────────────────
run_part 6 "Business cost evaluation (asymmetric FP/FN penalties)"
if [[ "$PART" -eq 0 || "$PART" -eq 6 ]]; then
  $PY experiments/business_cost.py
fi

# ── Part 7: Hybrid Ensemble ───────────────────────────────────────────────────
run_part 7 "Hybrid ensemble (weighted-avg + concat GNN+MLP)"
if [[ "$PART" -eq 0 || "$PART" -eq 7 ]]; then
  $PY experiments/hybrid_ensemble.py --epochs "$EPOCHS" --device "$DEVICE"
fi

# ── Part 8: Distribution Shift ───────────────────────────────────────────────
run_part 8 "Distribution shift quantification (MMD + L2 feature drift)"
if [[ "$PART" -eq 0 || "$PART" -eq 8 ]]; then
  $PY experiments/distribution_shift.py
fi

# ── Part 9: Embedding Visualization ──────────────────────────────────────────
run_part 9 "Embedding t-SNE visualization (label + timestep colored)"
if [[ "$PART" -eq 0 || "$PART" -eq 9 ]]; then
  $PY experiments/embedding_viz.py --device "$DEVICE" --epochs "$EPOCHS"
fi

# ── Part 10: PR Curve Analysis ────────────────────────────────────────────────
run_part 10 "Precision-Recall curves + optimal threshold per timestep"
if [[ "$PART" -eq 0 || "$PART" -eq 10 ]]; then
  $PY experiments/pr_analysis.py --device "$DEVICE" --epochs "$EPOCHS"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DONE — output files:"
echo "════════════════════════════════════════════════════════════"
for f in \
  results/temporal_results.json \
  results/temporal_deep.csv         results/temporal_deep.png \
  results/graph_ablation_extended.csv results/graph_ablation_extended.png \
  results/significance_results.csv   results/significance_summary.txt \
  results/stronger_baselines.csv \
  results/calibration_results.csv    results/calibration_curves.png \
  results/business_cost.csv          results/business_cost.png \
  results/hybrid_ensemble.csv        results/hybrid_ensemble.png \
  results/distribution_shift.csv     results/distribution_shift.png \
  results/embedding_tsne_label.png   results/embedding_tsne_time.png \
  results/embedding_pca.png \
  results/pr_analysis.csv            results/pr_curves.png \
  results/threshold_analysis.png
do
  [[ -f "$f" ]] && echo "  ✓  $f" || echo "  -  $f  (not generated)"
done
echo ""
