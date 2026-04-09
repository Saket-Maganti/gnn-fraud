#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_insights.sh
# Execution script for the GNN fraud detection insight pipeline.
#
# Expected runtimes (Apple M-series CPU, Elliptic 203K nodes):
#   --fast  (100 epochs)  : ~15-25 min total
#   --full  (300 epochs)  : ~60-90 min total
#
# Per-part breakdown (full mode):
#   graph_stats   :  ~5-10 min   (building 5 graph variants + NX stats)
#   baselines     :  ~3-5  min   (LR + RF + XGBoost on 165-dim features)
#   gnn_variants  :  ~35-55 min  (5 graph types × ~7-11 min each)
#   ablation      :  ~15-25 min  (4 experiments × ~4-6 min each)
#   explain       :  ~1-2  min   (gradient saliency, no retraining if ckpt exists)
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# ── Resolve Python — prefer gnn_env, fall back to venv ───────────────────────
if [ -f "gnn_env/bin/python" ]; then
    PYTHON="gnn_env/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi
echo "Python: $($PYTHON --version 2>&1)  ($PYTHON)"

# ── Verify / install xgboost ──────────────────────────────────────────────────
$PYTHON -c "import xgboost" 2>/dev/null || $PYTHON -m pip install xgboost --quiet

# ── Parse flags ───────────────────────────────────────────────────────────────
MODE="--fast"       # default: fast mode for quick results
DEVICE="cpu"
PART="all"

for arg in "$@"; do
  case $arg in
    --full)   MODE=""       ;;
    --fast)   MODE="--fast" ;;
    --mps)    DEVICE="mps"  ;;
    --cuda)   DEVICE="cuda" ;;
    --part=*) PART="${arg#*=}" ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     GNN FRAUD DETECTION — INSIGHT PIPELINE                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  mode   : ${MODE:-full (300 epochs)}"
echo "  device : $DEVICE"
echo "  part   : $PART"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
$PYTHON experiments/run_insights.py \
    --part "$PART" \
    --device "$DEVICE" \
    $MODE

echo ""
echo "── Done.  Outputs: ──────────────────────────────────────────────"
echo "  results/metrics.csv               (GNN vs baselines vs hybrid)"
echo "  results/ablation_results.csv      (edge removal, feature removal)"
echo "  results/graph_stats.csv           (topology per graph variant)"
echo "  results/gnn_feature_saliency.csv  (GNN gradient importance)"
echo "  results/rf_feature_importance.csv (RF Gini importance)"
echo "  results/xgb_feature_importance.csv(XGBoost importance)"
echo "  results/feature_importance_comparison.png"
echo "  results/checkpoints/sage_*.pt     (GNN model checkpoints)"
echo "─────────────────────────────────────────────────────────────────"
