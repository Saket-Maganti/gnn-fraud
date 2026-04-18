#!/usr/bin/env bash
# run_overnight.sh — PRE-RUN + OVERNIGHT batch for paper rigor pass.
#
# Stage order (GPU sequential; FIX 8a runs concurrently on CPU):
#   BG (CPU): FIX 8a  — scaler refit 10-seed RF        ~20 min
#   GPU: FIX 4 inductive    10 seeds                   ~60 min
#   GPU: FIX 4 transductive 10 seeds                   ~60 min
#   GPU: FIX 2 hybrid       10 seeds (5 cells)         ~120 min
#   GPU: FIX 3 shuffle      10 seeds (3 conditions)    ~180 min
# Total wall: ~7h GPU + CPU side stage parallel

set -u
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/overnight_${STAMP}"
mkdir -p "$LOG_DIR"

# Use the pyenv-managed Python shim (has torch + MPS + PyG wired up).
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
PYTHON=$(command -v python)

echo "[orchestrator] started $STAMP, logging to $LOG_DIR" | tee "$LOG_DIR/orchestrator.log"

# ── FIX 8a in background (CPU only) ───────────────────────────────────────
(
  echo "[fix8a] start $(date +%H:%M:%S)"
  "$PYTHON" experiments/scaler_refit_check.py \
    --seeds 10 \
    --out_dir "$LOG_DIR/fix8a_scaler_refit" \
    > "$LOG_DIR/fix8a.log" 2>&1
  echo "[fix8a] done  $(date +%H:%M:%S)  rc=$?"
) > "$LOG_DIR/fix8a.ok" 2>&1 &
FIX8A_PID=$!
echo "[orchestrator] fix8a launched pid=$FIX8A_PID" | tee -a "$LOG_DIR/orchestrator.log"

# ── FIX 4 sequential inductive + transductive, 10 seeds each ──────────────
echo "[fix4] start inductive $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
mkdir -p "$LOG_DIR/fix4_inductive" "$LOG_DIR/fix4_transductive"
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "  [fix4:ind seed=$s] $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
  "$PYTHON" experiments/run_inductive.py \
    --seed $s \
    --out "$LOG_DIR/fix4_inductive/inductive_seed${s}.json" \
    > "$LOG_DIR/fix4_inductive/inductive_seed${s}.log" 2>&1
done

echo "[fix4] start transductive $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "  [fix4:trans seed=$s] $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
  "$PYTHON" experiments/run_transductive.py \
    --seed $s \
    --out "$LOG_DIR/fix4_transductive/transductive_seed${s}.json" \
    > "$LOG_DIR/fix4_transductive/transductive_seed${s}.log" 2>&1
done
echo "[fix4] done $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"

# ── FIX 2 hybrid — 10 seeds, 5 cells ─────────────────────────────────────
echo "[fix2] start $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
"$PYTHON" experiments/mlp_embedding_baseline.py \
  --seeds 10 \
  --out_dir "$LOG_DIR/fix2_hybrid" \
  > "$LOG_DIR/fix2.log" 2>&1
echo "[fix2] done $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"

# ── FIX 3 shuffle — 10 seeds, 3 conditions ───────────────────────────────
echo "[fix3] start $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
"$PYTHON" experiments/graph_ablation_extended.py \
  --n_seeds 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --no_plot \
  > "$LOG_DIR/fix3.log" 2>&1
# graph_ablation_extended writes its output under results/ by default;
# copy the summary into the overnight dir for consolidated inspection.
if [ -f results/graph_ablation_extended.csv ]; then
  mkdir -p "$LOG_DIR/fix3_shuffle"
  cp results/graph_ablation_extended.csv "$LOG_DIR/fix3_shuffle/"
fi
echo "[fix3] done $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"

wait $FIX8A_PID
echo "[orchestrator] all stages complete $(date +%H:%M:%S)" | tee -a "$LOG_DIR/orchestrator.log"
echo "[orchestrator] log dir: $LOG_DIR"
