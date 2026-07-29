#!/usr/bin/env bash
# run_full_upgrade.sh
#
# End-to-end orchestration for the revision that addresses reviewer
# feedback: three datasets × four modern models × five seeds, plus the
# shuffled-edges ablation and the TPC+TTA mechanism study.
#
# Expects GPU ≥ 16 GB. On CPU this will run but DGraphFin × GPS is slow
# (~several hours per seed). The script is idempotent: every stage checks
# for its output files and skips completed work, so interrupting and
# resuming is safe.
#
# Stages
# ------
#   0  smoke test              — catches wiring bugs before burning compute
#   1  multi-dataset sweep     — Table A of the paper (protocol gap)
#   2  shuffled-edges ablation — Table B (real vs. shuffled vs. none)
#   3  TPC+TTA mechanism study — Fig. 1 (which ingredient does the work)
#   4  aggregate to CSV tables — paper-ready artefacts
#
# Flags
#   --quick    : tiny sweep (2 datasets, 2 models, 1 seed, 20 epochs)
#   --plan-only : write no-training manifests for the selected sweep and exit
#   --plan-limit N : cap generated full-sweep jobs in --plan-only mode
#   --num-shards N / --shard-index I : generate one shard of a full plan
#   --skip-smoke : skip stage 0 (you already ran it)
#   --device X   : pass through to each runner (auto|cpu|cuda|mps)

set -eu
cd "$(dirname "$0")"

QUICK=0
PLAN_ONLY=0
PLAN_LIMIT=""
NUM_SHARDS=1
SHARD_INDEX=0
SKIP_SMOKE=0
DEVICE="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)      QUICK=1; shift ;;
    --plan-only)  PLAN_ONLY=1; shift ;;
    --plan-limit) PLAN_LIMIT="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --device)     DEVICE="$2"; shift 2 ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
done

PY=${PY:-python}

# ── Plan-only mode: no data loading, smoke test, or model training ───────────
if [ "${PLAN_ONLY}" -eq 1 ]; then
  if [ "${QUICK}" -eq 1 ]; then
    echo "=== quick plan-only : runner-specific manifests ==="
    ${PY} experiments/run_multi_dataset.py --quick --device "${DEVICE}" --plan-only
    ${PY} experiments/run_shuffle_ablation_multi.py \
      --datasets elliptic \
      --models sage \
      --seeds 42 \
      --epochs 20 \
      --device "${DEVICE}" \
      --plan-only
    ${PY} experiments/run_tpc_tta.py \
      --datasets elliptic \
      --models sage \
      --seeds 42 \
      --epochs 20 \
      --device "${DEVICE}" \
      --plan-only
  else
    echo "=== plan-only : artifact audit ==="
    ${PY} scripts/audit_result_artifacts.py --sweep all

    echo "=== plan-only : resumable job manifest ==="
    MANAGE_ARGS=(
      scripts/manage_heavy_sweeps.py
      --sweep all
      --python-bin "${PY}"
      --device "${DEVICE}"
      --num-shards "${NUM_SHARDS}"
      --shard-index "${SHARD_INDEX}"
    )
    if [ -n "${PLAN_LIMIT}" ]; then
      MANAGE_ARGS+=(--limit "${PLAN_LIMIT}")
    fi
    ${PY} "${MANAGE_ARGS[@]}"

    echo "=== plan-only : runner-specific manifests ==="
    ${PY} experiments/run_multi_dataset.py \
      --datasets elliptic dgraphfin tfinance \
      --models graph_transformer gps pcgnn snapshot_tgn sage gcn \
      --seeds 42 43 44 45 46 \
      --device "${DEVICE}" \
      --plan-only
    ${PY} experiments/run_shuffle_ablation_multi.py \
      --datasets elliptic dgraphfin tfinance \
      --models sage graph_transformer gps pcgnn \
      --seeds 42 43 44 \
      --device "${DEVICE}" \
      --plan-only
    ${PY} experiments/run_tpc_tta.py \
      --datasets elliptic dgraphfin tfinance \
      --models sage graph_transformer gps pcgnn \
      --seeds 42 43 44 \
      --device "${DEVICE}" \
      --plan-only
  fi

  echo ""
  echo "Plan complete. No datasets were loaded and no training was launched."
  if [ "${QUICK}" -eq 1 ]; then
    echo "Inspect results/reports/*_plan.md."
  else
    echo "Inspect results/reports/*_artifact_audit.md and results/reports/*_plan.md."
  fi
  exit 0
fi

# ── Stage 0: smoke test ─────────────────────────────────────────────────
if [ "${SKIP_SMOKE}" -eq 0 ]; then
  echo "=== stage 0 : smoke test ==="
  ${PY} scripts/smoke_test.py
fi

# ── Stage 1: multi-dataset sweep ────────────────────────────────────────
echo "=== stage 1 : multi-dataset sweep ==="
if [ "${QUICK}" -eq 1 ]; then
  ${PY} experiments/run_multi_dataset.py --quick --device "${DEVICE}"
else
  ${PY} experiments/run_multi_dataset.py \
    --datasets elliptic dgraphfin tfinance \
    --models   graph_transformer gps pcgnn snapshot_tgn sage gcn \
    --seeds    42 43 44 45 46 \
    --device   "${DEVICE}"
fi

# ── Stage 2: shuffled-edges ablation ────────────────────────────────────
echo "=== stage 2 : shuffled-edges ablation ==="
if [ "${QUICK}" -eq 1 ]; then
  ${PY} experiments/run_shuffle_ablation_multi.py \
    --datasets elliptic \
    --models   sage \
    --seeds    42 \
    --epochs   20 \
    --device   "${DEVICE}"
else
  ${PY} experiments/run_shuffle_ablation_multi.py \
    --datasets elliptic dgraphfin tfinance \
    --models   sage graph_transformer gps pcgnn \
    --seeds    42 43 44 \
    --device   "${DEVICE}"
fi

# ── Stage 3: TPC+TTA mechanism study ────────────────────────────────────
echo "=== stage 3 : TPC+TTA mechanism study ==="
if [ "${QUICK}" -eq 1 ]; then
  ${PY} experiments/run_tpc_tta.py \
    --datasets elliptic \
    --models   sage \
    --seeds    42 \
    --epochs   20 \
    --device   "${DEVICE}"
else
  ${PY} experiments/run_tpc_tta.py \
    --datasets elliptic dgraphfin tfinance \
    --models   sage graph_transformer gps pcgnn \
    --seeds    42 43 44 \
    --device   "${DEVICE}"
fi

# ── Stage 4: aggregate ──────────────────────────────────────────────────
echo "=== stage 4 : aggregation ==="
${PY} experiments/aggregate_multi.py

echo ""
echo "Done. Next steps:"
echo "  * Inspect results/aggregated/*.csv for the paper tables."
echo "  * Update README.md Section 'Results (revised)' with the new numbers."
