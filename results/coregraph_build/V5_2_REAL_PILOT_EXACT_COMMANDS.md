# V5.2 Real Pilot Exact Commands

After the focused repair commit is pushed and exact-tip CI passes:

```bash
set -euo pipefail

REPO=/Users/saketmaganti/Projects/gnn-fraud/gnn-fraud-coregraph
EVIDENCE=/Users/saketmaganti/Projects/gnn-fraud/gnn-fraud-local-evidence-cache
SHORT_SHA="$(git -C "$REPO" rev-parse --short=7 HEAD)"
REAL_ROOT="/Users/saketmaganti/Projects/gnn-fraud/runs/coregraph-v5-real-pilot-v5.2-${SHORT_SHA}"
SYNTHETIC_ROOT="/Users/saketmaganti/Projects/gnn-fraud/runs/coregraph-v5-synthetic-rehearsal-v5.2-${SHORT_SHA}"
CONTROL_ROOT="/Users/saketmaganti/Projects/gnn-fraud/run-control/coregraph-v5-real-pilot-v5.2-${SHORT_SHA}"

cd "$REPO"

.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$EVIDENCE" \
  --output-root "$REAL_ROOT" \
  --plan --chunk-rows 50000 --max-workers 1

.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$EVIDENCE" \
  --output-root "$REAL_ROOT" \
  --validate-only --chunk-rows 50000 --max-workers 1

caffeinate -dimsu .venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$EVIDENCE" \
  --output-root "$REAL_ROOT" \
  --execute --resume --chunk-rows 50000 --max-workers 1 \
  --authorization-token AUTHORIZE_COREGRAPH_V5_PILOT_RUN
```

The synthetic rehearsal uses `SYNTHETIC_ROOT`, `--synthetic-fixture`, `--execute`, `--resume`, `--chunk-rows 3`, `--max-workers 1`, and `--fail-fast`.
