# V5 saved-output pilot execution runbook

Status: `EXECUTOR_VALIDATED_EMPIRICAL_PILOT_UNEXECUTED`

Repository: `${COREGRAPH_REPO_ROOT}`  
Branch: `codex/coregraph-iclr-buildout-2026`  
Evidence cache: `${COREGRAPH_EVIDENCE_CACHE}`  
Config: `configs/coregraph/pilot/saved_output_v5.yaml`  
Runner: `scripts/coregraph/run_saved_output_pilot_v5.py`

The saved-output pilot is CPU-first. Do not launch Kaggle or a GPU job for this campaign unless later measured evidence justifies changing the operational venue. Never modify the preregistration or config after viewing target results.

Before using any command, set three explicit paths without committing their
machine-local values:

```bash
export COREGRAPH_REPO_ROOT="<ABSOLUTE_COREGRAPH_CHECKOUT>"
export COREGRAPH_EVIDENCE_CACHE="<ABSOLUTE_CANONICAL_EVIDENCE_CACHE>"
export COREGRAPH_OUTPUT_ROOT="<ABSOLUTE_NEW_PILOT_OUTPUT_ROOT>"
```

## Safe preflight

```bash
cd "$COREGRAPH_REPO_ROOT"
git fetch origin --prune
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/coregraph-iclr-buildout-2026
gh pr view 2 --json state,isDraft,mergedAt,baseRefName,headRefName,headRefOid
df -h "$COREGRAPH_OUTPUT_ROOT"
```

The tree must be clean, local and remote SHAs must match, and PR #2 must remain open, draft, and unmerged. Keep at least 5 GiB free on the output filesystem; the runner also records actual free space and clearly labels its output-size values as estimates.

## Plan without fitting

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --plan
```

Verify `PILOT_PLAN.csv` has exactly 240 rows and that the report states 6 archives, 180 base artifacts, 60 scenarios, 540 bindings, and 240 coordinates.

## Full no-training validation

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --validate-only
```

This verifies all six archive hashes and ZIP CRCs, all 180 member hashes, the 180/60/540 surface, scenario-local 6+3 bindings, indexed chronology/order/alignment, and the 240-coordinate plan. It never fits or evaluates a method.

## Permitted synthetic rehearsal

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --output-root "${COREGRAPH_OUTPUT_ROOT}-synthetic-smoke" \
  --synthetic-fixture \
  --execute \
  --fail-fast
```

This builds tiny deterministic ZIP fixtures and exercises the same 180/60/540/240 schemas, all four methods, policy freezing, offline evaluation, aggregation, checksums, and the three-outcome gate implementation. Its numbers are synthetic and must never enter the paper.

## Later authorised real execution

Run only after an explicit new authorization decision:

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --execute \
  --resume \
  --chunk-rows 50000 \
  --max-workers 1 \
  --authorization-token AUTHORIZE_COREGRAPH_V5_PILOT_RUN
```

The runner creates `RUN_MANIFEST.json`, `PILOT_PLAN.csv`, and `PILOT_PLAN.sha256` before fitting. Each method moves atomically through `PLANNED`, `INPUTS_VALIDATED`, `SOURCE_ASSEMBLED`, `POLICY_FITTED`, `POLICY_FROZEN`, `TARGET_SCORED`, `EVALUATED`, and `COMPLETE`; failures become explicit records. Resume reuses only exact hash-matching `COMPLETE` coordinates. Changed code, config, preregistration, archive/member identity, dependency lock, scenario fingerprint, method, output schema, or file bytes makes a coordinate stale and forces a rerun.

Monitor without changing files:

```bash
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -name COMPLETE | wc -l
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -path '*/failures/*.json' -print
tail -n 5 "$COREGRAPH_OUTPUT_ROOT/PILOT_PLAN.csv"
```

After all 240 coordinates complete, package exactly once:

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --package
```

Packaging refuses an incomplete run and writes `OUTPUT_CHECKSUMS.sha256` before the ZIP. Do not populate paper results until the packaged output, gate result, and scientific claims pass a separate independent audit.
