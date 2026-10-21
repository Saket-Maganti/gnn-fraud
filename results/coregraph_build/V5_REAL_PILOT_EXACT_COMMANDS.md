# V5 real-pilot exact commands

These commands are an operator handoff. The real execution block was not run during final repair.

Set explicit machine-local paths without committing their values:

```bash
export COREGRAPH_REPO_ROOT="<ABSOLUTE_COREGRAPH_CHECKOUT>"
export COREGRAPH_EVIDENCE_CACHE="<ABSOLUTE_CANONICAL_EVIDENCE_CACHE>"
export COREGRAPH_OUTPUT_ROOT="<ABSOLUTE_NEW_REAL_V5_OUTPUT_ROOT>"
cd "$COREGRAPH_REPO_ROOT"
```

Preflight and authority:

```bash
git fetch origin --prune
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/coregraph-iclr-buildout-2026
gh pr view 2 --json state,isDraft,mergedAt,mergeable,baseRefName,headRefName,headRefOid
sha256sum specifications/V5_SAVED_OUTPUT_PILOT_SPECIFICATION.md
.venv/bin/python scripts/coregraph/hash_frozen_assets.py --verify
df -h "$(dirname "$COREGRAPH_OUTPUT_ROOT")"
```

Plan and record the exact clean-tip effective execution hash:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --plan \
  --chunk-rows 50000 \
  --max-workers 1
```

No-training validation:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --validate-only \
  --chunk-rows 50000 \
  --max-workers 1
```

Permitted synthetic rehearsal uses a separate root:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --output-root "${COREGRAPH_OUTPUT_ROOT}-synthetic" \
  --synthetic-fixture \
  --execute \
  --resume \
  --chunk-rows 3 \
  --max-workers 1 \
  --fail-fast
```

Later authorised real execution—do not run without explicit authority:

```bash
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

Read-only monitoring:

```bash
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -name COMPLETE | wc -l
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -path '*/failures/*.json' -print
tail -n 5 "$COREGRAPH_OUTPUT_ROOT/PILOT_PLAN.csv"
```

Exact packaging and post-package audit:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --package
unzip -t "${COREGRAPH_OUTPUT_ROOT}.zip"
```

Do not add method/scenario subsets, change worker count, bypass a dirty tree, alter a manifest, or reuse a root whose effective hash differs.
