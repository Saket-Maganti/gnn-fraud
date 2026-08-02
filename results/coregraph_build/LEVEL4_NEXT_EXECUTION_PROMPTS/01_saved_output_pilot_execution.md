# Saved-output pilot execution

The executor is implemented, but the real pilot remains unrun.
Execute only after a new explicit authorization decision. Read
`V5_SAVED_OUTPUT_PILOT_EXECUTION_RUNBOOK.md` and set absolute
`COREGRAPH_REPO_ROOT`, `COREGRAPH_EVIDENCE_CACHE`, and
`COREGRAPH_OUTPUT_ROOT` values without committing machine-local paths.

Run the exact no-training gates first:

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py   --config configs/coregraph/pilot/saved_output_v5.yaml   --evidence-cache "$COREGRAPH_EVIDENCE_CACHE"   --output-root "$COREGRAPH_OUTPUT_ROOT" --plan
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py   --config configs/coregraph/pilot/saved_output_v5.yaml   --evidence-cache "$COREGRAPH_EVIDENCE_CACHE"   --output-root "$COREGRAPH_OUTPUT_ROOT" --validate-only
```

Require exactly 6 archives, 180 base artifacts, 60 scenarios, 540 bindings,
240 coordinates, 180 member hashes, zero training, and zero target-label reads.
After later authorization, run sequentially and resumably:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py   --config configs/coregraph/pilot/saved_output_v5.yaml   --evidence-cache "$COREGRAPH_EVIDENCE_CACHE"   --output-root "$COREGRAPH_OUTPUT_ROOT"   --execute --resume --chunk-rows 50000 --max-workers 1   --authorization-token AUTHORIZE_COREGRAPH_V5_PILOT_RUN
```

The runner must refuse a dirty tree or missing token. Do not train experts or
regenerate predictions. Fit all deployable state on source train/validation
only; permit target known-label rows solely in the offline evaluator after each
policy and target-score hash is frozen. Resume only exact hash-valid COMPLETE
cells and retain explicit failures. After 240/240 valid completions, package:

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py   --output-root "$COREGRAPH_OUTPUT_ROOT" --package
```

Issue only the frozen GO/NO-GO/INCONCLUSIVE decision. Do not change thresholds,
start full training, populate empirical paper claims, or launch Kaggle work in
the same phase.

## Inherited non-negotiable controls

Work only in `${COREGRAPH_REPO_ROOT}` on
`codex/coregraph-iclr-buildout-2026`. Begin with read-only Git, authority,
manifest, and frozen-boundary checks; fetch without resetting; stop on an
unexpected branch, divergence, dirty user work, or failed checksum. Never
force-push or merge PR #2.

Use `${COREGRAPH_EVIDENCE_CACHE}` as the canonical RB09v3 authority. Require
the six archive hashes and the 180 member identities in the tracked manifests
to match before reading payload bytes. Do not use the SSD when the local cache
passes. Stream ZIP members; do not permanently extract prediction CSVs. Never
stage archives, predictions, data, checkpoints, credentials, private path maps,
or local runtime logs.

Preserve the 180 role-neutral artifacts, 60 held-out-protocol scenarios, 540
scenario-local bindings, source/target role separation, source-only fitting,
known-label evaluation filters, chronology, dataset identity, and seed-local
pairing. A target artifact may be evaluated only in its scenario-local target
role. Target labels and oracle quantities are forbidden during fitting,
threshold selection, calibration, model selection, or routing.

Run `scripts/coregraph/hash_frozen_assets.py --verify` before and after work and
require `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. Record code, config, data,
archive, member, prediction, environment, and output hashes. Fail closed on
missing or invalid coordinates. Never fabricate metrics, runtime, memory,
citations, CI, completeness, or scientific conclusions. Any new empirical
claim must pass the frozen claim and statistical gates before entering paper
prose.

## Required handoff

Report the exact Git SHA, input and output hashes, commands, completed and failed coordinates, leakage and frozen-boundary status, scientific conclusions permitted by the gate, remaining blockers, and the next separately authorised action.
