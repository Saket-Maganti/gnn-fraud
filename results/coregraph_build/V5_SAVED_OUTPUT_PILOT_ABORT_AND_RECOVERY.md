# V5 saved-output pilot abort and recovery

Use Ctrl-C to stop the foreground runner. Atomic writes leave either the previous complete file or the next complete file; temporary files are never accepted as results. Do not kill the machine during filesystem recovery, edit checkpoint stages, copy a `COMPLETE` marker, or merge rows by hand.

Inspect state:

```bash
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -name checkpoint.json -print
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -path '*/failures/*.json' -print
find "$COREGRAPH_OUTPUT_ROOT/scenarios" -name COMPLETE | wc -l
```

Resume with the identical repository SHA, config, preregistration, evidence cache, dependency lock, output root, chunk setting, and authorization:

```bash
cd "$COREGRAPH_REPO_ROOT"
.venv/bin/python scripts/coregraph/run_saved_output_pilot_v5.py \
  --config configs/coregraph/pilot/saved_output_v5.yaml \
  --evidence-cache "$COREGRAPH_EVIDENCE_CACHE" \
  --output-root "$COREGRAPH_OUTPUT_ROOT" \
  --execute --resume --chunk-rows 50000 --max-workers 1 \
  --authorization-token AUTHORIZE_COREGRAPH_V5_PILOT_RUN
```

A corrupt, partial, failed, or stale coordinate is rerun atomically. A hash-valid complete coordinate is skipped. If an archive/member checksum changes, the preregistration differs, the Git tree is dirty, target scores differ from the freeze manifest, or target rows no longer align at evaluation, stop: this is an evidence-identity failure, not a retryable model failure. Preserve the output directory and request scientific review.

Packaging remains forbidden until all 240 coordinates are complete. Never alter gate thresholds, impute a failure, remove an infeasible cell, or populate paper results as part of recovery.
