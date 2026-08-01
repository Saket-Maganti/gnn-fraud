# V5 executor implementation report

Status: `IMPLEMENTED_VALIDATED_REAL_PILOT_UNEXECUTED`

The authoritative V5 path is `scripts/coregraph/run_saved_output_pilot_v5.py`
with `configs/coregraph/pilot/saved_output_v5.yaml`. The frozen preregistration
hash is `9ace6ac65dfb7d9244c7ee74e651a66e91809d6c3479881f43b6d30bb815dc96`.

Implemented closure surface:

- strict role-neutral base-artifact, scenario, binding, coordinate, checkpoint,
  result, and gate types;
- exact 180 base artifacts, 60 scenarios, 540 bindings, and 240 primary
  coordinates;
- direct checksum-gated ZIP streaming with no permanent extraction, composite
  row identities, stable three-expert alignment, deterministic bounded source
  sampling, configurable target inference chunks, float32 target scores, and a
  sequential-safe default;
- a target-unlabelled bundle with no label field and a nonserialisable,
  single-use offline label vault that opens only after a checksum-bound policy
  freeze;
- exactly four primary methods: CoReGraph, uniform average, source-validation
  best fixed expert, and source-only logistic gate;
- source-only preprocessing, tuning, early stopping, threshold fitting, and
  deterministic seed separation;
- atomic checkpoints, explicit failure records, output hashes, stale-output
  rejection, exact resume, complete-run aggregation, three-outcome gate, and
  packaging refusal for incomplete runs;
- plan, validate-only, synthetic execute, guarded real execute, resume, shard,
  chunk, worker, output, cache, fail-fast, dry-run, and package interfaces;
- concrete local/Kaggle notebook wrappers and exact execution, checklist, and
  abort/recovery runbooks.

The real execution branch is reachable only with a clean tree and the exact
later-authorization token. No real method was fit and no real target metric or
oracle was computed during closure.
