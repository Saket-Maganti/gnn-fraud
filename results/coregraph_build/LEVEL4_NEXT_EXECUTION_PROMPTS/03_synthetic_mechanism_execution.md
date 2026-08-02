# Synthetic mechanism execution

Execute only with explicit authorisation for the full synthetic
grid. Verify all 15 mechanism definitions and deterministic tiny-fixture hashes
first. Freeze pilot/full sample sizes, seeds, held-out compositions, expected
directional sanity checks, and failure conditions before observing outcomes.

Run the scalable suite with distinct source and held-out contracts, preserving
ground-truth mechanism metadata. Separate theorem/counterexample validation
from empirical performance. Report every mechanism, seed, failed coordinate,
and resource block; do not cherry-pick mechanisms or promote the optional
latent-discovery extension without its gate. Package manifests and outputs with
hashes, then run the preregistered analysis without changing hypotheses.

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
