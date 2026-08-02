# CPU statistical analysis

Run only after a validated result import exists. Verify every
run manifest, code/config/data/prediction hash, completion status, execution
reason, feasibility set, and claim linkage. Reject partial or mismatched cells
before calculating summaries. Never pool same-number seeds across datasets as
one paired observation.

Apply the frozen per-dataset paired analysis, permutation/bootstrap procedure,
effect sizes, confidence intervals, Holm correction, missing-cell policy,
resource-blocked policy, and hierarchical secondary summary exactly as
preregistered. Emit machine-readable provenance for every table cell. Do not
run training, change gates after seeing outcomes, or write paper conclusions;
return supported, unsupported, inconclusive, and blocked claim decisions.

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
