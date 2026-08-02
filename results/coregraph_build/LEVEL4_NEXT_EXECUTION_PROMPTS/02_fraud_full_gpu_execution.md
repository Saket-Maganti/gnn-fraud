# Fraud full GPU execution

Execute only after a separately reviewed pilot GO and explicit
user authorisation. Freeze the accepted pilot code/config boundary, verify the
Kaggle dataset copy hashes, and use the validated fraud-full notebook/run
matrix. Create all manifests before jobs start. Partition seeds 1–10 across the
two T4 lanes without duplicate coordinates, checkpoint atomically, resume only
after hash equivalence, and classify OOM as `RESOURCE_BLOCKED_OOM`.

Run only the approved fraud training and ablation cells. Maintain source-only
selection and target-label-free fitting. Package validated outputs into one
checksum-indexed ZIP per runbook, with explicit failed/missing coordinates and
no silent skips. Do not reinterpret training runtime as inference latency. Do
not populate paper results until local import, completeness, statistical, and
claim gates pass.

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
