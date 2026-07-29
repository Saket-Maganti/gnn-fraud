# V4 manifest conversion report

Status: `COREGRAPH_V4_MANIFEST_CONVERSION_BLOCKED_METADATA_UNRESOLVED`

This was a read-only historical-artifact conversion and no-training audit.
No router or learned baseline was fitted, no target metric or oracle was
computed, and no historical prediction file was modified.

## Discovery

- Search roots: `$COREGRAPH_REPO`, `$HISTORICAL_GNN_FRAUD_REPO`
- Requested-pattern candidates: 201
- Prediction-validation reports inspected: 9
- Candidates with validation evidence: 70
- Structurally usable validated candidates: 70
- Conversion statuses: `{"BLOCKED_METADATA_UNRESOLVED": 201}`
- Expected-cell statuses: `{"BLOCKED_AMBIGUOUS_HISTORICAL_CANDIDATES": 82, "BLOCKED_METADATA_UNRESOLVED": 48, "MISSING_ARTIFACT": 230}`
- Frozen registry schema: `PASS_FROZEN_V4_REGISTRY`
- Complete contract binding audit: `BLOCKED_NO_LOADABLE_V4_MANIFESTS`

Candidates become loadable V4 manifests only when the original checksum,
full deployment contract, coordinate hash, complete contract ID, role,
config/code hashes, compute cost and its provenance are all evidenced.
Anything unresolved remains `BLOCKED_METADATA_UNRESOLVED`.

## Scope

The completeness matrix covers both artifact roles for every combination of
two datasets, three frozen protocol aliases, three experts, seeds 1--10 and
`fold0`. Split, `label_known`, score-domain, duplicate-ID and timestamp fields
were audited where present. Contract-registry and typed cross-role leakage
aliases were validated against the frozen registry; complete
coordinate/contract binding and typed cross-role leakage remain blocked for
cells without loadable V4 manifests.
No-training runner materialisation and gate completeness are likewise blocked
until the exact matrix is available.

The verdict is a manifest-conversion/readiness verdict only. It does not
authorize pilot execution.
