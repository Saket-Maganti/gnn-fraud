# Framework Validation Report

Status: **PASS**

The support relation was exercised with 14 in-memory claim mutations and evidence ablations. All expected transitions matched the validator. The cases include complete support, missing cells, a missing seed, a missing prediction manifest, scope widening into Large or Medium-GINE resource boundaries, failed V22 imports, V24 metadata-only stress labels, a non-independent construction alias, and two directional universal claims contradicted by observed cells.

## Status transitions

- `BLOCKED_INCOMPLETE_SCOPE`: 1
- `BLOCKED_INCOMPLETE_SEEDS`: 1
- `BLOCKED_MISSING_PREDICTIONS`: 1
- `EXCLUDED_CONSTRUCT_INVALID`: 3
- `EXCLUDED_INTEGRITY`: 1
- `REFUTED_IN_SCOPE`: 2
- `RESOURCE_BLOCKED`: 4
- `SUPPORTED`: 1

## False-promotion prevention

The audit identifies 310 result files and 310 prediction files that a filename/count-only pipeline could misclassify. The largest class is V24: 240 result and 240 prediction files are duplicate scientific cells carrying three metadata labels that never reach the benchmark harness. V22 contributes 38 result and 38 prediction files from three integrity-failed/noncomparable imports. The memory-reduced DGraphFin GAT outputs remain diagnostic rather than filling the fixed-configuration OOM cell.

## Deterministic regeneration

The saved deterministic regeneration audit was rehashed against the curated aggregate files. All 28 pre-existing generated analysis CSV hashes were identical, and all 8 audited canonical input hashes were unchanged. This validation does not claim that the framework proves scientific truth; it verifies the declared completeness, provenance, construct, prediction, and resource-status transitions.
