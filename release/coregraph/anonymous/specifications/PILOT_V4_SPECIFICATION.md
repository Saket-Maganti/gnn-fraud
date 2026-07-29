# Saved-output pilot V4 specification

Status: `FROZEN_BEFORE_HISTORICAL_MANIFEST_CONVERSION`

The machine-readable gate is
`PILOT_GATE_FROZEN_SPEC_V4.json`, SHA-256
`dcd676f38f0ee75bd9b57bc611989fc3285584da1c05b699a00528f312cb9a88`.
The protocol registry is `CONTRACT_PROTOCOL_REGISTRY_V4.json`, SHA-256
`64bd03a1cd9994833565f6be1d80e4e196da55810ca7236924cba048a4fd8de6`.
The V4 prediction-manifest schema has SHA-256
`b99d20e2b2fb8f1fbae483a1167e217a95f96d4e7cc7f45e1464195359f369c9`.

## Contract identities

V4 keeps three identities separate:

- `protocol_id` is the frozen semantic alias used for completeness:
  `strict_inductive`, `isolated_inductive`, or
  `transductive_structure`.
- `contract_coordinate_hash` is the exact scientific coordinate hash.
- `contract_id` is the complete `environment_id:artifact-prefix` identity.

Every target `(dataset, protocol_id, expert_prediction_seed, fold)` must bind
to exactly one coordinate hash and one complete contract ID. Reverse
coordinate/alias collisions are errors.

## Row and label scope

Source artifacts contribute only canonical `train` and `validation` rows.
Target artifacts contribute only their declared `test` evaluation split.
`unscored` rows are excluded. Provider split tokens are mapped explicitly;
the historical `val` to V4 `validation` normalization is evidenced by the
historical validation and TPC/TTA readers. Source rows in scope must be
label-known. Target scoring and offline oracle construction use only known
target rows. Provider class `0` remains unknown and is never converted to the
negative class. Every artifact emits included/excluded split and unknown-label
counts.

## Leakage

One typed report is required for every
`(dataset, target_protocol_id, expert_prediction_seed, fold)`. Atomic ID or
split overlap, duplicate rows, reused files or checksums under incompatible
metadata, timestamp-order violations, role/hash disagreement, target-label
selection metadata, and held-out coordinate equivalence block readiness.

## Risk and comparator taxonomy

- Training surrogate: `bce_surrogate_contract_regret`.
- Headline evaluation: `brier_contract_regret`.
- Selective evaluation: `selective_zero_one_risk`.

Losses are not compared across these definitions. The partial
`graphsafe_confidence_abstention_component` is a compatibility component, not
a complete strong baseline and not part of headline Holm families. The
contract-feasible oracle is the headline reference; the instance-clairvoyant
oracle is an excluded offline diagnostic ceiling.

## Inference

Pairing is exact within dataset and expert-prediction seed. Elliptic and
DGraphFin seeds with the same number are not a shared block. Exact Wilcoxon,
paired permutation, and seed-block bootstrap results are reported separately
per dataset. A hierarchical dataset-then-seed bootstrap is secondary evidence.
The robust gate requires positive effects on both datasets, corrected support
on at least one, and no contradictory dataset effect.

## Execution boundary

`--validate-only` may load manifests, validate the registry, filter/audit
rows, run typed leakage checks, and materialize the exact result-key surface.
It must keep training, metric computation, and target-oracle measurement
false. This specification does not authorize `--execute`.
