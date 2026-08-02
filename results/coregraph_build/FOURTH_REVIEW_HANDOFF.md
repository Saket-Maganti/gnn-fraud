# Fourth-review handoff

Verdict:
`COREGRAPH_V4_MANIFEST_CONVERSION_BLOCKED_METADATA_UNRESOLVED`

This is a conversion/readiness verdict only. It does not authorize the
saved-output pilot. A separate fourth independent review must decide whether
future completed manifests may be considered for pilot execution.

## Integration status

TR-01 through TR-08 are repaired and covered end to end. V4 keeps
`protocol_id`, `contract_coordinate_hash`, and complete `contract_id`
identities separate. Role-specific row filtering, exact split/label-known
alignment, provider-unknown rejection, per-artifact exclusions, typed
cross-role leakage, dataset-stratified inference, explicit risk names, and
corrected comparator taxonomy are enforced.

The deterministic fixture passes through tiny CSVs, V4 manifests, real
hash-bearing `DeploymentContract.contract_id` values, the frozen protocol
registry, row-scope and known-label filtering, typed leakage, the actual
`--validate-only` runner, exact result-row materialization, and no-training
gate completeness. It performs no fitting or metric computation.

## Historical conversion result

The read-only scan of `$COREGRAPH_REPO` and
`$HISTORICAL_GNN_FRAUD_REPO` found 201 requested-pattern prediction
candidates and nine prediction-validation reports.

- Elliptic: 196 candidates; 70 have validation evidence and pass structural
  row, split, label-known, score-domain, duplicate-ID, and timestamp audits.
- DGraphFin: five unvalidated one-row strict-inductive GCN candidates for
  seeds 1--3; no required feature-MLP/GraphSAGE or isolated/transductive grid
  is available.
- Converted V4 manifests: zero.
- Exact matrix: 360 cells; 82 ambiguous-historical, 48 metadata-blocked, and
  230 missing.

Every candidate lacks evidenced `contract_role`, full
`deployment_contract`, `config_hash`, `code_hash`, `compute_cost`, and
`compute_cost_provenance`. These fields were not guessed. Frozen discovered
aliases pass registry validation, but complete contract binding cannot run
without loadable manifests.

The exact candidate paths/checksums and unresolved fields are in the local
isolated tree:

- `results/coregraph_manifest_conversion_v4/discovery.json`
- `results/coregraph_manifest_conversion_v4/candidate_audits.json`
- `results/coregraph_manifest_conversion_v4/conversion_records.json`
- `results/coregraph_manifest_conversion_v4/no_training_audit_status.json`

Paths there use `$COREGRAPH_REPO` and `$HISTORICAL_GNN_FRAUD_REPO` aliases
with exact root-relative paths. The required-cell record is
`MANIFEST_COMPLETENESS_MATRIX.csv`.

## Audit boundary

Available rows received checksum, schema, split, label-known, provider-label,
score-domain, duplicate-ID, and timestamp audits without target metrics.
All 60 required typed source/target leakage reports are explicitly
`NOT_RUN_BLOCKED_INCOMPLETE_OR_UNRESOLVED_MANIFESTS`: no role-specific
loadable manifest pairs exist on which to run atomic overlap checks. Real
no-training runner validation and real gate completeness are blocked for the
same reason. They are not recorded as passes.

## Deterministic validation

- Compileall, Ruff, and mypy: pass.
- Full repository tests: 195 pass.
- Explicit critical-module coverage gates: pass.
- Theory, synthetic checks, CPU one-epoch smoke, and 12 notebooks: pass.
- Eight-page placeholder paper and claim audit: pass; eight empirical claims
  remain blocked.
- Anonymous release: 250 files; identity/path audit and package tests pass.
- Committed public tree: zero findings.
- Frozen boundary: `ZERO_TKDE_SCIENTIFIC_DELTAS` for all 249 files.

## Confirmed non-actions

No saved-output pilot was executed. CoReGraph and learned baselines were not
fit on real predictions. No target metric or oracle was computed. No official
baseline was installed. No dataset was downloaded, Kaggle was not launched,
and no empirical paper result was populated. PR #2 was not merged and no
force-push was used.

## Fourth-review prerequisite

An independent reviewer must first establish evidence-map records for the six
missing metadata fields, resolve duplicate historical candidates, and supply
the 230 missing role-specific cells. The converter should then be rerun.
Only an exact, loadable 360-cell matrix with passing contract bindings, all 60
typed leakage reports, the no-training runner, and no-training gate
completeness may be considered for a separate pilot-execution decision.
