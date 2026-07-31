# Canonical RB09v3 Reconciliation

Verdict: `COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES`

## Canonical claim

`$HISTORICAL_GNN_FRAUD_REPO/results/runs_rb09v3/ARTIFACT_FAMILY.json` and
`predictions_manifest.json` agree on exactly 180 prediction CSVs:
2 datasets × 3 protocols × 3 experts × 10 seeds. The matching `runs.csv`
contains exactly 180 result rows. Source and target roles are not counted here.

## Reconciled current state

- Canonical inventory records: **180**
- Canonical or explicitly compatible local artifacts: **0**
- Index-referenced local prediction/archive members missing: **180**
- Result-sidecar/metadata primary blockers: **0**
- Integrity-confirmed never-created artifacts: **0**
- Raw same-coordinate navigation candidates inspected: **201**
- Authoritative evidence files inspected: **421**

The raw candidates are not promoted to compatible aliases without an
authoritative alias/checksum link. This avoids silently substituting a different
run family merely because a basename encodes the same coordinate.

## Previously consumed source archives

| recorded archive path | recorded archive SHA-256 | present now | indexed members |
|---|---|---:|---:|
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_inductive_isolated.zip` | `6ce0d2e37893a7a162d6d575347f6606eb63f90b7940cc3a259dc309cf88b8c8` | false | 30 |
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_strict_inductive.zip` | `e0055d3482107d16c7d52574b0a32adc1e7ae9236b67dbdf12f57327c0e6bce5` | false | 30 |
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_transductive.zip` | `6d0167aae53b681bb7ffc037b84c723869ba718f30feb955c333890bfe8783d5` | false | 30 |
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_inductive_isolated.zip` | `20f25a1f93604ea5eb8537c8808f9b69dae2fc82eccbccfd36c50c443aee94e8` | false | 30 |
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_strict_inductive.zip` | `24752f5ffdc082dc79ca5084701fccd04d2ac9588b4b15712598bdfe8daa1e4a` | false | 30 |
| `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_transductive.zip` | `99d2f7ad1ad95fd7c30c193da9003c091b0c3fdce028dccd2bd0019f35869c08` | false | 30 |

All six archives are Category D recovery dependencies when absent locally.
Their SHA-256 values and all 180 member paths survive in RB15/RB16 import
manifests. No new model run is justified while these indexed archives remain
recoverable from their original external storage.

## Scenario consequence

The corrected surface contains 180 base cells, 60 held-out-protocol scenarios,
and 540 role bindings. Since the immutable CSV bytes are unavailable locally,
row-scope and scenario leakage materialisation remain blocked; no target metric,
oracle, router fit, or pilot execution was attempted.
