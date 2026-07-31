# Future Run Necessity Report

Verdict: `COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES`

## Classification

- Category A — converter/discovery miss repaired: 0
- Category B — present archive not extracted: 0
- Category C — prediction present but metadata incomplete: 0
- Category D — canonical index references a missing local file/archive: 180
- Category E — artifact never existed or failed integrity: 0

## Decision

**No future GPU prediction-generation run is recommended.** Category D remains
non-zero and Category E is zero. The evidence proves the 180 members existed and
were consumed; the correct next action is canonical archive recovery, not
retraining.

## Exact archive dependencies

- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_inductive_isolated.zip` — expected SHA-256 `6ce0d2e37893a7a162d6d575347f6606eb63f90b7940cc3a259dc309cf88b8c8`; restore the exact archive and validate it before extracting or streaming its indexed members.
- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_strict_inductive.zip` — expected SHA-256 `e0055d3482107d16c7d52574b0a32adc1e7ae9236b67dbdf12f57327c0e6bce5`; restore the exact archive and validate it before extracting or streaming its indexed members.
- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/dgraphfin_10seed_transductive.zip` — expected SHA-256 `6d0167aae53b681bb7ffc037b84c723869ba718f30feb955c333890bfe8783d5`; restore the exact archive and validate it before extracting or streaming its indexed members.
- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_inductive_isolated.zip` — expected SHA-256 `20f25a1f93604ea5eb8537c8808f9b69dae2fc82eccbccfd36c50c443aee94e8`; restore the exact archive and validate it before extracting or streaming its indexed members.
- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_strict_inductive.zip` — expected SHA-256 `24752f5ffdc082dc79ca5084701fccd04d2ac9588b4b15712598bdfe8daa1e4a`; restore the exact archive and validate it before extracting or streaming its indexed members.
- `$HISTORICAL_GNN_FRAUD_REPO/kaggleoutputs/elliptic_10seed_transductive.zip` — expected SHA-256 `99d2f7ad1ad95fd7c30c193da9003c091b0c3fdce028dccd2bd0019f35869c08`; restore the exact archive and validate it before extracting or streaming its indexed members.

After restoring an archive, verify its recorded SHA-256, then rerun:

`python scripts/coregraph/recover_canonical_manifests_v5.py --historical-root "$HISTORICAL_GNN_FRAUD_REPO"`

The converter can stream ZIP members; extraction is not required. If extraction
is desired, use `unzip -n <exact-archive> -d <new-dedicated-directory>` only
after checksum verification.
