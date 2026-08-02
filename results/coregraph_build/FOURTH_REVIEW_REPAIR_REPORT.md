# Fourth-Review Repair Report

Verdict: `COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES`

## Outcome

The runner–manifest–gate architecture is repaired and validated without
training or empirical scoring. V5 represents the scientific evidence as 180
role-neutral base artifact cells and represents evaluation use separately as
60 deterministic held-out-protocol scenarios with 540 role bindings. One base
artifact may be a source in scenario A and a target in scenario B; the same
artifact cannot occupy both roles inside one scenario.

The old 360-row V4 role-cell matrix is retained only as explicitly superseded
audit history. It double-counted the 180 scientific prediction CSVs and its
global reuse findings were not valid across independent scenarios.

## Integration repair

- `BasePredictionArtifact` contains immutable artifact identity, prediction
  checksum, role-neutral contract coordinates, row schema, provider
  split/label mappings, typed configuration/code provenance, separate routing
  cost and measured-compute evidence, archive lineage, and validation evidence.
- `EvaluationScenarioBinding` contains the scenario identity, held-out target
  protocol, two source protocols, target operational contract, access regime,
  and exactly six source plus three target bindings.
- Source bindings permit only train/validation rows; target bindings permit
  only known-label test rows. Provider-unknown target rows are excluded before
  scoring and target-label values are not exposed by the no-training output.
- Leakage is atomic only inside one scenario. It rejects protocol rebinding,
  source test scope, target train/validation scope, provider-unknown scoring,
  target-label fitting access, ID overlap, equal held-out coordinates,
  chronology violations, and same-scenario dual roles.
- The V5 runner refuses `--execute` and exposes only plan/validate paths during
  this review. The V5 gate evaluates manifest readiness only and always returns
  `pilot_authorized: false`.

The end-to-end fixture passes two tiny datasets, three protocols, three
experts, and two seeds through real CSV parsing, 36 base manifests, 12
scenarios, 108 bindings, row filtering, registry checks, scenario leakage,
runner validation, and the readiness gate. It proves that no training, fitting,
metric, oracle, or target-label-selection path is reachable.

## Canonical evidence recovery

The read-only search used this precedence: final evidence lock, final merged
prediction index, per-lane merge validation, package import validation, result
sidecar, then raw navigation. It inspected 421 authoritative evidence files,
including 1,498 structured prediction-index records, 5,372 structured
result-index/JSONL records, 31 locks, 147 import/alias sources, 95 validation
reports, 72 result sidecars, and six JSONL sources.

The canonical RB09v3 inventory and result sidecar agree on exactly 180
coordinates: two datasets × three protocols × three experts × ten seeds.
RB15/RB16 import evidence maps those 180 members to six source archives and
preserves each archive checksum. None of the six archive files is present under
either inspected repository. The 201 raw same-coordinate candidates have no
authoritative alias/checksum link to RB09v3 and were not silently substituted.

Current classification:

- `RECOVERED_CANONICAL`: 0
- `RECOVERED_COMPATIBLE_ALIAS`: 0
- `INDEX_REFERENCED_FILE_MISSING`: 180
- metadata-primary blockers: 0
- integrity-confirmed never-created artifacts: 0
- future-run categories A/B/C/E: 0
- future-run category D: 180

This is evidence that the canonical files previously existed and were
consumed, but their current local archive containers are missing. It is not
evidence that the prediction runs never occurred.

## Provenance

All 180 canonical result rows supply command/config fields and a per-run
runtime record. The historical code values are not evidenced 40-character Git
commits, so all 180 remain typed `UNRESOLVED_LEGACY_CODE`. No supported
routing-cost proxy is evidenced, so all 180 routing costs remain
`UNRESOLVED`. Base artifact hashes, row audits, and exact target operational
contract bindings remain blocked for all 180 until the canonical bytes are
restored and verified. No current-code hash, chunk-runtime allocation, cost
proxy, or contract payload was fabricated.

## Completeness and leakage status

- Base matrix: 180 expected rows; 0 usable; 180
  `BLOCKED_BASE_ARTIFACT`.
- Scenario matrix: 60 expected rows; 0 materialisable; 60
  `BLOCKED_BASE_ARTIFACT`.
- Binding index: exactly 540 records; 360 source and 180 target bindings.
- Structural scenario leakage: passes all 60 expected scenario shapes.
- Production row/split/label-known/registry/leakage audit: blocked before row
  access because the canonical CSV bytes are absent.
- Deterministic fixture row/split/label-known/registry/leakage audit: pass.

No future prediction-generation run is currently necessary or recommended.
The next action is restoration of one or more exact checksum-locked archives,
followed by the same read-only recovery and no-training audits.

## Validation

- Compileall and Ruff: pass.
- Mypy: pass, 57 source files.
- Full tests: pass, 205 tests.
- Critical coverage: V5 scenarios 81%, canonical recovery 88%, scenario
  leakage 86%, readiness gate 85%, plan/validate runner 58% with the separately
  authorized empirical branch retained.
- Theory/status: pass.
- Notebook audit: pass, 12 notebooks.
- Paper placeholder audit: pass; eight empirical claims remain blocked.
- Synthetic/no-provider CPU smoke: pass.
- Anonymous release: pass, 257 files and package tests.
- Frozen boundary: `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.

## Non-execution statement

No provider model was trained. CoReGraph and baselines were not fit on real
predictions. No target metric or oracle was calculated. The saved-output pilot
was not executed. No official baseline was installed, no dataset was
downloaded, Kaggle was not launched, no paper result was populated, PR #2 was
not merged, and no force-push was used.

This verdict does not authorize pilot execution. It hands the canonical archive
recovery blocker and the corrected V5 readiness surface to a fifth independent
review.
