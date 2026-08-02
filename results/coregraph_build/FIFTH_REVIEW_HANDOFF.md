# Fifth-Review Handoff

Verdict: `COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES`

## Handoff identity

- Branch: `codex/coregraph-iclr-buildout-2026`
- Pull request: draft PR #2, open and unmerged
- Final SHA: recorded in the PR body and terminal handoff after the final
  fast-forward push (a tracked file cannot contain its own commit SHA)
- CI: exact-tip hosted check rollup recorded in the PR body after completion
- Next authority: fifth independent review

## What is ready for review

The engineering correction is complete. V5 separates 180 role-neutral base
artifact cells from 60 evaluation scenarios and 540 role bindings. Scenario
roles, operational target contract, and row scopes are explicit; cross-scenario
reuse is valid; same-scenario source/target reuse is atomic. The plan/validate
runner and readiness gate pass the end-to-end no-training fixture and cannot
authorize or execute the pilot.

The old V4 360-role-cell matrix is superseded audit history and must not be used
for scientific completeness or readiness decisions.

## Canonical recovery evidence

Canonical scientific identity comes from:

1. RB09v3 `ARTIFACT_FAMILY.json` and `predictions_manifest.json`, asserting
   exactly 180 prediction CSVs;
2. RB09v3 `runs.csv`, containing exactly 180 matching result records;
3. RB15/RB16 import manifests, preserving all 180 archive-member references,
   six archive paths and six archive SHA-256 values;
4. searched V22/V24 evidence locks, merged prediction/result indexes, JSONL
   sidecars, package import validations, source traces, archive/alias records,
   and raw fallback candidates.

The exhaustive scan inspected 421 authoritative files, 1,498 structured
prediction-index records, and 5,372 structured result-index/JSONL records.
There are 201 raw same-coordinate CSV candidates, but none is an evidenced
RB09v3-compatible alias.

## Current blocker

- Canonical base artifacts expected: 180
- Locally usable canonical artifacts: 0
- `INDEX_REFERENCED_FILE_MISSING`: 180
- Source archives expected: 6
- Source archives locally present: 0
- Materialisable scenarios: 0 of 60
- Expected role bindings: 540
- True never-created/integrity-failed artifacts: 0

All 180 missing references are Category D. Exact archive paths, members, and
checksums are in `CANONICAL_RB09V3_MISSING_INDEX_REFERENCES.csv` and
`FUTURE_RUN_NECESSITY_REPORT.md`. A future prediction-generation run is not
necessary or recommended while Category D remains non-zero and Category E is
zero.

## Unresolved provenance

Canonical result records recover configuration/command provenance and measured
per-run runtime for all 180 cells. All 180 code identities remain typed
`UNRESOLVED_LEGACY_CODE`; all 180 routing-cost proxies remain `UNRESOLVED`;
and all 180 base hashes, row audits, and exact target operational contract
bindings remain blocked until the canonical bytes are restored and verified.
No metadata was invented.

## Validation

- Compileall, Ruff, and mypy over 57 source files: pass.
- Full test suite: 205 passed.
- Critical coverage: V5 scenarios 81%, canonical recovery 88%, leakage 86%,
  readiness gate 85%, runner plan/validate file 58%.
- Theory/status, 12 notebooks, paper placeholders, ten deterministic synthetic
  checks, and no-provider CPU smoke: pass.
- Anonymous 257-file release and package tests: pass.
- Frozen FraudShiftBench/TKDE boundary:
  `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.
- Committed public-tree audit and exact-tip hosted CI: reported in PR #2 after
  the final push.

## Required next review action

The fifth reviewer should first obtain one or more of the six exact canonical
archives from their original storage. Each archive must match its recorded
SHA-256 before the converter streams or extracts members. Then rerun only the
V5 canonical recovery plus no-training completeness, split, label-known,
registry, and scenario-leakage audits.

Even a fully passing rerun would mean only that the recovered manifests may be
considered for a separately authorized pilot. It would not itself authorize
pilot execution.

## Explicit non-execution statement

No model was trained. CoReGraph and baselines were not fit on real predictions.
No target metric or oracle was calculated. The saved-output pilot was not run.
No official baseline was installed, no dataset was downloaded, Kaggle was not
launched, and no paper result was populated. The 249 frozen assets were not
modified. PR #2 was not merged and no force-push was used.
