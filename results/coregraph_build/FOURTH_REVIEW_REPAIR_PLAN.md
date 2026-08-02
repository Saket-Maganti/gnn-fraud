# Fourth-Review Repair Plan

Status: `COMPLETE_WITH_CANONICAL_ARCHIVE_RECOVERY_BLOCKER`

Governing specification: `COREGRAPH_FOURTH_REVIEW_CANONICAL_EVIDENCE_RECOVERY.md`.

## Scope

1. Preserve the 249-file frozen FraudShiftBench/TKDE boundary and keep the historical repository read-only.
2. Establish and document canonical evidence precedence before assigning missing-artifact status.
3. Add failing regression tests where practical for FR-01 through FR-08.
4. Replace V4 role-specific prediction artifacts with role-neutral base artifacts and explicit evaluation-scenario bindings.
5. Scope row materialization, contract binding, and leakage audits to a single held-out-target-protocol scenario.
6. Recover config, code, routing-cost, measured-compute, lock, import, result, archive, and alias provenance without guessing or fabricating values.
7. Reconcile the canonical RB09v3 claim of 180 prediction artifacts and publish the authoritative artifact and missing-reference indexes.
8. Generate exactly 180 base-artifact completeness rows, 60 scenario rows, and 540 scenario-binding records.
9. Exercise the corrected path end to end with a deterministic no-training fixture that cannot compute target metrics or invoke fitting.
10. Run only the authorized completeness, split, label-known, contract-registry, leakage, and no-training historical audits.
11. Run the governing deterministic validation suite, update the V5 readiness specification and handoff, commit focused changes, and push normally to the existing branch.

## Evidence precedence

The implementation will resolve conflicts in this order:

1. final evidence lock;
2. final merged prediction index;
3. per-lane merge validation;
4. package import validation;
5. result sidecar;
6. raw filename/content navigation.

Filename inference will be a fallback only. Higher-precedence dataset, protocol, model, seed, path, and checksum fields will override it, and unresolved conflicts will remain blocked.

## Explicit exclusions

- No model training.
- No CoReGraph or baseline fitting on real predictions.
- No target metric or oracle calculation.
- No saved-output pilot execution.
- No official-baseline installation.
- No dataset download or Kaggle launch.
- No edits to the 249 frozen FraudShiftBench/TKDE assets.
- No paper-result population.
- No merge of PR #2 and no force-push.

## Terminal condition

The final verdict will use a token permitted by the governing specification. Any readiness result will mean only that the converted manifests may be considered for pilot execution by a separate fifth independent review.

## Completion

FR-01 through FR-08 are implemented and validated. The canonical reconciliation
accounts for all 180 RB09v3 coordinates, 60 expected scenarios, and 540
role bindings. Production row-scope materialisation cannot proceed because all
180 indexed members resolve to six checksum-locked source archives that are
absent from both inspected repositories. They are Category D recovery
dependencies, not evidence that a prediction run never occurred. No future GPU
run is recommended while those archives remain recoverable.
