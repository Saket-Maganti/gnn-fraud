# Third-Review Repair Plan

Status: `COMPLETE`

Governing specification: `COREGRAPH_THIRD_REVIEW_INTEGRATION_AND_MANIFEST_CONVERSION.md`.

## Scope

1. Add failing regressions for TR-01 through TR-08 where practical.
2. Introduce a V4 prediction manifest and frozen protocol registry that keep protocol aliases, coordinate hashes, and full contract IDs distinct.
3. Enforce explicit row-scope and `label_known` semantics, with per-artifact exclusion reports.
4. Integrate typed cross-role leakage checks with the existing contract-split/leakage infrastructure.
5. correct dataset-stratified inference, risk terminology, and comparator taxonomy before any historical conversion.
6. Add a deterministic end-to-end `--validate-only` runner-to-gate fixture that performs no fitting.
7. Build a read-only V4 converter; discover historical artifacts without modifying them or guessing metadata.
8. Run only authorized completeness, split, label-known, contract-registry, leakage, and no-training validations.
9. Run all deterministic validation gates, verify the 249 frozen assets byte-for-byte, update the handoff/status artifacts, commit focused changes, and push normally to the existing branch.

## Explicit exclusions

- No saved-output pilot execution.
- No CoReGraph or learned-baseline fitting on real predictions.
- No target metric or target-oracle computation.
- No official-baseline installation.
- No dataset download or Kaggle launch.
- No paper-result population.
- No edits to the 249 frozen FraudShiftBench/TKDE assets.
- No PR merge and no force-push.

## Terminal verdict

The final verdict will use exactly one governing-specification token. Any readiness token means only that a separate fourth independent review may consider pilot authorization.
