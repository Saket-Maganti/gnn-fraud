# CoReGraph third-review handoff

Status: `LOCAL_DETERMINISTIC_GATES_PASS_DRAFT_PR_CI_PENDING`.

Review the pilot V3 semantics and frozen gate, not empirical performance. No
real prediction was connected and no pilot or multi-seed experiment ran.

## Review order

1. `SECOND_REVIEW_FINDINGS.csv` and
   `SECOND_REVIEW_REPAIR_REPORT.md`.
2. `tests/coregraph/test_second_review_pilot_semantics.py`.
3. `tests/coregraph/test_second_review_gate_semantics.py`.
4. `PILOT_V3_SPECIFICATION.md` and
   `PILOT_GATE_FROZEN_SPEC.json`.
5. `configs/coregraph/analysis_families.yaml`, the method/objective/statistical
   documentation, and the paper oracle/experiment placeholders.
6. `FINAL_GATE_STATUS.json`, the anonymous audit, and the frozen-boundary
   result.

## Required third-review questions

- Does every target metric consume the stored abstention decision without
  target-label refitting?
- Is `contract_feasible_oracle` the only headline regret reference, with
  `instance_clairvoyant_oracle_ceiling` diagnostic only?
- Are source budgets/capacities group-local, and is target capacity absent
  from source fitting?
- Are blocked sentinel predictions excluded from rankings while fallback
  remains an executable state?
- Are the GraphSafe and Mowst-inspired claims faithful to their implemented
  scope?
- Does the frozen gate fail on incomplete cells, zero coverage, ineffective
  ablations, uncorrected/too-small effects, unmatched minima, unstable routing,
  or target-label selection?

## Stop boundary

Do not authorize the pilot from this handoff alone. The next phase is
conversion of real provider artifacts into manifests followed by a dry-run
completeness audit and independent review. Official baseline installation,
data download, pilot execution, Kaggle, multi-seed campaigns, PR merge, and
empirical claim population remain out of scope.
