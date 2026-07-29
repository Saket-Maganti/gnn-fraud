# CoReGraph second-review handoff

Status: local deterministic gates pass; draft-PR CI pending.

Review the scientific repair, not empirical results: no pilot or heavy run was
performed.

## Suggested review order

1. `INDEPENDENT_AUDIT_FINDINGS.csv` and
   `INDEPENDENT_AUDIT_REPAIR_REPORT.md`.
2. Contract V3, split, and graph-semantics regression tests.
3. Availability, router, score-domain, abstention, and composite-objective
   regression tests.
4. Prediction-manifest, pilot-baseline/training, and paired-statistics tests.
5. Theory/support and deterministic synthetic tests.
6. `PILOT_V2_SPECIFICATION.md`, `THEORY_STRENGTHENING_PLAN.md`, and the final
   gate status.

## Review boundaries

- Treat GraphSafe and the graph-feature gate as compatibility adapters already
  present in the repository; no official baseline installation was attempted.
- Treat `MOWST_INSPIRED_REIMPLEMENTATION` only as the explicitly labelled
  reimplementation.
- Do not interpret deterministic pilot-code tests as an empirical pilot.
- Do not promote the finite-sample plan beyond `PROOF_SKETCH_INCOMPLETE`.
- Do not infer execution readiness from this repair verdict.
- Confirm PR #2 is still draft and that its branch advanced only by normal
  fast-forward commits.
