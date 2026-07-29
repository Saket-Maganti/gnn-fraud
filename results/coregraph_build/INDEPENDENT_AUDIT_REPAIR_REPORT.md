# CoReGraph independent-audit repair report

Local deterministic verdict:
`REPAIRS_COMPLETE_DRAFT_PR_CI_PENDING`.

This pass repairs the scientific-core defects enumerated in
`COREGRAPH_INDEPENDENT_AUDIT_METHOD_AND_PILOT_REPAIR.md`. It does not run or
claim results for the saved-output pilot.

## Repaired scientific surfaces

- DeploymentContract V3 preserves the six top-level coordinates while making
  visibility, construction, budget, and resource properties compositional. A
  one-way V2 migration, coordinate hash, and complete artifact/environment
  hash are implemented.
- Contract splits now make target access regime and selection policy agree,
  derive manifest access from target contracts, and reject atomic-ID leakage.
- Expert availability enforces explicit blocks, device, memory, latency, task,
  graph, edge features, full-graph guards, licence, and integration state with
  structured reason codes.
- CoReRouter tokens include expert and optional family identity, shared and
  per-expert diagnostics, cost, availability, score, and contract context.
  All-unavailable attention rows have an exact finite sentinel path.
- Score domains are explicit (`PROBABILITY`, `LOGIT`, `RANK_SCORE`);
  incompatible loss, calibration, entropy, and mixture operations fail closed.
- Abstention now has selective risk, coverage, AURC, cost, capacity,
  source-validation threshold selection, forced no-expert abstention, and
  ablations.
- Prediction artifacts are environment- and seed-bound, checksum-checked, and
  aligned exactly. Pilot baselines are distinct implementations rather than
  misleading aliases.
- The saved-output training path uses explicit source groups, source
  train/validation separation, the differentiable composite objective, real
  resource masks and costs, stability perturbations, abstention, and
  source-only early stopping.
- Inference uses paired seed blocks, exact Wilcoxon, paired permutation,
  seed-block bootstrap, frozen Holm families, and paired worst-contract
  outcomes.
- The theorem, code, numerical checker, proof audit, and status formula now
  agree on
  `2*sum(axis_errors) + 2*interaction_residual + router_error`. The deeper
  finite-sample result remains honestly labelled `PROOF_SKETCH_INCOMPLETE`.
- SupportEngine requires verified, hash-bound proof artifacts; ignores
  irrelevant resource blocks; and distinguishes exact, narrower, wider, and
  incompatible scopes.
- DGraphFin isolated nodes retain an explicit unobserved state and are excluded
  from temporal node-task masks. Graph direction composes independently with
  history, degree, and edge-feature policies.
- Deterministic synthetic checks now cover every qualitative regime required
  by the audit.

The issue-by-issue reproducer, consequence, repair, and status are recorded in
`INDEPENDENT_AUDIT_FINDINGS.csv`.

## Validation posture

All local deterministic gates pass: 158 repository tests, 131 anonymous-package
tests, 85% total core coverage, and at least 85% in each mandated critical
area. Compileall, full Ruff, the 50-module mypy surface, theory, synthetic
method checks, one-epoch CPU smoke, notebooks, paper build/audit, sanitized
public-tree audit, anonymous release, and the 249-file frozen boundary pass.
Exact results are recorded in `INDEPENDENT_AUDIT_COMMAND_LOG.md`,
`TEST_COVERAGE_REPORT.md`, and `FINAL_GATE_STATUS.json`. The preferred final
verdict remains gated on the draft PR CI run.

## Explicit non-execution statement

This pass did not install official baselines, download datasets, connect real
saved predictions, execute the pilot, launch Kaggle, or run multi-seed
experiments. It does not establish empirical superiority, official-baseline
parity, provider-data readiness, or execution readiness. The 249 frozen
FraudShiftBench/TKDE assets remain outside the repair boundary.
