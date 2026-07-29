# Test and coverage report

Status: `PASS_LOCAL_DETERMINISTIC_GATES`

- Full repository suite: 176 passed.
- Anonymous release package: 149 CoReGraph tests passed from inside the
  history-free release tree.
- All `coregraph/` statement coverage: 85%.
- Critical-area statement coverage:
  - contracts: 96%;
  - routing: 94%;
  - objectives: 86%;
  - saved-output pilot: 87%;
  - statistics: 94%;
  - SupportEngine: 88%;
  - pilot gate evaluator: 86%;
  - contract splits: 96%.
- Expanded typed-core mypy gate: 50 contract, routing, objective,
  availability, pilot, statistics, support, and theory modules; zero issues.
- Full Ruff gate across `coregraph`, CoReGraph scripts, and tests: zero issues.
- Compileall: pass.
- One-epoch CPU smoke: 96 synthetic examples, one epoch, finite gradients and
  sampled GCN predictions; no provider data.
- Theory numerical/status, 12-notebook validation, ten-scenario synthetic
  method checks, seven-page placeholder paper build/audit, anonymous release,
  sanitized public-tree audit, and the frozen TKDE boundary all pass.

High-risk paths have dedicated regression coverage for structured contract
migration and hashes, access-consistent splits, all availability reason codes,
expert-aware routing and permutation, all-unavailable attention, strict score
domains, functional abstention, differentiable feasible-oracle regret,
environment/seed-bound manifests, honest baselines, source-only pilot fitting,
paired seed statistics, theorem/status identity, SupportEngine proof/scope
semantics, DGraphFin unobserved nodes, directed graph composition, and every
required synthetic qualitative regime. Second-review regressions additionally
cover exact non-0.5 abstention decisions, zero coverage, contract and instance
oracles, group-local budgets and capacities, target-capacity non-leakage,
blocked ranking, honest GraphSafe naming, source-fitted Mowst routing, exact
two-dataset/ten-seed coverage, corrected ablation contribution, Holm verdict
effects, and adversarial matched-contract minima.

No coverage exclusion, threshold reduction, or synthetic fallback was added to
make the gates pass.
