# Test and coverage report

Status: `PASS_LOCAL_GATES`

- Pytest: 94 passed.
- Combined frozen FraudShiftBench/TKDE regression suite and new CoReGraph
  tests: pass.
- Statement coverage for `coregraph/`: 77% (2,680 of 3,484 statements).
- Typed-core mypy gate: 23 contract, routing, objective, and evidence modules;
  zero issues.
- Ruff undefined/import gate across `coregraph`, CoReGraph scripts, and tests:
  zero issues.
- Compileall: pass.
- One-epoch CPU smoke: feature expert, factorised router gradient, and sampled
  GCN predictions pass.
- Anonymous package: import smoke and all 67 packaged CoReGraph tests pass from
  inside the history-free release tree.
- Theory numerical/status, notebook, paper skeleton, anonymous release, and
  frozen TKDE gates pass.

High-risk modules have dedicated mutation or sentinel coverage: contracts,
evidence/support, graph views/leakage, task IDs/labels, multiplicity,
calibration, top-K ties, objectives/regret/CVaR, masks/fallback/abstention,
saved-output routing, sampling/memory guards, resume/import hashes, baseline
pins/parity schema, synthetic regimes, theory, and anonymity.

The machine-readable coverage detail is `COVERAGE.json`. Low-coverage paths are
primarily provider-file branches, pending official subprocess execution, and
blocked resource/error branches that cannot be exercised without the expressly
forbidden external downloads or heavy runs.
