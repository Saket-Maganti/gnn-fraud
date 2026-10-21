# V5.2 Test Report

## Repair-stage status

`PASS`

- Complete repository suite: 353 tests passed.
- Focused V5.2 numerical/executor/package suite: 115 tests passed after final additions.
- Deterministic float32 failure-scale reproduction: pass.
- Adversarial weight, masking, zero-sum, non-finite, and negative-weight tests: pass.
- Hull boundary/projection/substantive-failure tests: pass.
- Unchanged matched-oracle and regret-tolerance tests: pass.
- 170,207-row, three-expert float64 large-array test: pass and deterministic.
- V5.2 synthetic end-to-end smoke: 240/240, zero failures.
- V5.2 package and post-extraction smoke: pass.
- Compileall, Ruff, and mypy: pass.

Mutation-oriented tests reject float32 scientific arrays, missing residual correction, unavailable contributions, substantive hull violations, old schemas/identities, and weakened regret enforcement.
