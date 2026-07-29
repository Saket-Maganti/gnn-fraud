# Theory readiness report

Status: `THEORY_READY_WITH_STATED_ASSUMPTIONS`

- Fixed-mixture lower bound: `PROVED`.
- Axis-additive compositional bound: `PROVED`.
- Resource-mask feasible-oracle monotonicity: `PROVED`.
- Numerical checks: deterministic script implemented; all checks must pass in CI.

The first result concerns randomised selection risk. It does not automatically
apply to nonlinear averaging of calibrated predictions. The second result
requires observed axis values and includes an interaction residual; an XOR
counterexample shows why the residual is necessary. Neither result is evidence
of empirical fraud performance.
