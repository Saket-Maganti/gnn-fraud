# Theory readiness report

Status: `THEORY_READY_WITH_STATED_ASSUMPTIONS`

- Fixed-mixture lower bound: `PROVED`.
- Axis-additive compositional bound: `PROVED`, with implementation and
  numerical audit both using
  `2*sum(axis_errors) + 2*interaction_residual + router_error`.
- Resource-mask feasible-oracle monotonicity: `PROVED`.
- Numerical checks: deterministic script implemented; all checks must pass in CI.

The first result concerns randomised selection risk. It does not automatically
apply to nonlinear averaging of calibrated predictions. The second result
requires observed axis values and includes an interaction residual; an XOR
counterexample shows why the residual is necessary. A near-tight adversarial
case approaches the declared bound and a just-above-bound case is rejected.
Neither result is evidence of empirical fraud performance.
