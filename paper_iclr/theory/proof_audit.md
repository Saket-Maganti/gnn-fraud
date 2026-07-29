# CoReGraph proof audit

| Result | Status | Machine check | Scope and gap |
|---|---|---|---|
| Fixed-mixture lower bound | `PROVED` | Dense grid agrees with the analytic minimax value | Randomised expert selection with linear mixture risk; prediction-space mixtures require separate assumptions |
| Axis-additive compositional bound | `PROVED` | Triangle-inequality fixture passes | Requires observed axis values, uniformly bounded axis estimation error, bounded interaction residual, and declared router optimisation error |
| Feasible-oracle monotonicity | `PROVED` | Subset-mask check passes | Removing feasible experts cannot decrease minimum risk; says nothing about learned-router approximation |

No incomplete result is presented as a theorem. The synthetic generator includes
an interaction regime that violates pure factorisation and confirms why the
residual term is necessary.
