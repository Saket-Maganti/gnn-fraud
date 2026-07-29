# CoReGraph proof audit

| Result | Status | Machine check | Scope and gap |
|---|---|---|---|
| Fixed-mixture lower bound | `PROVED` | Dense grid agrees with the analytic minimax value | Randomised expert selection with linear mixture risk; prediction-space mixtures require separate assumptions |
| Axis-additive compositional bound | `PROVED` | Near-tight and above-bound adversarial cases check \(2\sum_j\epsilon_j+2\epsilon_{\mathrm{int}}+\epsilon_{\mathrm{router}}\) | Requires observed axis values, uniformly bounded axis estimation error, bounded interaction residual, and declared router optimisation error |
| Contract-level feasible-oracle monotonicity | `PROVED` | Subset-mask check passes on whole-contract expert risks | Removing feasible experts cannot decrease the minimum whole-contract expert risk; the result does not concern an instance-clairvoyant oracle or learned-router approximation |

No incomplete result is presented as a theorem. The synthetic generator includes
an XOR/high-interaction regime with zero marginal axis errors. Pure
factorisation fails there; the declared \(2\epsilon_{\mathrm{int}}\) term covers
the scoped case.
