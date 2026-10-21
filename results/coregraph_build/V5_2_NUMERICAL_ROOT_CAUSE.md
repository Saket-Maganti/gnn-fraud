# V5.2 Numerical Root Cause

V5.1 converted aligned expert scores, routing weights, routed scores, and expected compute to float32. The source logistic gate normalized probabilities in float64 and immediately cast normalized weights to float32 before multiplying them by float32 expert scores. Other primary methods and the CoReGraph path also returned or stored float32 arrays.

The float32 cast could move a valid simplex sum above one by one ULP. With a positive binary label and expert scores below one, the over-summed mixture could lie just beyond the feasible convex hull and appear to beat the matched row-wise expert-or-abstain oracle by about `1e-08`.

The scientific oracle, loss, methods, and thresholds were correct. The defect was the finite-precision realization between routing and evaluation.
