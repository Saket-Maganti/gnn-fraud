# CoReGraph method card

Inputs are declared probability, logit, or rank-score tensors, expert identity
and optional family embeddings, a factorised deployment-contract encoding,
shared and per-expert label-free diagnostics, feasibility masks, and per-expert
costs. Outputs are an explicitly typed blended score, expert weights,
selection, abstention probability, routing entropy, expected compute, and
explanations. Probability and logit mixtures are never interchanged
implicitly; rank scores cannot enter calibration or entropy computations.

The saved-output pilot predeclares each single expert, feasible averaging,
source-validation best expert, a source-validation convex mixture, the actual
GraphSafe V2 compatibility implementation, the current graph-feature gate,
learned atomic- and no-contract routers, a clearly labelled
`MOWST_INSPIRED_REIMPLEMENTATION`, and an offline feasible-oracle ceiling.
CoReGraph may be called robust only when held-out-combination, worst-contract,
CVaR, calibration, and budget results are complete.

Fallback is explicit. The default feature-only-safe strategy chooses a feasible
feature expert; alternatives are uniform-feasible or abstain. No graph expert
may receive positive mass when its graph, edge-feature, memory, or licence
requirements are unmet. An all-unavailable row bypasses attention, assigns
zero expert weight, selects expert `-1`, emits the declared blend sentinel,
and forces abstention to one.
