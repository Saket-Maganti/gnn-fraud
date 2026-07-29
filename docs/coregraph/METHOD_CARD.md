# CoReGraph method card

Inputs are expert scores or embeddings, a factorised deployment-contract
encoding, optional label-free diagnostics, feasibility masks, and cost
metadata. Outputs are a score, expert weights, selection, abstention
probability, routing entropy, expected compute, and explanations.

Required comparisons are the best single feasible expert, equal mixture,
GraphSafe feature-only fallback, current FraudShiftBench gate, atomic-contract
router, and no-contract router. CoReGraph may be called robust only when
held-out-combination, worst-contract, CVaR, calibration, and budget results are
complete.

Fallback is explicit. The default feature-only-safe strategy chooses a feasible
feature expert; alternatives are uniform-feasible or abstain. No graph expert
may receive positive mass when its graph, edge-feature, memory, or licence
requirements are unmet.
