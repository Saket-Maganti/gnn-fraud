# CoReGraph method card

Inputs are declared probability, logit, or rank-score tensors, expert identity
and optional family embeddings, a factorised deployment-contract encoding,
shared and per-expert label-free diagnostics, feasibility masks, and per-expert
costs. Outputs are an explicitly typed blended score, expert weights,
selection, abstention probability, routing entropy, expected compute, and
explanations. Probability and logit mixtures are never interchanged
implicitly; rank scores cannot enter calibration or entropy computations.

The saved-output pilot predeclares each single expert, feasible averaging,
source-validation best expert, a source-validation convex mixture, the current
graph-feature gate, learned atomic- and no-contract routers, and the
source-trained `MOWST_INSPIRED_REIMPLEMENTATION` as complete deployable
comparators. The partial `graphsafe_confidence_abstention_component` is a
compatibility component, not a full GraphSafe comparator or a member of the
headline Holm family. Headline Brier contract regret uses one contract-level
feasible oracle expert for the whole contract. The instance-clairvoyant oracle
is an offline diagnostic ceiling excluded from significance and deployable
methods. The BCE quantity used to fit the robust objective is named
`bce_surrogate_contract_regret`; selective deployment error is
`selective_zero_one_risk`. Values from these three loss definitions are never
compared as though they shared a scale.
CoReGraph may be called robust only when held-out-combination, worst-contract,
CVaR, calibration, and budget results are complete.

Fallback is explicit. The default feature-only-safe strategy chooses a feasible
feature expert; alternatives are uniform-feasible or abstain. No graph expert
may receive positive mass when its graph, edge-feature, memory, or licence
requirements are unmet. An all-unavailable row bypasses attention, assigns
zero expert weight, selects expert `-1`, emits the declared blend sentinel,
and forces abstention to one.

The frozen source-validation abstention threshold and decision are preserved
exactly. Source-contract abstention capacities govern fitting; target
operational capacity can constrain only the final label-free decision.
Fallback produces an executable prediction from another allowed expert,
whereas abstention produces no accepted prediction. Every method-contract row
declares its execution state, and unavailable scores never enter ranking
metrics. Expert-prediction seed is the inferential block within each dataset;
same-numbered seeds from different datasets are never paired. Router-training
seed is derived deterministically per method or ablation and is never counted
as a replication.
