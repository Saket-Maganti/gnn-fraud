# Objective specification

The implementation provides binary cross-entropy, focal and class-balanced
classification losses; pairwise ranking; a documented soft top-k surrogate;
contract-level feasible-oracle worst-contract regret; empirical and
variational CVaR;
compute cost; Brier calibration; and a weighted composite objective.

The headline contract-level feasible oracle first aggregates each expert's
loss over a whole contract, then chooses one expert executable for that whole
contract. The instance-clairvoyant oracle may be reported only as a diagnostic
ceiling and never supplies the training or headline regret loss. Ranking
metrics never select a probability threshold.

Source-contract budgets are converted to review counts inside each group and
then balanced across constrained groups. Unconstrained groups receive no
invented review count; unsupported mixtures of constrained review modes fail
closed. Source-contract abstention capacities are likewise applied within
group. Threshold selection uses balanced source validation only, and a target
capacity can constrain only the already-frozen, label-free target decision.

Cost terms carry a provenance tag (`MEASURED`, `PROFILED`, or
`DRY_RUN_ESTIMATE`). A result cannot be described as latency-constrained when
its cost provenance is only a dry-run estimate.
