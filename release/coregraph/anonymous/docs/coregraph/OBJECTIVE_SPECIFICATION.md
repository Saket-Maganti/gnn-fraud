# Objective specification

The implementation provides binary cross-entropy, focal and class-balanced
classification losses; pairwise ranking; a documented soft top-k surrogate;
source-only robust regret surrogates and V5 matched-action evaluation regret; empirical and
variational CVaR;
compute cost; Brier calibration; and a weighted composite objective.

The V5 headline evaluation oracle acts row by row over feasible experts plus
the same frozen-cost abstention action available to the method. Primary
`contract_regret_vs_feasible_row_oracle` is mean method loss minus this matched
oracle loss and must be nonnegative up to the frozen `-1e-12` tolerance. The
whole-contract best fixed feasible non-abstaining expert remains a separately
named diagnostic and never replaces the primary comparator. Ranking metrics
never select a probability threshold.

Source-contract budgets are converted to review counts inside each group and
then balanced across constrained groups. Unconstrained groups receive no
invented review count; unsupported mixtures of constrained review modes fail
closed. Source-contract abstention capacities are likewise applied within
group. Threshold selection uses balanced source validation only, and a target
capacity can constrain only the already-frozen, label-free target decision.

Cost terms carry a provenance tag (`MEASURED`, `PROFILED`, or
`DRY_RUN_ESTIMATE`). A result cannot be described as latency-constrained when
its cost provenance is only a dry-run estimate.
