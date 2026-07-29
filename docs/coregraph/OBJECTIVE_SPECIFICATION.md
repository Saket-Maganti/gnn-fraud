# Objective specification

The implementation provides binary cross-entropy, focal and class-balanced
classification losses; pairwise ranking; a documented soft top-k surrogate;
oracle-normalised worst-contract regret; empirical and variational CVaR;
compute cost; Brier calibration; and a weighted composite objective.

The feasible oracle is computed only over experts allowed by the contract and
resource mask. Ranking metrics never select a probability threshold. Threshold
selection is limited to classification or cost objectives on validation data;
budget cutoffs are handled separately.

Cost terms carry a provenance tag (`MEASURED`, `PROFILED`, or
`DRY_RUN_ESTIMATE`). A result cannot be described as latency-constrained when
its cost provenance is only a dry-run estimate.
