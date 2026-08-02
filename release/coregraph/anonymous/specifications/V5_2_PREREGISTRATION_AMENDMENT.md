# CoReGraph Saved-Output Pilot V5.2 Preregistration Amendment

## Status and timing

This amendment was frozen before any V5.2 real coordinate was executed. It supersedes V5.1 only for executable numerical realization and preserves the complete V5.1 failure provenance.

## V5.1 provenance

The V5.1 campaign was aborted before a gate outcome because a float32 implementation violated a mathematically required convex-mixture invariant by approximately 1.68e-08. V5.2 changes only numerical realization, dtype, invariant enforcement, output schema, and audit diagnostics. The method set, datasets, seeds, scenarios, oracle, abstention cost, metrics, and decision thresholds remain unchanged. No V5.1 coordinate is reused.

The V5.1 stop is classified as `NUMERICAL_IMPLEMENTATION_INVARIANT_FAILURE`, `NOT_A_SCIENTIFIC_NO_GO`, and `NO_FROZEN_GATE_OUTCOME`.

## Frozen V5.2 numerical realization

- Scientific routing, routed scores, feasible hulls, Brier losses, oracle losses, regret, aggregation, and gate inputs use IEEE-754 float64.
- Unavailable expert weights are set to exactly zero before routing.
- Feasible nonzero rows are normalized in float64. The residual `1 - sum(weights)` is added deterministically to the largest feasible normalized weight.
- Tiny negative input weights down to `-1e-12` may be canonicalized to zero. More-negative or non-finite weights fail closed.
- A row with no feasible positive weight is forced to abstain.
- Routed scores are checked against the feasible expert-score hull. Projection is allowed only for numerical violation at most `1e-12`; larger violations fail closed.
- Primary row regret is computed from the unchanged matched feasible expert-or-abstain oracle. Raw regret below `-1e-12` fails closed. Raw values in `[-1e-12, 0)` may be canonicalized to zero for aggregation while remaining visible in diagnostics.
- Target scores, routing weights, and expected compute are stored as float64.

## Unchanged scientific contract

- Primary methods: `coregraph`, `uniform_average`, `best_fixed_expert`, `source_logistic_gate`.
- Experts: `feature_mlp`, `gcn`, `graphsage`.
- Datasets, protocols, provider seeds, scenario definitions, bindings, source sampling, review fraction, and resource profiles are unchanged.
- Predictive method loss remains Brier loss; abstaining method loss remains the frozen abstention cost `0.20`.
- The primary oracle remains the row-wise minimum over every feasible expert Brier loss and the identical abstention action.
- The primary metric remains `contract_regret_vs_feasible_row_oracle` under `coregraph_v5_metric_schema_v2`.
- Gate comparisons and thresholds are unchanged.

## Identity and reuse

V5.2 uses a new preregistration hash, implementation version, output schema, effective execution identity, repository SHA, and empty real/synthetic roots. V5.1 checkpoints, coordinates, packages, and partial outputs are invalid for V5.2 resume or packaging.
