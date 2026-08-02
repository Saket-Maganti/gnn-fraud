# V5.2 Numerical Repair Specification

V5.2 introduces the authoritative `normalize_feasible_weights_float64` and `routed_scores_in_feasible_hull_float64` layer.

- Convert routing weights and expert scores to float64.
- Reject non-finite and materially negative weights; canonicalize only negative roundoff within `1e-12`.
- Set unavailable weights to exactly zero.
- Normalize feasible positive rows in float64.
- Add `1 - sum(normalized)` to the largest feasible normalized weight deterministically.
- Force abstention for zero-feasible-weight rows.
- Recompute routed scores from corrected float64 weights.
- Check and, only within `1e-12`, project routed scores into the feasible expert-score hull.
- Compute Brier loss, matched oracle, raw regret, aggregates, and gate inputs in float64.
- Fail if raw regret is below `-1e-12`; preserve raw negative diagnostics and canonicalize only tolerated negative regret for aggregation.
- Store scores, weights, and expected compute as float64.

Implementation version: `coregraph_v5_2_float64_simplex_v1`.

Output schema: `coregraph_v5_output_schema_v3`.

Metric schema remains `coregraph_v5_metric_schema_v2`.
