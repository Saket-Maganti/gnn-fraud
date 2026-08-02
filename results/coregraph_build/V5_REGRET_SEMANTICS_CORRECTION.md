# V5 regret-semantics correction

The primary regret comparator was corrected before real pilot execution because the previous method and comparator used unequal feasible action spaces.

V5.0 compared method loss, which could include frozen-cost abstention, with the best feasible non-abstaining expert aggregated over the contract. V5.1 instead uses the row-wise feasible hindsight oracle over feasible expert predictions plus the same abstention action. Primary regret is `contract_regret_vs_feasible_row_oracle`. Under exact arithmetic it is nonnegative; values below `-1e-12` fail closed.

The old comparator survives only as `best_fixed_nonabstaining_expert_brier` and `excess_cost_vs_best_fixed_nonabstaining_expert`. It does not drive the primary gate.

No real target label, metric, oracle, fit, or paper result was viewed before this correction.
