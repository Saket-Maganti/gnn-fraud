# Frozen Level-4 statistical analysis plan

Status: `FROZEN_BEFORE_REAL_LEVEL4_RESULTS`. The companion JSON specs and this file are hashed together. Any future amendment must retain the old hash, state the reason, and be labelled post hoc.

## Units and pairing

The primary paired unit is `dataset × held-out target contract × expert-prediction seed`. Router-training randomness is provenance, not an additional independent replication. Same-numbered seeds across datasets are never paired. Primary inference is separate for Elliptic and DGraphFin. A hierarchical dataset-seed bootstrap is secondary and cannot override a per-dataset contradiction.

## Outcomes

The V5 primary outcomes use metric schema `coregraph_v5_metric_schema_v2`: `contract_regret_vs_feasible_row_oracle`, its maximum and CVaR summaries, `global_target_auprc`, `recall_at_frozen_review_fraction` with the exact fraction recorded, selective risk, coverage, routing stability, expert invocation count, warmed inference latency, peak memory, cost, and abstention rate. The primary oracle is row-wise, feasible, hindsight-only, and includes the same abstention action available to the method. The whole-contract `best_fixed_nonabstaining_expert_brier` is a separate diagnostic. BCE surrogate regret is not renamed Brier regret.

## Comparisons

The frozen comparisons are CoReGraph against uniform averaging, best source-validation fixed expert, source-trained logistic and MLP gates, the strongest task-valid feasible graph-MoE implementation, and the strongest task-valid feasible graph-OOD implementation. If an official comparator remains unavailable, its cell is reported with the exact blocker and no internal approximation inherits its name.

## Inference

Within each dataset, matched differences use exact Wilcoxon and paired permutation tests, raw paired effects, robust standardized effects, and seed-block bootstrap 95% intervals. Holm correction applies within each frozen claim family. Two-sided inference is retained while the desirable direction is preregistered. Worst-case improvement is the minimum matched-contract difference in `contract_regret_vs_feasible_row_oracle` within seed; CVaR is formed from that same corrected estimand before pairing.

Missing cells are never imputed. Integrity-invalid cells are excluded with a reason. Resource-blocked cells are not silently dropped from feasibility claims; they are reported and restrict the common comparison set. A comparison needs all ten seeds for every target contract available in the dataset unless a preregistered integrity exclusion applies.

## Pilot decision

GO requires complete integrity and leakage gates, directional improvement over uniform averaging, lower maximum `contract_regret_vs_feasible_row_oracle` on both fraud datasets, no practically concerning primary-metric reversal on either dataset, exact mask behavior, finite selective risk at nonzero useful coverage, effects not driven by one seed, and no oracle or target-label feature. The V5.1 correction and its hash were frozen before any real outcome. No threshold may be chosen after seeing target results.

## Claims

Every paper claim must resolve through `LEVEL4_CLAIM_LEDGER.csv` and `CLAIM_GATE_SPEC.json`. A statistically significant cell alone is insufficient: scope, feasibility, missingness, multiplicity, baseline fidelity, effect direction, and confidence interval must all pass.
