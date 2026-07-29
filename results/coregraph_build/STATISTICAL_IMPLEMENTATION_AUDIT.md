# Statistical implementation audit

Status: `PASS_PRE_RUN`

## Corrections

- Holm adjusted p-values use a monotone cumulative maximum in sorted order.
- Rejection is explicitly step-down: no hypothesis after the first failure can
  be rejected.
- Benjamini-Hochberg uses reverse cumulative minima.
- Bonferroni caps adjusted values at one.
- Empty families are defined; NaN, infinity, out-of-range p-values, and invalid
  alpha fail.
- Holm, BH, and Bonferroni outputs are parity-tested against statsmodels.

## Pairing and uncertainty

- Exact Wilcoxon removes zero differences and uses the exact method.
- Sign tests omit ties from the binomial denominator.
- Paired permutation enumerates all signs through 20 blocks and otherwise uses
  a seeded Monte Carlo design.
- Paired bootstrap resamples seed blocks, not individual examples.
- Contexts are averaged within seed before confirmatory comparisons.
- Non-finite paired values fail before family construction.

## Metrics and calibration

- Average ranks preserve ties; top-K is deterministically broken by typed ID.
- Ranking metrics are prohibited as threshold-selection objectives.
- F1, balanced accuracy, and declared cost risk are the only threshold
  selection rules; review-budget cutoff is separate.
- Calibration includes Brier, fixed/adaptive ECE, binomial logistic
  slope/intercept, temperature, isotonic, and bootstrap intervals.
- Feasible-oracle regret, worst-contract summaries, CVaR, Recall/Precision@K,
  and budget-curve area have deterministic fixtures.

The final family definitions and hashes are frozen in
`ANALYSIS_PLAN_FREEZE.json`. No final-run p-value was computed in this session.
