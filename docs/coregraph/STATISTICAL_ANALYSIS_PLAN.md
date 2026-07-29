# Statistical analysis plan

The unit of pairing is a dataset-task-contract-seed block. Aggregation to a
confirmatory test occurs within seed context before comparisons across
contexts. Primary paired tests are exact Wilcoxon when its assumptions are
usable, sign tests for discrete/tied settings, paired permutation tests, and
paired bootstrap intervals. Report paired mean/median differences and rank
biserial or common-language effects.

Multiplicity families are declared in
`configs/coregraph/analysis_families.yaml`. Confirmatory families use Holm;
exploratory families use Benjamini-Hochberg and are labelled exploratory.
Bonferroni is available for audit. Non-finite p-values are errors. The Holm
implementation is step-down and stops rejecting after the first failure.

CVaR uses the frozen tail probability. Calibration includes Brier, fixed-bin
ECE, adaptive-bin ECE, temperature scaling, logistic calibration, isotonic
calibration, and paired bootstrap intervals. Missing or resource-blocked cells
are reported, not imputed as wins or losses.
