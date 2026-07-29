# Statistical analysis plan

The exact pairing unit is a
dataset-task-target-contract-expert-prediction-seed-fold block. The
expert-prediction seed is the inferential block. A method-specific
router-training seed is deterministic secondary provenance and is never
pseudoreplicated. Primary paired tests are exact Wilcoxon when its assumptions
are usable, sign tests for discrete/tied settings, paired permutation tests,
and expert-seed block bootstrap intervals.

Multiplicity families are declared in
`configs/coregraph/analysis_families.yaml`. Confirmatory families use Holm;
exploratory families use Benjamini-Hochberg and are labelled exploratory.
Bonferroni is available for audit. Non-finite p-values are errors. The Holm
implementation is step-down and stops rejecting after the first failure.

CVaR uses the frozen tail probability. Calibration includes Brier, fixed-bin
ECE, adaptive-bin ECE, temperature scaling, logistic calibration, isotonic
calibration, and paired bootstrap intervals. Missing or resource-blocked cells
are reported, not imputed as wins or losses, and do not enter predictive
ordering. Zero coverage has undefined selective risk plus its declared
abstention cost; it cannot be scored as a perfect win.

Worst-case improvement is formed by subtracting method and baseline inside
each matched target contract and then taking the minimum improvement within
expert-prediction seed. Contract regret is formed first; maximum and CVaR
regret are then aggregated within seed. Full-versus-ablation effects use the
same pairing, declared effect thresholds, confidence intervals and a frozen
Holm correction. Raw p-values alone cannot pass the pilot gate.
