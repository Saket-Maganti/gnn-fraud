# Statistical analysis plan

The exact pairing unit is a
dataset-task-target-protocol-and-contract-expert-prediction-seed-fold block.
The expert-prediction seed is the inferential block within a dataset.
Elliptic seed 1 and DGraphFin seed 1 are unrelated blocks and are never
averaged into a matched pair. Each dataset receives separate exact Wilcoxon,
paired permutation, and seed-block bootstrap results. A hierarchical
dataset-then-seed bootstrap is secondary combined evidence only. A
method-specific router-training seed is deterministic secondary provenance
and is never pseudoreplicated.

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

The robust gate requires a positive effect on both datasets, corrected support
on at least one dataset, and no contradictory dataset-level effect.
Worst-case improvement is formed by subtracting method and baseline inside
each matched target contract and then taking the minimum improvement within
expert-prediction seed. Contract regret is formed first; maximum and CVaR
regret are then aggregated within seed. Full-versus-ablation effects use the
same pairing, declared effect thresholds, confidence intervals and a frozen
Holm correction. Raw p-values alone cannot pass the pilot gate.

Risk names are loss-specific. `bce_surrogate_contract_regret` is used for
training, `brier_contract_regret` is the frozen headline evaluation risk, and
`selective_zero_one_risk` is thresholded deployment error on accepted rows.
No generic `contract_regret` result label may mix these quantities.
