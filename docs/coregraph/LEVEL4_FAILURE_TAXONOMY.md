# Level-4 failure taxonomy

Every future evaluated row receives zero or more labels from this frozen set. Labels explain failure; they do not excuse missing cells.

| Code | Meaning | Detection evidence |
|---|---|---|
| `WRONG_EXPERT_SELECTED` | a feasible lower-risk whole-contract expert exists | offline contract risks after evaluation unlock |
| `ALL_EXPERTS_POOR` | all feasible experts exceed the preregistered source-justified risk boundary | expert risk panel |
| `OVERCONFIDENT_DIAGNOSTICS` | diagnostic confidence is high while its source-validated error proxy fails | calibration/diagnostic audit |
| `ROUTING_INSTABILITY` | small admissible perturbation causes excessive policy change | stability suite |
| `CORRELATED_SHIFT` | graph and feature shift co-occur beyond isolated mechanisms | contract/diagnostic record |
| `LATENT_CONTRACT_AMBIGUITY` | distinct mechanisms produce indistinguishable inferred factors | controlled identifiability check |
| `BUDGET_COLLAPSE` | useful ranking vanishes at a tighter review budget | budget frontier |
| `RESOURCE_MASK_COLLAPSE` | no expert remains feasible or mass violates a mask | mask invariant audit |
| `ABSTENTION_COLLAPSE` | zero coverage or near-universal acceptance despite high declared uncertainty | risk-coverage audit |
| `SOURCE_OVERFITTING` | source validation improves while held-out source-contract validation degrades | nested source validation |
| `BASELINE_UNFAIRNESS` | access, expert set, tuning, or resource envelope differs | parity manifest |
| `METRIC_DISAGREEMENT` | conclusions reverse across declared primary outcomes | metric comparison |

The machine-readable labels are generated in `results/coregraph_build/LEVEL4_FAILURE_LABELS.json`.
