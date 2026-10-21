# Table and Float Placement Audit

## Final documents inspected

- Main paper: `paper_tkde/main.pdf`, 14 letter-size pages, SHA-256 `ded29df1ec98b9ba4a185f325516bed412490b46d3698d72017b4017c5b4d9ad`.
- Supplement: `paper_tkde/supplement/supplement.pdf`, 47 letter-size pages, SHA-256 `1644093f3bd53321254c39bb77e247440a9f30192e11799d31992a4642417706`.
- Every page was rendered with Poppler after the final bibliography cycle and inspected in page order. No blank page, clipping, overlap, margin violation, unreadable caption, or float backlog was found. Both LaTeX logs have zero overfull boxes. Dense seed-level tables are intentionally landscape and remain legible when viewed at normal PDF zoom.

## Main-paper floats

| Float | Purpose | PDF page | Width | First citation or introduction | Readability | Keep in main? |
| --- | --- | ---: | --- | --- | --- | --- |
| Table I | Position the benchmark against the closest literature families | 4 | `table*`, full text width | Related Work, immediately after the comparison discussion | Legible; compact but headings remain distinct | Yes; it carries the novelty argument |
| Fig. 1 | Show the six contract coordinates, evidence unit, support relation, typed claim, and statuses | 4 | `0.96\textwidth` | Deployment Contract, after the formal definitions | Legible labels and status colors; no rasterized text | Yes; it is the conceptual overview |
| Table II | Give exact dataset/task sizes, temporal windows, and prevalence | 6 | `table*`, full text width | Benchmark Instantiation, dataset section | Legible at column-spanning size | Yes; it prevents task and denominator ambiguity |
| Table III | Name the scientific model/construction controls and blocked GINE cell | 6 | `table*`, full text width | Benchmark Instantiation, model section | Legible; internal run IDs are absent | Yes; it anchors model identity |
| Table IV | Report paired strict-to-isolated AUPRC effects with uncertainty and Holm correction | 8 | `table*`, full text width | RQ1/RQ2 opening | Legible; MLP control and six comparisons fit without clipping | Yes; it is the primary controlled result |
| Fig. 2 | Visualize protocol effects and the Elliptic leaderboard reversal | 8 | `0.98\textwidth` | RQ1/RQ2 after Table IV | Legible at two-column width; intervals and slopes are separable | Yes; it makes the ranking change immediate |
| Table V | Report IBM baseline and graph-grid winners within exact feasibility sets | 9 | `table*`, full text width | RQ3 opening | Legible; two panels avoid an over-wide single table | Yes; it contains the principal IBM values |
| Table VI | Keep every blocked/resource-blocked cell visible without a score | 9 | `table*`, full text width | RQ5, immediately after the resource discussion | Legible; status and interpretation columns are clear | Yes; it enforces the evidence boundary |
| Fig. 3 | Compare IBM AUPRC, AUROC, and fixed-threshold F1 across scale and construction | 10 | `0.99\textwidth` | Table V caption and RQ3 text | Legible; six panels use common visual grammar | Yes; it carries the scale/metric interaction |
| Table VII | Give the bounded GraphSafe decision summary over seed blocks | 10 | single-column `table` | Worked GraphSafe case | Legible in one column | Yes; small enough to retain and important as a negative case |
| Fig. 4 | Show dependence-aware matched IBM ablation effects | 11 | `0.98\textwidth` | RQ3 matched-control discussion | Legible; titles state ten seed blocks and four fixed contexts | Yes; it exposes the repaired uncertainty analysis |
| Fig. 5 | Show AUPRC/F1 rank divergence and one exact IBM reversal | 11 | `0.98\textwidth` | RQ4 opening | Legible labels and rank paths | Yes; it separates ranking from one operating point |
| Fig. 6 | Plot AUPRC/runtime tradeoffs and resource-blocked GINE cells | 12 | `0.98\textwidth` | RQ5 opening | Legible; log runtime axis and blocked annotation are clear | Yes; it links performance to the resource contract |
| Fig. 7 | Show observed claim-status transitions under evidence mutations | 13 | `0.98\textwidth` | RQ6 validator discussion | Legible and unclipped; statuses remain distinguishable | Yes; it demonstrates executable claim discipline |

The main paper contains no orphaned float. All seven tables and seven figures occur on or shortly after the page where their scientific role is introduced. The reference list begins on page 13 and completes on page 14 without creating a blank tail page.

## Supplementary figure floats

| Float | Purpose | PDF page | Width | First citation or introduction | Readability | Keep in main? |
| --- | --- | ---: | --- | --- | --- | --- |
| Fig. 1 | Repeat the contract diagram beside full definitions and propositions | 4 | `0.92\textwidth` | Supplement Section II | Legible | No; detailed formal context belongs in the supplement |
| Fig. 2 | Repeat the paired real-data visibility effects beside complete rows | 11 | `0.88\textwidth` | Supplement Section VI | Legible | No; main Fig. 2 already carries the concise version |
| Fig. 3 | Repeat the seed-blocked IBM ablation plot beside full sensitivity tables | 20 | `0.90\textwidth` | Supplement Section VII-C | Legible; an otherwise sparse page is justified by the plot's full-page scale | No; main Fig. 4 is sufficient |
| Fig. 4 | Give precision/recall curves at fixed review capacities | 42 | `0.92\textwidth` | Supplement Section VIII-C | Legible with distinct uncertainty bands | No; this is a bounded operational diagnostic |
| Fig. 5 | Repeat validator mutations beside complete support tables | 43 | `0.90\textwidth` | Supplement Section IX-B | Legible | No; the main contains the concise validator figure |

## Supplementary longtables

The supplement uses `longtable` rather than floating tables so complete evidence cannot drift away from its section. All 23 generated tables were inspected.

| Printed table | Generated source | Purpose | Pages | Width/orientation | Readability | Main-paper decision |
| --- | --- | --- | --- | --- | --- | --- |
| I | `table_s19_claim_ledger.tex` | Complete typed claim ledger | 5 | portrait, full text width | Legible at tiny type | Supplement only |
| II | `table_s01_dataset_cards.tex` | Dataset cards | 7 | portrait, full width | Legible | Main Table II is the compact replacement |
| III | `table_s02_protocols.tex` | Full protocol contracts | 7 | portrait, full width | Legible | Supplement only |
| IV | `table_s03_models.tex` | Full model inventory | 9 | landscape, full width | Legible | Main Table III is the compact replacement |
| V | `table_s04_training.tex` | Optimization and selection settings | 9 | landscape, full width | Legible; three harmless underfull word breaks only | Supplement only |
| VI | `table_s05_rb09_seed.tex` | All 180 RB09 seed rows | 12--14 | landscape, repeating header | Dense but legible and unclipped | Supplement only |
| VII | `table_s06_rb09_effects.tex` | All paired RB09 effects | 14 | landscape, full width | Legible | Main Table IV carries AUPRC subset |
| VIII | `table_s07_v24_duplicate_audit.tex` | V24 construct-duplication audit | 14--15 | landscape, repeating header | Legible | Supplement only |
| IX | `table_s08_v22_lanes.tex` | Canonical V22 lane completeness | 15 | landscape, full width | Legible | Supplement only |
| X | `table_s09_v22_stats.tex` | Complete V22 paired statistics | 16--18 | landscape, repeating header | Dense but legible | Supplement only |
| XI | `table_s10_ibm_seed.tex` | All 840 IBM seed rows | 21--32 | landscape, repeating header | Dense but legible; no truncation | Supplement only |
| XII | `table_s11_ibm_cells.tex` | All IBM cell aggregates and intervals | 32--33 | landscape, repeating header | Legible | Main Table V is the compact replacement |
| XIII | `table_s12_ranks.tex` | Exact rank/decision disagreement sets | 33 | landscape, full width | Legible | Main Fig. 5 shows one example |
| XIV | `table_s13_ablation.tex` | Seed-blocked IBM construction effects | 33--34 | landscape, repeating header | Legible; raw 40 and inferential 10 are both explicit | Main Fig. 4 carries AUPRC subset |
| XV | `table_s13b_ablation_contexts.tex` | All 208 context-specific IBM sensitivities | 34--37 | landscape, repeating header | Dense but legible; no clipped columns | Supplement only; needed for dependence audit |
| XVI | `table_s14_runtime.tex` | Runtime, Pareto, and blocked status | 38 | landscape, full width | Legible | Main Fig. 6/Table VI are compact replacements |
| XVII | `table_s15_graphsafe.tex` | GraphSafe aggregate summary | 40 | landscape, full width | Legible | Main Table VII is the compact replacement |
| XVIII | `table_s16_graphsafe_tests.tex` | All GraphSafe paired tests | 40 | landscape, full width | Legible | Supplement only |
| XIX | `table_s17_review_budget.tex` | Complete review-budget rows | 40--41 | landscape, repeating header | Legible | Supplement only |
| XX | `table_s18_resources.tex` | Full resource cases | 42 | portrait, full width | Legible | Main Table VI is the compact replacement |
| XXI | `table_s21_framework_validation.tex` | All 14 validator cases | 44 | portrait, full width | Legible | Main Fig. 7 summarizes status transitions |
| XXII | `table_s22_false_promotion.tex` | False-promotion audit | 44 | portrait, full width | Legible | Supplement only |
| XXIII | `table_s20_evidence_families.tex` | Run-family, eligibility, and lock map | 45 | portrait, full width | Legible | Supplement only |

## Verdict

**PASS.** Float placement is publication-quality for professor review. The main paper is dense but readable; the supplement is long because it renders complete seed- and context-level evidence, not because floats generated empty pages. Any later source edit requires a fresh compile, render, and page-order inspection.
