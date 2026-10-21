# Visual object inventory

This registry covers every active visual, table, display equation, and landscape block in the frozen 14-page main paper and 47-page supplement. Algorithms: 0. The inventory records the baseline object and its required final disposition; final-source reconciliation is performed separately after reconstruction.

## Counts

- Total objects: 72
- equation: 25
- figure: 12
- landscape_block: 5
- longtable: 23
- table: 7

## Dispositions

- `KEEP_AS_IS`: 25
- `KEEP_WITH_MINOR_EDIT`: 9
- `MOVE_MAIN_TO_SUPPLEMENT`: 1
- `MOVE_SUPPLEMENT_TO_ARTIFACT`: 5
- `REDESIGN`: 20
- `REMOVE_REDUNDANT`: 3
- `REPLACE_WITH_DIFFERENT_VISUAL_FORM`: 3
- `SPLIT`: 6

## Object-level decisions

| ID | Type | Document | Page | Label/title | Disposition | Destination | Planned replacement |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| M-F01 | figure | main | 4 | fig:deployment-contract | `REDESIGN` | main | Compact six-coordinate band plus minimal contract-to-evidence-to-claim flow and labeled status key. |
| M-F02 | figure | main | 8 | fig:protocol-effects | `KEEP_WITH_MINOR_EDIT` | main | Retain forest/slope structure with 8 pt typography, redundant markers, and tighter caption. |
| M-F03 | figure | main | 10 | fig:ibm-results | `REPLACE_WITH_DIFFERENT_VISUAL_FORM` | main | Readable two-panel IBM baseline-family AUPRC comparison; construction effects move to the matched multi-metric figure/table. |
| M-F04 | figure | main | 11 | fig:rank-divergence | `KEEP_WITH_MINOR_EDIT` | main | Retain direct rank comparison with grayscale-first styling and repaired source schema. |
| M-F05 | figure | main | 11 | fig:ablation-effects | `REDESIGN` | main | Two-panel multi-metric forest showing AUPRC, AUROC, and F1 deltas with significance and feasibility encoded redundantly. |
| M-F06 | figure | main | 12 | fig:runtime-pareto | `REDESIGN` | main | Grayscale-first four-cell Pareto view with only reference, GINE, Pareto, and blocked states emphasized. |
| M-F07 | figure | main | 13 | fig:claim-validation | `REPLACE_WITH_DIFFERENT_VISUAL_FORM` | main | Data-driven expected-versus-observed status matrix plus compact mutation-family counts. |
| M-T01 | table | main | 4 | tab:related-comparison | `REDESIGN` | main | Compact symbol matrix over temporal, graph-contract, capacity, resource, prediction, and executable-support distinctions. |
| M-T02 | table | main | 6 | tab:datasets | `REDESIGN` | main | Readable task-card summary with unit, temporal extent, prior, graph/features, protocol, and empirical/resource scope. |
| M-T03 | table | main | 6 | tab:models | `MOVE_MAIN_TO_SUPPLEMENT` | supplement | Role-specific model and construction cards in the supplement; replace main float with a protocol/visibility matrix. |
| M-T04 | table | main | 8 | tab:rb09-effects | `REDESIGN` | main | Footnotesize aligned numeric table with the same six paired effects and concise note. |
| M-T05 | table | main | 9 | tab:ibm-results | `SPLIT` | main | Separate IBM baseline AUPRC table and graph-construction matched-effect table. |
| M-T06 | table | main | 9 | tab:resource-boundaries | `REDESIGN` | main | Four-column cell, envelope, short status tag, and benchmark-treatment table. |
| M-T07 | table | main | 10 | tab:graphsafe | `KEEP_WITH_MINOR_EDIT` | main | Footnotesize compact table with shorter caption and explicit comparator boundary. |
| M-E01 | equation | main | 3 | eq:eligibility | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| M-E02 | equation | main | 3 | eq:deployment-contract | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| M-E03 | equation | main | 3 | eq:evidence-unit | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| M-E04 | equation | main | 3 | eq:typed-claim | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| M-E05 | equation | main | 5 | eq:sage | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| M-E06 | equation | main | 6 | eq:cost-risk | `KEEP_AS_IS` | main | Retain with unchanged scientific notation. |
| S-F01 | figure | supplement | 4 | fig:s-contract | `REMOVE_REDUNDANT` | artifact | Use the redesigned main figure once or replace with a text cross-reference. |
| S-F02 | figure | supplement | 11 | fig:s-protocol-effects | `REMOVE_REDUNDANT` | artifact | Use readable aggregate effect tables and refer to the main figure. |
| S-F03 | figure | supplement | 20 | fig:s-ibm-ablation | `REPLACE_WITH_DIFFERENT_VISUAL_FORM` | supplement | Add a compact context-direction/heterogeneity visual if it improves the aggregate tables. |
| S-F04 | figure | supplement | 42 | fig:s-review-budget | `KEEP_WITH_MINOR_EDIT` | supplement | Retain with 8 pt typography and grayscale-redundant line styles/markers. |
| S-F05 | figure | supplement | 43 | fig:s-framework-validation | `REMOVE_REDUNDANT` | artifact | Use the complete controlled-case table; main contains the quantitative status matrix. |
| S-T01 | longtable | supplement | 7 | tab:s-datasets | `SPLIT` | supplement | Readable Elliptic, DGraphFin, and IBM dataset cards. |
| S-T02 | longtable | supplement | 7 | tab:s-protocols | `REDESIGN` | supplement | Explicit label/feature/graph visibility matrix including IBM 50%/60%. |
| S-T03 | longtable | supplement | 9 | tab:s-models | `SPLIT` | supplement | Role-specific baseline, node-GNN, edge-model, and construction cards. |
| S-T04 | longtable | supplement | 9 | tab:s-training | `REDESIGN` | supplement | Readable family-specific training and selection cards. |
| S-T05 | longtable | supplement | 12 | tab:s-rb09-seed | `MOVE_SUPPLEMENT_TO_ARTIFACT` | artifact | Completeness count and schema excerpt; exhaustive 180 rows in artifact. |
| S-T06 | longtable | supplement | 14 | tab:s-rb09-effects | `REDESIGN` | supplement | Readable aggregate effects split by metric/dataset. |
| S-T07 | longtable | supplement | 14 | tab:s-v24-duplicate | `KEEP_WITH_MINOR_EDIT` | supplement | Compact construct-audit table with 120/240 counts. |
| S-T08 | longtable | supplement | 15 | tab:s-v22-lanes | `KEEP_WITH_MINOR_EDIT` | supplement | Readable lane completeness and blocked fixed-GAT row. |
| S-T09 | longtable | supplement | 16 | tab:s-v22-stats | `MOVE_SUPPLEMENT_TO_ARTIFACT` | artifact | Hypothesis-family summary; all 198 rows in artifact. |
| S-T10 | longtable | supplement | 21 | tab:s-ibm-seed | `MOVE_SUPPLEMENT_TO_ARTIFACT` | artifact | Completeness/schema summary; all 840 rows in artifact. |
| S-T11 | longtable | supplement | 32 | tab:s-ibm-cells | `SPLIT` | supplement | Baseline, graph-grid, and scale-focused aggregate tables. |
| S-T12 | longtable | supplement | 33 | tab:s-ranks | `KEEP_WITH_MINOR_EDIT` | supplement | Readable exact-feasibility rank summary. |
| S-T13 | longtable | supplement | 33 | tab:s-ablation | `REDESIGN` | supplement | Multi-metric aggregate table with ten seed blocks and correction status. |
| S-T14 | longtable | supplement | 34 | tab:s-ablation-contexts | `MOVE_SUPPLEMENT_TO_ARTIFACT` | artifact | Directional heterogeneity counts plus representative contexts; all 208 rows in artifact. |
| S-T15 | longtable | supplement | 38 | tab:s-runtime | `REDESIGN` | supplement | Reference/Pareto/blocked resource summary by scale and protocol. |
| S-T16 | longtable | supplement | 40 | tab:s-graphsafe | `KEEP_WITH_MINOR_EDIT` | supplement | Comparator-focused aggregate table. |
| S-T17 | longtable | supplement | 40 | tab:s-graphsafe-tests | `REDESIGN` | supplement | Selected decision metrics plus 48-test family summary. |
| S-T18 | longtable | supplement | 40 | tab:s-review-budget | `REDESIGN` | supplement | Focused 1% table plus curves for 0.5/1/2%. |
| S-T19 | longtable | supplement | 42 | tab:s-resources | `SPLIT` | supplement | Readable case-study cards with evidence/no-evidence and safe wording. |
| S-T20 | longtable | supplement | 5 | tab:s-claims | `SPLIT` | supplement | Five thematic claim tables with concise permitted/prohibited quantifiers. |
| S-T21 | longtable | supplement | 45 | tab:s-families | `MOVE_SUPPLEMENT_TO_ARTIFACT` | artifact | Concise provenance map; exhaustive paths/checksums in artifact. |
| S-T22 | longtable | supplement | 44 | tab:s-framework | `REDESIGN` | supplement | Readable mutation family, violated rule, expected/observed status, pass table. |
| S-T23 | longtable | supplement | 44 | tab:s-false-promotion | `KEEP_WITH_MINOR_EDIT` | supplement | Compact category/count/correct-treatment table. |
| S-E01 | equation | supplement | 3 | Raw label semantics | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E02 | equation | supplement | 3 | Eligibility and binary target | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E03 | equation | supplement | 3 | Scientific cell key | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E04 | equation | supplement | 3 | Deployment contract | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E05 | equation | supplement | 4 | Evidence unit | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E06 | equation | supplement | 4 | Typed claim | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E07 | equation | supplement | 4 | Requirement monotonicity | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E08 | equation | supplement | 7 | GCN update | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E09 | equation | supplement | 7 | GraphSAGE update | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E10 | equation | supplement | 7 | IBM neighborhood summary | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E11 | equation | supplement | 7 | IBM edge score | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E12 | equation | supplement | 7 | GINE update | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E13 | equation | supplement | 10 | Normalized AUPRC diagnostic | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E14 | equation | supplement | 10 | Precision, recall, and F1 | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E15 | equation | supplement | 10 | Illustrative cost risk | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E16 | equation | supplement | 10 | Brier score | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E17 | equation | supplement | 10 | Paired Cohen dz | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E18 | equation | supplement | 39 | GraphSafe reliability risk | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-E19 | equation | supplement | 39 | GraphSafe switching score | `KEEP_AS_IS` | supplement | Retain in the corresponding curated technical section. |
| S-L01 | landscape_block | supplement | 9 | Model and training cards | `REDESIGN` | supplement | Remove the wrapper; use portrait footnotesize curated tables and natural pagination. |
| S-L02 | landscape_block | supplement | 12-18 | Node protocol raw tables | `REDESIGN` | supplement | Remove the wrapper; use portrait footnotesize curated tables and natural pagination. |
| S-L03 | landscape_block | supplement | 21-38 | IBM raw and aggregate tables | `REDESIGN` | supplement | Remove the wrapper; use portrait footnotesize curated tables and natural pagination. |
| S-L04 | landscape_block | supplement | 40-41 | GraphSafe tables | `REDESIGN` | supplement | Remove the wrapper; use portrait footnotesize curated tables and natural pagination. |
| S-L05 | landscape_block | supplement | 44 | Validator tables | `REDESIGN` | supplement | Remove the wrapper; use portrait footnotesize curated tables and natural pagination. |
