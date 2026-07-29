# Object reconciliation report

Verdict: **PASS**

## Frozen-baseline registry

The registry contains 72 unique baseline objects:

- 12 figures;
- 7 main tables;
- 23 supplement longtables;
- 25 displayed equations;
- 5 landscape blocks;
- 0 algorithms.

Every object has one final disposition and destination. Dispositions are: 25 `KEEP_AS_IS`, 9 `KEEP_WITH_MINOR_EDIT`, 20 `REDESIGN`, 6 `SPLIT`, 3 `REPLACE_WITH_DIFFERENT_VISUAL_FORM`, 1 `MOVE_MAIN_TO_SUPPLEMENT`, 5 `MOVE_SUPPLEMENT_TO_ARTIFACT`, and 3 `REMOVE_REDUNDANT`. Destination accounting is 19 main, 45 supplement, and 8 artifact entries.

## Final active surface

- Main: 7 figure instances, 8 table fragments, 6 displayed equations, and no landscape block.
- Supplement: 6 figure instances, 43 portrait curated table fragments, 19 displayed equations, and no landscape block.
- Unique generated figures: F01--F08, all represented in `FIGURE_DATA_PROVENANCE.csv` and referenced by at least one active PDF.
- Active tables: exactly the 8 rows of `MAIN_TABLE_DATA_PROVENANCE.csv` plus the 43 rows of `CURATED_SUPPLEMENT_TABLE_MANIFEST.csv`.
- Active LaTeX closure contains no legacy `table_s*.tex`, `tiny`, `scriptsize`, `resizebox`, `landscape`, or `sidewaystable` object.

The strict reconciler reports `objects=72 problems=0`. The scientific-delta gate independently checks 72 allocation rows, 8 figure provenance rows, 51 table provenance rows, 22 typed claims, and 50 cited references.

## Raw rows moved to the artifact

| Family | Rows | Frozen source | SHA-256 |
| --- | ---: | --- | --- |
| RB09 seed grid | 180 | `results/runs_rb09v3/runs.csv` | `d9f77bfd14ccaf157780858e35ecfee96af1e2fbf60f3559ad888d14c0ace2e9` |
| V22 paired tests | 198 | `manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv` | `513f1fb5c2dbae2f63255fd384c62fe86394a15746c31792a92a1f287b8ddadd` |
| IBM seed grid | 840 | `results/tkde_rebuild/IBM_IMPORTED_SEED_ROWS.csv` | `dae5d51178b9a746946c9fc05bf31e74840b6663d69031597ef2be45d5e766b6` |
| IBM context effects | 208 | `results/tkde_rebuild/IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv` | `626f0e014a0b8d339fe344bb736d513ed63a6ec435be6104acad1f53f7bbe202` |

Their complete schemas, counts, checksums, and replacements are recorded in `RAW_TABLE_ARTIFACT_INDEX.csv` and `RAW_TABLE_ARTIFACT_INDEX.md`. No source row was deleted or altered.

## Preservation result

- Frozen-hash rows checked: 36.
- Baseline archives preserved: 2 of 2 at their original hashes.
- Claim ledger: 22 of 22 IDs and status counts unchanged.
- References: 50 verified and 50 cited.
- Blocked-semantics CSVs checked: 41.
- Scientific result: `ZERO_SCIENTIFIC_DELTAS`.
