# Generator and formatting audit

## Verdict

The frozen generators reproduce the scientifically correct baseline, but they encode the visual defects that this pass must remove. The dominant failure is architectural: the same scripts that should provide reproducibility also hard-code small fonts, raw-row rendering, wide floats, status colors, and vague provenance. Regenerating the baseline without first hardening these scripts would restore the defects.

## Figure generator

Source: `scripts/tkde_rebuild/make_figures.py`

- Every canvas is 7.16 inches wide, including supplement-only figures.
- Global tick labels are 6.8 pt and explicit diagram/status labels fall to 5.7 pt before LaTeX insertion scaling.
- Configuration identity often depends on color, especially the IBM construction and runtime figures.
- All main figures are inserted as `figure*`; the main paper therefore has seven double-column figure floats.
- PDF creation metadata is not fixed, so byte hashes can change between visually identical runs.
- Figure 1 is largely code-authored. Its flow, status vocabulary, prose, colors, and positioning are not bound to a typed object registry.
- Figure 8 reads the validation CSV but selects only eight of fourteen cases while the caption refers to all fourteen; most visible text is hard-coded.
- The review-budget confidence bands are calculated inside plotting code rather than exported in the figure source CSV.
- The rank-divergence source CSV concatenates two incompatible schemas, producing a sparse 22-column file.
- Provenance rows use generic language that does not identify exact inputs, filters, interval semantics, blocked-state treatment, or hard-coded annotations.

Exact active inputs:

| Figure | Inputs and scientific filter |
| --- | --- |
| F01 | `DEPLOYMENT_CONTRACT_AXES.csv`; flow/status content is code-authored. |
| F02 | `RB09_AUPRC_MAIN.csv`; strict versus isolated AUPRC, paired seeds 1--10. |
| F03 | `IBM_CELL_SUMMARY.csv`; V27 reference and V28 constructions, sender--receiver alias excluded. |
| F04 | `IBM_RANK_DIVERGENCE.csv` and `IBM_METRIC_RANKS.csv`; HI-Medium late-holdout example selected in code. |
| F05 | `IBM_MATCHED_ABLATION_EFFECTS.csv`; AUPRC only, fixed contexts averaged within each of ten seed blocks, alias excluded. |
| F06 | `IBM_RUNTIME_FEASIBILITY.csv`; Pareto computed within variant/protocol, x range is min--max and y error is SD. |
| F07 | `REVIEW_BUDGET_CURVES.csv`; four policies and 0.5/1/2% budgets, six contexts averaged within seed. |
| F08 | `FRAMEWORK_VALIDATION_CASES.csv`; only a selected subset is drawn although all fourteen cases are cited. |

## Table generator

Source: `scripts/tkde_rebuild/build_tables.py`

- All seven main tables use `\scriptsize`; six are unconditional `table*` floats.
- Twenty-two of twenty-three supplementary tables use `\tiny`; the remaining table uses `\scriptsize`.
- A generic longtable helper defaults to small type and accepts unrestricted raw iterables, so it naturally emits CSV dumps.
- Missing, blocked, not applicable, and unavailable values collapse into a generic `--` token instead of typed states.
- Main IBM Table V mixes two scientific questions and repeats categorical winner columns.
- Main resource Table VI includes administrative file counts instead of the concise benchmark treatment requested for the paper.
- Main related-work and model tables are code-authored rather than generated from the verified literature/model inventories.
- Supplement table provenance often records only a bare filename. It does not resolve the path, hash, row filter, seed scope, feasibility set, or claim IDs.

The raw rendered surfaces that must move to the artifact are:

| Baseline table | Rows | Final PDF treatment |
| --- | ---: | --- |
| S05 RB09 seed rows | 180 | Completeness/schema note plus aggregate protocol effects in supplement; all rows in artifact. |
| S09 V22 paired tests | 198 | Hypothesis-family summary and selected aggregate effects in supplement; full table in artifact. |
| S10 IBM seed rows | 840 | Completeness/schema note and aggregate cell tables in supplement; all rows in artifact. |
| S13b context effects | 208 | Direction/heterogeneity summary and selected contexts in supplement; all rows in artifact. |

## LaTeX and float architecture

- Main: 7 `figure*` plus 6 `table*` floats (13 double-column floats); the remaining GraphSafe table is one column.
- Supplement: 5 landscape blocks, 2 forced `\newpage` commands inside landscape groups, and 1 document-level `\clearpage` after the contents.
- The baseline supplement spends most pages 12--18 and 21--38 on narrow raw-row tables centered in a landscape page with large unused margins.
- The frozen main float audit classified placement as passing, but page-order inspection shows a real backlog: IBM RQ3 starts on page 7 while its table first appears on page 9; RQ6 begins on page 8 while the validation figure appears on page 13.
- Ten of fourteen main floats lack an explicit local in-body reference.

## Compilation and audits

- `compile_papers.sh` treats undefined citations/references as fatal but does not fail on overfull boxes, Type 3 fonts, unembedded fonts, duplicate labels/destinations, or visual inventory mismatches.
- `audit_manuscript.py` records several PDF/log findings but its exit status does not enforce them.
- Source discovery uses a hard-coded main-file list instead of the actual LaTeX dependency closure.
- No existing test imports or exercises `scripts/tkde_rebuild/*`; the frozen 941-test result therefore does not validate publication generators.

## Release architecture

- The frozen baseline archives pass CRC, member-hash, exclusion, private-path, credential, and metadata checks.
- `build_release.py` writes to the authoritative baseline filenames. This visual pass must preserve those archives and use new namespaced outputs.
- The current double-build check repeats ZIP creation; it is not an extract/regenerate/compile clean-room test.
- `tkde_source_tables.zip` omits three direct table-generator inputs: `results/runs_rb09v3/runs.csv`, `manuscript_assets/tables/V22_GPU_EVIDENCE_STATUS_TABLE.csv`, and `manuscript_assets/tables/V22_STAT_TESTS_FULL10.csv`. It therefore cannot reproduce every baseline table despite the README claim.

## Required hardening

1. Add one publication-style module for widths, fonts, colors, markers, precision, captions, and blocked-state rendering.
2. Add an object registry that binds every active object to its scientific question, disposition, exact source paths/hashes, filters, claims, final geometry, outputs, and provenance.
3. Replace unrestricted raw table rendering with curated, typed data frames and explicit human/artifact row thresholds.
4. Export every plotted interval/band endpoint to source data.
5. Make all figure identity redundant in grayscale through markers, line styles, direct labels, or patterns.
6. Enforce PDF and LaTeX findings as fatal in strict mode.
7. Build namespaced visual-rebuild archives and perform a true extraction, regeneration, bibliography cycle, PDF audit, and scientific-delta check in a clean temporary directory.
8. Add focused tests for object reconciliation, table readability, figure typography, blocked/missing semantics, scientific scalar preservation, and package completeness.
