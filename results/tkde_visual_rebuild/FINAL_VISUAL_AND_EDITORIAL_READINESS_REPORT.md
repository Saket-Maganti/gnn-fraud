# Final visual and editorial readiness report

## Verdict

**`TKDE_VISUAL_REBUILD_COMPLETE_PROFESSOR_REVIEW_READY`**

The V2 publication-design reconstruction is complete over the frozen `PROFESSOR_REVIEW_READY` scientific baseline. The main paper is concise and decisive, the supplement is detailed and readable, and exhaustive rows/provenance remain in the machine-readable artifact. This verdict is not `TKDE_SUBMISSION_READY` and does not remove any empirical limitation recorded in the frozen baseline.

## Final publication surface

| Surface | Pages | Figure instances | Active tables | Displayed equations | Landscape blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main paper | 14 | 7 | 8 | 6 | 0 |
| Supplement | 30 | 6 | 43 | 19 | 0 |

The eight unique generated figure assets are represented in provenance and used by at least one active PDF. The active LaTeX dependency closure contains no retired raw-dump table, `tiny`, `scriptsize`, `resizebox`, `landscape`, or `sidewaystable` object.

## Exhaustive object disposition

All 72 frozen-baseline objects have exactly one disposition and one destination.

| Disposition | Objects |
| --- | ---: |
| Kept as-is | 25 |
| Kept with a minor edit | 9 |
| Redesigned | 20 |
| Replaced with a different visual form | 3 |
| Split | 6 |
| Moved main to supplement | 1 |
| Moved supplement to artifact | 5 |
| Removed as redundant | 3 |

Destination accounting is 19 main-paper entries, 45 supplement entries, and 8 artifact entries. The strict reconciler reports `objects=72 problems=0`.

## Table and supplement curation

Microscopic CSV-style tables were replaced by aggregate, paired-effect, statistical, dataset-card, model-card, protocol, construction, feasibility, resource, claim, and provenance tables with definitions and surrounding interpretation. The curated supplement uses 43 portrait table fragments in a 20-part reviewer path; exhaustive raw rows remain machine-readable.

| Raw family moved out of the PDF | Rows retained in artifact | Preservation |
| --- | ---: | --- |
| RB09 seed grid | 180 | Complete source rows and SHA-256 retained. |
| V22 paired tests | 198 | Complete source rows and SHA-256 retained. |
| IBM seed grid | 840 | Complete source rows and SHA-256 retained. |
| IBM context effects | 208 | Complete source rows and SHA-256 retained. |

Total raw rows retained outside the human-readable PDFs: **1,426**. No row was deleted or altered. `RAW_TABLE_ARTIFACT_INDEX.csv` records schema, source, count, checksum, and readable replacement for each family.

## Build, test, and audit outcomes

| Gate | Outcome |
| --- | --- |
| Strict main/supplement BibTeX cycles | PASS: 14-page main and 30-page supplement. |
| Object reconciliation | PASS: 72 objects, 0 problems. |
| Table readability | PASS: 51 tables, 0 errors, 0 warnings. |
| PDF/layout audit | PASS: 2 PDFs, 44 pages, 0 errors, 0 warnings. |
| Scientific-delta gate | PASS: 36 frozen rows, 0 errors, 0 warnings; `ZERO_SCIENTIFIC_DELTAS`. |
| Canonical Pytest | PASS: 954 tests; two documented nonfatal pre-existing warnings. |
| Corrected unittest discovery | PASS: 831 tests in 234.384 s. |
| V2 release/audit tests | PASS: 13 tests. |
| Ruff / Compileall | PASS / PASS. |
| Claim language / claim gates / safety | PASS with zero findings or issues. |
| Anonymization | PASS: `double_blind_ok=true`, zero high-severity paper-source findings; manual PDF metadata check found no author identity. |
| Clean-room rebuild | PASS in a newly extracted temporary directory. |
| Archive manifest, hygiene, CRC, and checksum gates | PASS. |

The main PDF embeds all 34 fonts and the supplement embeds all 30 fonts; neither contains a Type 3 font. Final logs contain no overfull box, undefined citation/reference, duplicate label, missing character, or rerun warning. All 50 verified bibliography entries remain cited across the package.

## Print-scale and grayscale QA

Every one of the 44 final pages was rendered at 200 dpi and inspected at full-page, print-scale, and contact-sheet views in color and grayscale. All eight unique figures were also inspected separately at native resolution.

The final pass found no clipped label, tick, rule, caption, or cell; no blank or severely underused landscape page; no stranded float or unresolved backlog; and no raw-row dump. Key distinctions remain intelligible without color through marker shape, fill, line style, direct label, or explicit status text. Missing and resource-blocked cells remain nonnumeric and outside performance axes and rankings.

## Scientific preservation

The final result is **`ZERO_SCIENTIFIC_DELTAS`**:

- all retained scalars trace to the same frozen rows;
- all 22 typed-claim IDs, statuses, and scopes are unchanged;
- uncertainty intervals, effect sizes, multiplicity outcomes, winners, feasibility sets, and resource labels are unchanged;
- no blocked or missing cell became numeric;
- no unmatched feasibility set was pooled;
- all 50 cited references remain verified and used;
- both original release archives retain their exact baseline hashes.

## Remaining scientific limitations

Publication design does not resolve the frozen evidence limits:

- architecture breadth remains narrower than a modern temporal/heterogeneous GNN survey;
- predictive breadth covers two public real graphs and synthetic IBM AML regimes, not a private-bank or cross-institution deployment;
- IBM visibility uses disclosed shared account-history covariates and does not support a pure first-50%-only interpretation;
- ten seeds quantify optimization variation on fixed data/splits, not population variation across institutions or future periods;
- IBM Large, Medium GINE, DGraphFin GAT h64/l2, and the DGraphFin max-pool rerun remain resource-blocked/unmeasured, with no imputed performance direction;
- GraphSafe remains a bounded saved-prediction case: Elliptic favors simple averaging, and no comparison survives correction across the declared 48-test family;
- the support schema and 14 mutation cases validate implementation behavior, not ontology completeness, scientific truth, curator agreement, fairness, or operational safety;
- deterministic saved-output reconstruction is verified, but a clean-machine full-training rerun and an external artifact badge are not claimed.

## Remaining human and professor-review gates

Before submission, humans must still:

1. confirm the current TKDE regular-paper, overlength, and supplementary-material policies for the 14-page main paper;
2. supply and verify author metadata, affiliations, acknowledgments, declarations, and biographies/photos if the current venue workflow requires them;
3. complete professor/coauthor scientific review and decide whether the genuine architecture/domain-breadth limitations are acceptable;
4. perform the final submission-system PDF/preflight and external artifact validation or badge process if pursued;
5. confirm final licensing, privacy, fairness, intended-use, and institutional-governance statements for the intended release context.

## Final deliverables

- Main PDF: `output/pdf/FraudShiftBench_TKDE_main.pdf`
- Supplement PDF: `output/pdf/FraudShiftBench_TKDE_supplement.pdf`
- Curated manuscript package: `release/tkde_visual_rebuild/tkde_visual_manuscript_package.zip`
- Source/analysis and machine-readable evidence package: `release/tkde_visual_rebuild/tkde_visual_source_analysis_package.zip`
- Artifact manifest: `release/tkde_visual_rebuild/tkde_visual_artifact_manifest.csv`
- Excluded-file manifest: `release/tkde_visual_rebuild/tkde_visual_excluded_file_manifest.csv`
- Reproduction README: `release/tkde_visual_rebuild/tkde_visual_reproducibility_readme.md`
- Clean-room report: `release/tkde_visual_rebuild/CLEAN_ROOM_BUILD_REPORT.md`
- Checksum authority: `release/tkde_visual_rebuild/CHECKSUMS.sha256`

The visual/editorial reconstruction is complete and professor-review ready. It is intentionally not presented as evidence that the manuscript has cleared venue policy, author administration, external validation, or human scientific judgment.
