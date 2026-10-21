# Final TKDE Readiness Report

## 1. Final title

**FraudShiftBench: Deployment-Contract Evaluation for Temporal Graph Fraud Detection**

## 2. Final thesis

A fraud-model score is not scientifically interpretable from a model name, dataset name, and split alone. It is a conditional claim under a deployment contract that records temporal order, graph visibility, graph construction, model selection, decision policy, and resource envelope. The validated evidence shows that changing those coordinates can reverse rankings, change metric winners, or make a comparison unmeasurable. FraudShiftBench couples the contract to a typed support relation so incomplete, construct-invalid, prediction-missing, contradicted, and resource-blocked claims cannot silently enter a leaderboard.

## 3. Final contribution hierarchy

1. **Deployment-contract formalism and typed support.** Six separately recorded, non-substitutable coordinates define the evaluated contract. Evidence units, typed claims, support conditions, and five propositions make scope widening and missing support explicit.
2. **Conditional empirical map.** Ten-seed evidence on Elliptic, DGraphFin, and four measured IBM AML-Data Small/Medium regimes separates visibility, architecture, graph construction, metric, prevalence, scale, and runtime effects. It does not pool different prediction units into a global score.
3. **Executable claim discipline and artifact.** Fourteen claim/evidence mutations, false-promotion tests, evidence locks, prediction manifests, scalar provenance, resource statuses, and deterministic builders connect manuscript statements to validated artifacts. GraphSafe is retained only as a bounded decision-level case with negative comparisons, not as a universally validated method.

## 4. Main paper size

- Final PDF: `paper_tkde/main.pdf`.
- Pages: **14**.
- Approximate active-section source words: **6,800**.
- PDF-extracted words, including tables and references: **11,265**.
- Layout: two-column IEEE journal format, zero overfull boxes, zero undefined citations/references, and zero Type 3 fonts.

## 5. Supplement size

- Final PDF: `paper_tkde/supplement/supplement.pdf`.
- Pages: **47**.
- Content: full notation/propositions, dataset and protocol cards, model/training definitions, 180 RB09 rows, canonical V22 robustness, all 840 IBM seed rows, dependence-aware aggregate and 208 context-specific IBM ablation rows, GraphSafe decision tables, resource cases, validator fixtures, provenance, reproduction, limitations, and ethics.

## 6. Verified references

- Verified bibliography entries: **50**.
- Distinct entries cited in the main paper: **49**.
- Distinct entries cited in the supplement: **26**.
- Distinct entries cited anywhere: **50**.
- Missing keys, unused verified entries, duplicate labels, and undefined references: **0**.
- Verification surfaces: `CITATION_VERIFICATION.md`, `LITERATURE_MATRIX.csv`, and `CITATION_COVERAGE_AUDIT.md`.

## 7. Major figures and tables regenerated

- **8** evidence-derived vector figure assets, each in PDF and PNG with a source-data CSV and provenance row.
- **7** main-paper tables.
- **23** supplementary longtables.
- Principal visuals cover the deployment contract, real-data protocol effects, IBM metric/scale/construction effects, rank-versus-decision divergence, dependence-aware matched ablations, runtime/resource Pareto cases, review budgets, and validator mutations.
- `TABLE_AND_FLOAT_PLACEMENT_AUDIT.md` records every main float, every supplementary figure, and every supplementary longtable with final page placement and readability.

## 8. New CPU-only analyses performed

- Exact dataset/task/prevalence reconstruction from locally staged data and locked result metadata.
- Paired strict-to-isolated visibility effects with deterministic percentile intervals, paired effect sizes, Wilcoxon tests, and Holm correction.
- IBM metric-rank divergence within exact feasibility sets.
- Dependence-aware IBM construction analysis: four fixed contexts averaged within seed before inference, giving ten seed blocks; all **208** context-specific effects remain visible.
- Runtime/Pareto analysis restricted to matched variant--protocol cells.
- Review-budget, calibration, and fixed-cost decision diagnostics from saved predictions.
- GraphSafe comparisons corrected to ten seed blocks rather than 60 context rows; no one of 48 Holm-adjusted tests is significant.
- V24 construct-duplication audit, false-promotion audit, 14-case support-relation mutation suite, and deterministic before/after hash validation.
- No full model training or new GPU experiment was performed; these analyses consume validated saved artifacts.

## 9. Exact empirical evidence used

- **RB09 real-data visibility grid:** 180 result rows and 180 prediction files across Elliptic and DGraphFin, three models, three protocols, and ten seeds. The main paired claim uses strict versus isolated visibility with shared seeds.
- **Canonical V22 robustness:** five passing full-10 lanes with 640 results and 640 prediction references. A DGraphFin GAT h64/l2 lane remains blocked; failed/noncomparable imports are excluded.
- **V24 rerun:** 120 scientific dataset--protocol--model--seed cells. The nominal 360 rows contain 240 metadata-label duplicates because the stress label never reached the harness; no temporal-stress conclusion uses them.
- **IBM AML-Data V26--V28:** 840 result JSON files and 840 prediction exports: 240 baseline-grid rows, 80 h64-reference rows, and 520 construction/feature-ablation rows. Every measured configuration group has ten seeds.
- **GraphSafe case:** locked RB17 saved-output analyses with six contexts averaged within each seed for inference. It is reported as a bounded decision example with negative as well as favorable results.
- Evidence scope and locks are enumerated in `EVIDENCE_INVENTORY.csv`; **6,796** scalar manuscript/analysis values map through `NUMBER_PROVENANCE_MAP.csv`.

## 10. Blocked and resource-blocked evidence retained

The following cells remain visible and receive no predictive score or rank:

1. IBM AML HI-Large: safe resource block, 0 results / 0 predictions.
2. IBM AML LI-Large: safe resource block, 0 / 0.
3. IBM AML HI-Medium GINE h64: single-T4 CUDA OOM, 0 of 20 planned / 0 predictions.
4. IBM AML LI-Medium GINE h64: single-T4 CUDA OOM, 0 of 20 planned / 0 predictions.
5. DGraphFin GAT h64/l2: T4 CUDA OOM, 0 of 20 planned / 0 predictions; the h32/l1 diagnostic is not a replacement.
6. DGraphFin GraphSAGE max-pool rerun: awaiting a larger GPU, 0 / 0.

Failed V22 imports, V24 duplicate labels, hardware aliases, partial diagnostics, and the sender--receiver construction alias are separately recorded and are never promoted to independent performance evidence.

## 11. Tests, compilation, and audits

| Gate | Final status |
| --- | --- |
| Deterministic rebuild | PASS: 247 evidence rows, 22 claims, 6,796 scalar records, 8 figures, 7+23 tables, 50 references |
| Support validator | PASS: 14/14 expected status transitions |
| Regeneration audit | PASS: 28 generated-analysis hashes and 8 canonical-input hashes unchanged |
| `pytest` | PASS: 941 tests collected, 100% complete, exit 0 |
| Corrected CI `unittest` discovery | PASS: 831 tests, exit 0 |
| Ruff | PASS, exit 0 |
| Compileall | PASS, exit 0 |
| Claim gate / heavy-default safety | PASS / PASS |
| Claim-language audit | PASS: zero findings |
| Main full BibTeX cycle | PASS: 14 pages, zero undefined citations/refs or overfull boxes |
| Supplement full BibTeX cycle | PASS: 47 pages, zero undefined citations/refs or overfull boxes |
| Citation coverage | PASS: 50/50 verified works cited |
| Font audit | PASS: all fonts embedded/subset as applicable; zero Type 3 fonts |
| Visual audit | PASS: all 61 PDF pages rendered and inspected after the final source change |

The exact commands, one lint repair, one CI collection repair, and the two remaining nonfatal dependency/statistics warnings are recorded in `COMMAND_LOG.md`.

## 12. Unresolved risks

1. **Venue administration:** the main is 14 pages. Current IEEE/TKDE base-page, maximum-page, supplementary-material, and mandatory-overlength-charge rules must be confirmed at submission time. A shorter 12-page version would require scientific prioritization, not silent deletion.
2. **Architecture breadth:** the primary real-data grid contains MLP, GCN, and GraphSAGE rather than a broad modern temporal/heterogeneous GNN survey. The graph-ML reviewer remains weak reject on this genuine evidence limit.
3. **Domain breadth:** predictive evidence covers two public real graphs and synthetic IBM AML-Data regimes, not a private-bank deployment, new institution, jurisdiction, or laundering typology.
4. **IBM visibility:** early-to-late labels use the first 50%, while the shared label-free account-history matrix uses the first 60%. This transductive covariate visibility is disclosed and prevents a pure first-50%-only interpretation.
5. **Inference scope:** ten seeds quantify optimization variation on fixed datasets/splits. They do not estimate population variation across banks or future periods. Several deterministic library baselines repeat exactly across nominal seeds.
6. **Blocked cells:** IBM Large, Medium GINE, DGraphFin GAT h64/l2, and the max-pool rerun remain unmeasured. No performance direction follows from possible future hardware success.
7. **GraphSafe:** descriptive DGraphFin outcomes are favorable, Elliptic favors simple averaging, and no comparison survives correction across 48 tests. The case is not a validated universal method.
8. **Artifact scope:** deterministic reconstruction from saved outputs is verified; a clean-machine, full-training rerun and an external artifact badge are not claimed.
9. **External validation:** the support schema and 14 mutations were authored inside this project. They validate implementation behavior, not ontology completeness, scientific truth, curator agreement, fairness, or responsible deployment.
10. **Human review:** professor/coauthor review, author metadata, venue declarations, and final submission-system checks remain outside this rebuild.

Reviewer outcomes after repair are: benchmark minor revision / weak accept; graph ML major revision / weak reject on breadth; AML minor revision / weak accept within public-data scope; statistics/artifact minor revision / weak accept after a passing archive preflight.

## 13. Submission verdict

**`PROFESSOR_REVIEW_READY`**

There is no known fabricated, falsely widened, prediction-missing, resource-promoted, citation-invalid, or mechanically fatal claim in the rebuilt manuscript. The analysis, PDFs, audit surfaces, and curated release archives are internally synchronized. The deterministic double-build, member-hash, ZIP-integrity, exclusion, private-path, and credential checks pass; `release/tkde_artifact_manifest.csv` is the checksum authority. The package is not declared TKDE-submission-ready because architecture/domain breadth, venue-format confirmation, external validation, and human scientific review remain material.
