# Current manuscript forensic audit

Audit date: 2026-07-10 IST  
Audited package: `paper_tkde/` before the complete rebuild  
Baseline main PDF: 12 pages, approximately 5,354 extracted words  
Baseline supplement PDF: 1 page, approximately 305 extracted words

## Executive finding

The current package is an evidence-organizing draft, not a submission-shaped journal paper. Its strongest feature is disciplined scope control. Its central weakness is that this discipline is expressed as repository bookkeeping rather than as a scientific framework supported by literature, formal definitions, experimental design, and quantitative analysis. The main PDF also overstates its effective length: large floats dominate pages 7--10, and page 12 contains only a small table near the top of an otherwise empty page. The supplement is a pointer sheet rather than supplementary material.

The rebuild will preserve the evidence boundaries but replace the title, thesis hierarchy, paper architecture, formalism, narrative, bibliography, quantitative surfaces, figures, tables, and supplement.

## Issue-by-issue audit

| ID | Severity | Evidence in the starting package | Scientific or editorial risk | Repair strategy | Status before rewrite |
| --- | --- | --- | --- | --- | --- |
| F01 | Fatal | No `\\cite{...}` command appears in `paper_tkde/main.tex` or any included `paper_tkde/sections/*.tex`. | Factual and novelty claims are unsupported; the paper cannot be reviewed as scholarship. | Build a verified literature matrix, insert claim-local citations, run BibTeX, and audit undefined/unused entries. | OPEN |
| F02 | Fatal | `paper_tkde/references.bib` contains only four entries, and none is used. | The paper omits the fraud, graph anomaly, temporal benchmark, leakage, benchmark governance, selective prediction, and operational screening literatures. | Replace the bibliography with verified primary-source metadata and use every retained entry for a specific argument. | OPEN |
| F03 | Major | `paper_tkde/sections/03_related_work.tex:1-14` is six short topical paragraphs plus an external citation-TODO pointer. | The paper cannot establish a precise novelty gap or defend its contribution against adjacent benchmarks. | Rewrite related work analytically across five literature clusters and add a comparison table. | OPEN |
| F04 | Fatal | Submission-meta language appears at `paper_tkde/main.tex:20` ("Draft for Review"), `paper_tkde/sections/02_introduction.tex:6` ("deliberately TKDE-first"), `paper_tkde/sections/03_related_work.tex:14` (citation TODO pointer), and `paper_tkde/sections/09_artifact_reproducibility.tex:18` ("this TKDE draft"). | The text reads as internal production commentary and signals incompleteness. | Remove venue targeting, drafting commentary, TODO pointers, and repository-production language from scientific prose. | OPEN |
| F05 | Fatal | `paper_tkde/sections/04_problem_protocols.tex:3` defines a binary supervised problem with `y\\in\\{0,1,2\\}` and mixes the raw Elliptic unknown code with supervised labels. | The task definition is mathematically wrong and can invalidate downstream notation. | Separate raw dataset labels from the supervised binary target; define an eligibility mask and binary label only on eligible instances. | OPEN |
| F06 | Major | Internal labels V26/V27/V28 and RB09/RB15--RB17 dominate `01_abstract.tex:2`, `02_introduction.tex:23`, `06_models_metrics_diagnostics.tex:5-9`, and `07_experimental_evidence.tex:5-23`. | Engineering chronology obscures the scientific experiment families and makes the paper hard to compare with prior work. | Use scientific family names in the main paper; move version/run identifiers to a supplement provenance map. | OPEN |
| F07 | Major | `paper_tkde/sections/01_abstract.tex:2` foregrounds 840 JSON files and 840 prediction exports, then lists blocked lanes and artifact components. | The abstract reports package administration instead of the problem, method, empirical findings, and implications. | Rewrite from scratch with the scientific problem, formal deployment/evidence contracts, evaluated scope, quantitative findings, and implications. | OPEN |
| F08 | Fatal | `paper_tkde/sections/05_benchmark_design.tex:1-22` defines the benchmark mainly through cards, locks, manifests, and exclusions. | The contribution is vulnerable to the criticism that it is a well-organized repository rather than a technical benchmark methodology. | Formalize a multi-axis deployment contract, typed evidence unit, typed empirical claim, support relation, and status semantics; validate the framework with mutation/ablation tests. | OPEN |
| F09 | Fatal | `paper_tkde/sections/05_benchmark_design.tex:21-38` explicitly calls ClaimGate "intentionally boring" and presents a linear sequence of checks. | A trivial validator cannot carry the paper's central novelty. | Make the validator an implementation of the formal support relation, demote it from primary novelty, and evaluate correctness properties and failure cases. | OPEN |
| F10 | Fatal | No complete dataset table or methods section reports graph sizes, feature schemas, split windows, label mappings, class priors, access constraints, model equations, optimizer, learning rate, stopping, thresholds, or selection rules. | Results are not interpretable or reproducible; dataset/task mismatches can be hidden. | Recover exact values from official papers, loaders, run JSON, manifests, and configs; put compact essentials in the main paper and full cards/configs in the supplement. | OPEN |
| F11 | Major | `paper_tkde/sections/08_results.tex:17-40` emphasizes winner labels and aggregate means; the main text does not report paired uncertainty, effect sizes, corrected tests, or exact sample sizes for most findings. | Rankings can be artifacts of seed variation, unequal feasible cells, thresholds, or prevalence. | Recompute matched-cell deltas, confidence intervals, paired tests, effect sizes, correction status, and rank disagreement; keep unequal GINE coverage separate. | OPEN |
| F12 | Major | Mechanical formulations such as "Supported:", "Not supported:", and "Boundary:" recur throughout `paper_tkde/sections/08_results.tex:5-72` and figure captions. | The paper reads like an audit checklist rather than a scholarly argument. | Integrate scope into prose and retain a compact claim-boundary table only where it aids interpretation. | OPEN |
| F13 | Major | Starting PDF page 4 shows the protocol taxonomy as a branching hierarchy, although time, visibility, construction, selection, budget, and resources require separate specification and are not substitutes for one another. | The visual model is conceptually misleading. | Replace it with a coordinate-based contract and evidence-to-claim flow. | OPEN |
| F14 | Major | Starting PDF page 7 uses a two-column protocol heatmap across most of a full-width float. | The figure spends substantial page area on little information and hides uncertainty. | Replace with paired effect-size/forest plots and confidence intervals. | OPEN |
| F15 | Major | Starting PDF page 8 uses categorical winner cells; magnitude and uncertainty are absent. | Readers cannot judge practical relevance or stability. | Plot actual metric values and intervals, and add matched rank-versus-decision comparisons. | OPEN |
| F16 | Major | Starting PDF page 10 shows two nearly overlapping review-budget curves and two narrow calibration bars. | These figures add little comparative evidence and do not support main-paper space. | Use budget lift/difference summaries where meaningful; move diagnostic calibration material to the supplement. | OPEN |
| F17 | Major | Starting PDF page 11 includes an administrative supported/blocked bar chart. | Administrative counts are not a primary empirical result. | Replace with a claim-scope mutation case study or move the status inventory to the supplement. | OPEN |
| F18 | Major | Starting PDF page 9's runtime scatter does not make variant, protocol, construction, or blocked resource cells legible. | Runtime differences can be misread as general hardware or deployability claims. | Build a matched runtime-performance view with explicit protocol/scale encodings and a separate blocked-feasibility table. | OPEN |
| F19 | Major | Starting PDF pages 7--10 are float-heavy; page 12 contains a small table and is otherwise almost empty. | The nominal 12 pages overstate substantive density and violate the requested float discipline. | Redesign assets for column width, place them near first discussion, and audit every final page and float. | OPEN |
| F20 | Fatal | `paper_tkde/supplement/supplement.tex` compiles to one page and repeatedly says "See ... CSV" instead of rendering the material. | Reviewers cannot inspect methods, seed-level data, tests, proofs, or reproducibility details in the submission package. | Rebuild an independent substantive supplement with rendered dataset cards, configurations, full tables, tests, proofs, provenance, and commands. | OPEN |
| F21 | Fatal | The current manuscript contains no direct, cited comparison with GADBench, GAD in the Wild, TGB/TGB 2.0, BenchTemp, BAG, BetterBench, BenchmarkCards, Eval Factsheets, or core graph-fraud methods. | The novelty may duplicate or lag current work, especially 2024--2026 benchmark papers. | Search through the current date, verify primary sources, and state a narrower differentiation that survives comparison. | OPEN |
| F22 | Major | `paper_tkde/sections/02_introduction.tex:18-24` lists five partially overlapping contributions. | The contribution hierarchy is diffuse and includes packaging details as peer contributions. | Select one primary identity and no more than three secondary contributions after the evidence/literature audit. | OPEN |
| F23 | Major | `paper_tkde/sections/06_models_metrics_diagnostics.tex:23` and `07_experimental_evidence.tex:15-18` retain GraphSafe-TTA without a complete policy definition or corrected statistical context in the main text. | A mixed, bounded case study may be mistaken for a universal method claim. | Keep only a compact comparator-aware case study in the main paper if it reinforces the deployment-contract thesis; move full policy and tests to the supplement. | OPEN |
| F24 | Major | The paper does not distinguish dataset task units rigorously: Elliptic/DGraphFin are node/entity classification, while IBM AML is transaction/edge classification. | Cross-dataset aggregates can become scientifically invalid. | Put prediction unit in the formal evidence type, dataset table, every cross-family result, and claim scope. Do not pool unlike units as one leaderboard. | OPEN |
| F25 | Major | Hardware language relies on "T4-class" in places while legacy source paths contain P100 identifiers. | Run names can be mistaken for verified hardware evidence. | Report only hardware supported by runtime/environment records; identify legacy aliases as names, not hardware facts. | OPEN |
| F26 | Fatal | The V24 RB41 runner loops over `late_window_holdout`, `early_to_late_transfer`, and `rolling_window_stress`, but `run_one_config()` passes none of those labels to the benchmark harness. For every one of 120 dataset/protocol/model/seed base cells, all performance metrics are exactly identical across the three labels; only runtime differs. | Treating the 360 files as three temporal stress conditions would triple-count one scientific design and fabricate a temporal-regime effect from metadata. The evidence lock validates artifact integrity and counts, not this construct-validity defect. | Exclude stress-label contrasts, retain at most one deduplicated strict/isolated rerun grid as supplementary robustness evidence, record the 240 duplicate-label rows in false-promotion tests, and base the primary protocol analysis on RB09v3. | OPEN |
| F27 | Major | `ibm_aml_graph_builder.py` materializes one node-history feature matrix from the late-window protocol's first 60% of transactions and reuses it for both IBM protocols. The early-to-late classifier uses labels from the first 50%, so its node features include label-free covariates from 50--60%, which belongs to that protocol's validation interval. The saved leakage note says only that node history uses "train-period transactions." | Calling both IBM feature maps protocol-specific training-only history would be inaccurate, and early-versus-late differences cannot be interpreted without this visibility asymmetry. It is not test-label leakage, but it is an explicit transductive-covariate choice. | Define the early-to-late contract as 50% labeled training with a shared 60% label-free node-history map; state the resulting construct limitation in methods and threats; avoid causal attribution of protocol differences to time windows alone. | OPEN |

## Baseline visual inspection

All 12 starting main-paper pages and the one supplement page were rendered to PNG before rewriting.

- Pages 1--3 are dense text with underdeveloped citations and formalism.
- Pages 4--6 mix large administrative tables/diagrams with sparse explanatory text.
- Pages 7--10 are dominated by low-information full-width figures.
- Page 11 mixes limitations, a resource table, an administrative bar chart, and the conclusion.
- Page 12 is almost empty except for a supported/blocked-claims table near the top.
- The supplement page is visibly a pointer list; the reproduction command block also runs into the right margin.

## Post-rebuild resolution audit

The `OPEN` values above are the state before rewriting. The following table is
the final resolution after source regeneration, compilation, scalar-provenance
validation, claim-language checks, reviewer simulation, and visual inspection.

| IDs | Final status | Evidence of repair |
| --- | --- | --- |
| F01--F03, F21 | FIXED | 50 verified works, 50/50 cited, analytical related work, novelty matrix, and zero undefined/unused entries |
| F04 | FIXED | Submission-meta and placeholder scan is clean |
| F05, F24 | FIXED | Eligibility mask, binary supervised target, raw-label mapping, and prediction unit are explicit in the formalism and dataset cards |
| F06--F07, F12, F22 | FIXED | Scientific experiment names and a three-level contribution hierarchy replace repository chronology and checklist prose in the main paper |
| F08--F09 | FIXED | Deployment contract, typed evidence, typed claims, support relation, five propositions, and 14 mutation cases replace the administrative gate narrative |
| F10 | FIXED | Exact dataset cards, protocol cards, model equations, feature/construction definitions, training settings, selection, thresholds, and resources are rendered |
| F11 | FIXED | Paired intervals, effect sizes, declared Holm families, ten-seed blocks, rank divergence, and 208 IBM context-sensitivity rows are reported |
| F13--F19 | FIXED | All superseded figures were replaced; final main float audit covers seven figures and seven tables with no blank page, clipping, or float backlog |
| F20 | FIXED | The supplement is an independent 47-page document with 23 rendered longtables, five figures, proofs, methods, complete results, and reproduction commands |
| F23 | RESOLVED_BY_SCOPE | GraphSafe is a bounded decision case; none of 48 adjusted comparisons is significant and Elliptic supplies the negative comparison |
| F25 | FIXED | Hardware claims use recorded runtime environments; aliases and diagnostics cannot fill measured cells |
| F26 | FIXED_BY_EXCLUSION | The 240 V24 duplicate-label rows are excluded from stress claims; only 120 deduplicated base cells remain supplementary evidence |
| F27 | RESOLVED_BY_DISCLOSURE | The shared 60% label-free history and first-50% classifier-label window are stated in the abstract, methods, supplement, and threats; no pure first-50%-history claim remains |

Final machine state: 14-page main PDF, 47-page supplement, 6,796 scalar
provenance records, 28 generated-analysis and eight canonical hashes stable,
zero undefined citations/references, zero overfull boxes, zero Type 3 fonts, and
all 61 pages visually inspected. Genuine evidence limits are retained in
`FINAL_TKDE_READINESS_REPORT.md`; they are not mislabeled as repairs.
