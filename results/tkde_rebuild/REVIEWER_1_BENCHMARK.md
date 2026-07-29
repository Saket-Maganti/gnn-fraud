# Reviewer 1: Data Engineering and Benchmark Design

## Recommendation

**Minor revision / weak accept.** The submission is unusually careful about evidence provenance and blocked cells, and the conditional empirical results are useful. The acceptance case still depends on whether the deployment contract and typed support relation are judged to be a scientific benchmark contribution rather than a disciplined artifact schema. The revision now states the coordinates as separately recorded and non-substitutable, uses dependence-aware IBM inference, and exposes every context-specific effect. The remaining concerns are external-validation and breadth limits, not known defects in the presented evidence.

Scores use a 1--5 scale, where 1 is poor, 3 is adequate for the venue, and 5 is outstanding.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Novelty | 3 | The six-axis contract plus executable claim--evidence relation is a useful synthesis, but it is adjacent to BenchmarkCards, Eval Factsheets, BetterBench, temporal graph benchmarks, and mature documentation practice. |
| Technical depth | 4 | Typed evidence, status semantics, formal properties, integrity locks, prediction manifests, and mutation tests are substantially deeper than a checklist. Several propositions are still close to consequences of the definitions. |
| Empirical rigor | 4 | Ten-seed, prediction-backed evidence; explicit negative controls; matched contrasts; construct-invalid and resource-blocked cells retained. Breadth is limited to two real graphs and one synthetic data family. |
| Statistical validity | 4 | The real-data and GraphSafe analyses use ten seed blocks. IBM construction inference now averages four fixed contexts within seed, uses n=10, and exposes 208 context-specific sensitivity rows. |
| Clarity | 5 | Prediction units, coordinate semantics, resource states, inferential blocks, and prohibited generalizations are explicit. |
| Literature positioning | 4 | The paper now engages graph fraud, dynamic benchmarks, leakage, benchmark governance, documentation, and decision metrics. The novelty is appropriately described as narrow and combinatorial. |
| Reproducibility | 5 | The 6,796-number provenance map, 840/840 IBM result/prediction coverage, 180/180 real-grid coverage, deterministic builders, locks, and exclusions form an unusually strong audit surface. |
| Significance | 4 | The paper exposes how deployment conditions change scientific and operational conclusions. The impact will depend on whether other benchmark maintainers can adopt the support model. |
| Limitations | 5 | The manuscript directly discloses the synthetic IBM scope, task-unit mismatch, first-60% IBM covariate map, invalid stress labels, blocked runs, and lack of institutional deployment evidence. |

## What the paper establishes

The strongest result is not a new leaderboard. It is a disciplined mapping from a scoped claim to the evidence needed to support it. The manuscript correctly keeps transaction-node, account-node, and transaction-edge tasks separate. It also gives concrete examples where scope matters: Elliptic changes its leading AUPRC model under a graph-visibility intervention; DGraphFin changes the margin but not the leader; IBM's ranking and thresholded metrics disagree; and missing GINE/Large cells remain feasibility observations rather than implicit losses.

This is backed by more than manuscript prose. `CLAIM_EVIDENCE_LEDGER.csv` records 22 typed claims and their permitted wording. `FRAMEWORK_VALIDATION_CASES.csv` exercises 14 expected status transitions. `NUMBER_PROVENANCE_MAP.csv` connects 6,796 scalar values to sources and transformations. `IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv` renders all 208 context-specific sensitivity rows. `RESOURCE_BOUNDARIES.md` preserves five unmeasured cells without assigning performance ranks.

## Fatal flaws and major concerns

### B1. The benchmark contribution could still be read as schema engineering

**Severity:** Potentially fatal to acceptance; not a truthfulness defect.  
**Status:** Unresolved external-validation risk.

The support relation is useful, but several formal properties are immediate consequences of how `Required(c)` and the cell key are defined. The 14 mutation tests show that the implementation returns its specified statuses; they do not show that independent users can specify ambiguous scientific claims consistently, that two curators agree on construct validity, or that the framework changes benchmark practice outside this repository. The manuscript now says explicitly that the validator cannot certify scientific truth, which is the correct boundary. Even so, the paper needs reviewers to accept a framework validated primarily on its own artifact.

No existing repository evidence can eliminate this risk. The strongest honest response is to keep the primary novelty at the level of a fraud-specific contract and auditable support relation, avoid “general solution” language, and identify cross-curator or external benchmark adoption as future validation.

### B2. Coordinate non-substitutability is now stated correctly

**Severity:** Former major conceptual issue.  
**Status:** Repaired.

The earlier draft called time, visibility, construction, selection, decision budget, and resources independent. The main text, supplement, and figure now call them separately recorded, non-substitutable coordinates. This makes the intended claim without asserting mathematical, statistical, or causal independence. Interactions remain possible and visible.

### B3. Benchmark breadth is too small for universal design claims

**Severity:** Major acceptance risk.  
**Status:** Unresolved evidence boundary.

There are two real public graphs and four Small/Medium regimes from one synthetic generator. The three datasets also have different prediction units. This is sufficient for scoped counterexamples and for validating the artifact machinery, but not for estimating how often deployment contracts change conclusions across fraud domains. The manuscript mostly contains this risk by rejecting a pooled leaderboard and universal quantifiers. The title and significance claims should continue to signal a framework and conditional map rather than comprehensive coverage of temporal fraud detection.

### B4. Construct-invalid and noncomparable artifacts were present in the repository

**Severity:** Fatal if promoted; repaired in the rebuilt manuscript.  
**Status:** Repaired by exclusion.

The V24 runner produced three labels per base cell without passing the stress argument to the harness. The resulting 360 rows represent 120 scientific cells and 240 labeled duplicates. The evidence audit also identifies 38 V22 result files and 38 prediction files from integrity-failed or noncomparable imports, plus memory-reduced diagnostics that cannot fill fixed-configuration OOM cells. The rebuilt text reports the defect, the support validator rejects the promotion, and the primary conclusions do not use the false contrasts. This is an important repair, but the release manifest must retain the exclusion rationale.

### B5. IBM early-to-late visibility is asymmetric

**Severity:** Fatal if described as first-50%-only inductive history; nonfatal as currently disclosed.  
**Status:** Repaired by contract disclosure; empirical asymmetry remains.

The classifier uses labels from the first 50% of transactions, while the shared account-history map uses label-free covariates from the first 60%. The main paper and supplement now state this explicitly and avoid test-label-leakage language. That repair prevents a false protocol claim, but it does not make the early-to-late and late-window feature constructions symmetric. Protocol differences therefore cannot be attributed to time windows alone.

### B6. Submission-format compliance needs a final venue check

**Severity:** Non-scientific but potentially blocking.  
**Status:** Open administrative preflight.

The compiled main paper is 14 pages and the supplement is 47 pages. The authors should verify the current TKDE regular-paper and supplementary-material rules immediately before submission. This is not a reason to change scientific scope silently; it is a packaging gate.

## Required revisions

1. Keep external adoption and inter-curator agreement explicitly unvalidated.
2. Preserve the V24/V22 exclusions and the IBM 50/60 visibility statement in both the paper and release manifest.
3. Preserve seed-blocked IBM inference and the context-specific table when compressing the supplement.
4. Verify final venue page and supplement rules.

## Decision rationale

I would not reject this work for missing Large or Medium GINE results because the paper treats them correctly as resource boundaries. I would reject it if the contract were sold as an independently validated universal benchmark standard, if invalid stress labels re-entered the evidence, or if IBM's covariate visibility were hidden. The conceptual wording and IBM inferential-unit defects have been repaired. Under the current bounded thesis, the remaining risk is whether the venue considers self-contained framework validation and the evaluated breadth sufficient.
