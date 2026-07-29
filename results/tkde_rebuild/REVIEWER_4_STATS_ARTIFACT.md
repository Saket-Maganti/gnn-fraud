# Reviewer 4: Statistics, Reproducibility, and Artifact

## Recommendation

**Minor revision / weak accept.** The evidence handling is substantially better than typical research-code packaging. The earlier IBM unit-of-inference concern has been repaired: four fixed contexts are averaged within seed, confirmatory inference uses $n=10$, and all 208 context-specific sensitivity rows remain inspectable. The changed conclusions are reported rather than suppressed. The curated archive now passes deterministic rebuild, checksum, exclusion, private-path, credential, and ZIP-integrity checks. The remaining artifact limitation is the inherent distinction between saved-output reconstruction and clean-room retraining.

Scores use a 1--5 scale.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Novelty | 3 | Typed support and resource semantics are useful; the statistical tools themselves are standard. |
| Technical depth | 4 | The artifact connects claims, predictions, seeds, locks, exclusions, and deterministic derivations. |
| Empirical rigor | 4 | Ten seeds, paired contrasts, negative controls, complete prediction manifests, and explicit blocked cells are strong. |
| Statistical validity | 4 | Visibility, IBM, and GraphSafe analyses now all use ten seed blocks for their aggregated inference, with context-specific IBM effects also reported. |
| Clarity | 5 | Declared analysis families, thresholds, deterministic baselines, seed blocks, and inferential limits are explicit. |
| Literature positioning | 4 | Leakage, selection bias, rare-event metrics, calibration, and benchmark governance are covered. |
| Reproducibility | 5 | 6,796 scalar provenance records, deterministic generators, hash checks, and no-training reconstruction are excellent. |
| Significance | 4 | Making unsupported promotion executable is valuable if the artifact stays synchronized with the paper. |
| Limitations | 5 | The manuscript distinguishes conditional seed uncertainty from population generalization and reports failed constructs. |

## Statistical strengths

1. The primary real-data contrast is paired by seed, and the MLP provides a graph-intervention negative control.
2. Visibility tests report mean changes, percentile bootstrap intervals, paired Cohen's dz, Wilcoxon p-values, and Holm correction within declared metric families.
3. GraphSafe correctly averages six protocol--model contexts within each seed before inference, giving n=10 rather than the pseudoreplicated n=60. Holm correction spans the declared 48 comparisons, and no adjusted p-value is below 0.05.
4. IBM construction inference now averages four fixed contexts within each seed, also giving n=10. The 208 context-specific ten-seed effects remain available for heterogeneity review.
5. Blocked cells are neither imputed nor ranked. Small-only GINE is not pooled with the Medium feasible set.
6. The paper separates descriptive winner counts and rank correlations from confirmatory paired tests.
7. The label convention and eligibility mask prevent unknown nodes from entering supervised metrics.

## Artifact strengths

- The IBM primary package has 840 validated result files and 840 corresponding prediction exports; the real-data visibility family has 180 result rows and 180 predictions.
- `NUMBER_PROVENANCE_MAP.csv` contains 6,796 scalar records linking source paths, filters, seed sets, aggregation, outputs, locks, and claim IDs.
- Fourteen controlled claim/evidence mutations return all expected support statuses.
- Regeneration preserved 28 generated-analysis CSV hashes and eight audited canonical input hashes.
- The audit catches 310 result files and 310 prediction files that count-only logic could promote incorrectly, including 240 V24 duplicates per artifact type.
- Resource failures, compatibility aliases, partial imports, and reduced diagnostics remain visible without being promoted to evidence.

## Fatal flaws and major concerns

### S1. IBM dependence-aware inference has been repaired

**Severity:** Former major concern.  
**Status:** Repaired from existing evidence.

For each size, the revised analysis first computes candidate-minus-reference differences in the four fixed variant--protocol contexts and then averages them within seed. Bootstrap intervals, Cohen's $d_z$, and Wilcoxon tests therefore use ten seed blocks. `IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv` contains 208 context-specific rows, rendered as Supplement Table S13b. This is not a cosmetic repair: NoEdge Small AUPRC now has Holm $p=0.082$, and GINE's F1 reduction has $p=0.252$. The manuscript updates the claims while preserving the descriptive effect sizes and context directions.

### S2. Percentile bootstrap intervals with ten seeds have limited resolution

**Severity:** Major limitation, not a computational error.  
**Status:** Disclosed but only partly addressed.

Ten seeds permit paired analysis, but a 10,000-resample percentile interval does not create more independent information. The inferential target is variability under the repository's stochastic training seeds on fixed datasets and splits, not variation across banks, time periods, or data-generating processes. The threats section states this. Context-specific raw points and exact nonparametric p-values should remain available in the supplement.

### S3. Correction-family wording is now accurate

**Severity:** Former nonfatal wording risk.  
**Status:** Repaired.

The manuscript now uses “declared analysis families” and makes no prospective-preregistration claim. Holm correction applies to those stated families, not to every historical exploratory comparison in the repository.

### S4. The validator tests specification conformance, not construct truth

**Severity:** Fatal if sold as scientific verification; repaired by limitation language.  
**Status:** Repaired by scope, with residual circularity.

The same project defines the claim schema, constructs the mutations, and implements the validator. Passing 14/14 cases demonstrates behavior on those fixtures, not completeness of the ontology or correctness of curator judgment. The manuscript explicitly states that the validator cannot establish construct validity or causal explanation by itself. Independent fixtures or a second implementation would strengthen the artifact but are not present.

### S5. No-training reproducibility is verified; full training reproducibility is not

**Severity:** Major artifact limitation.  
**Status:** Partially resolved.

The rebuilt paper, analyses, tables, and figures can be regenerated from saved evidence. The rebuild did not train models, and raw datasets have access and licensing constraints. A command path for full reruns is not equivalent to a verified clean-machine rerun. The artifact claim should remain “deterministic reconstruction from locked saved outputs,” with full training listed separately and without an artifact-badge claim.

### S6. Generated audit state matches the current PDFs

**Severity:** Mechanical final-build gate.  
**Status:** Repaired in the current build; rerun after any later source change.

The compiled PDFs currently have 14 main pages and 47 supplement pages. `MANUSCRIPT_MACHINE_AUDIT.csv` records those same counts, no overfull boxes, no Type 3 fonts, and no undefined citations or references. This audit must still be rerun after any later source or layout change.

### S7. Identical deterministic seed rows need careful interpretation

**Severity:** Nonfatal.  
**Status:** Repaired by explicit interpretation.

Some library baselines produce identical values across nominal seeds. The main threats section and supplementary statistics section now state that these rows document repeated pipeline evaluation, not ten independent stochastic fits, and leave zero variance visible. Winner means remain factual without rhetorically amplifying their uncertainty evidence.

## Required revisions

1. Keep the inference target conditional on fixed datasets and splits.
2. Rerun the machine audit after any final source change to the 14/47-page PDFs and inspect every page.
3. Preserve the passing curated-archive manifest and rerun it after any packaged-file change.
4. Describe full training as an unverified rerun path unless it is actually executed cleanly.

## Decision rationale

The artifact discipline is a clear strength. V24 duplicate labels are excluded, V22 failed imports are not promoted, IBM and GraphSafe use seed blocks, deterministic baselines are identified, and blocked hardware cells receive no score. The dependence-aware repair changed nominal significance where it should have, which increases confidence in the audit process. The final release archive and machine audit pass, so I recommend acceptance from the statistical/artifact perspective.
