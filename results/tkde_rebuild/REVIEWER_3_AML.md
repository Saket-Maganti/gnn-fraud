# Reviewer 3: Financial Fraud and AML

## Recommendation

**Minor revision / weak accept as a public-data benchmark paper.** The paper is responsible about the difference between benchmark scores and AML deployment. None of the evidence measures investigator outcomes, confirmed financial harm, alert disposition, feedback loops, or performance in a private institution, but the manuscript now keeps every operational statement scenario-based. It also makes the IBM winner disagreement explicitly specific to the saved 0.5 threshold.

Scores use a 1--5 scale.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Novelty | 4 | Joining temporal access, graph construction, review capacity, and resource feasibility in one fraud-specific contract is useful. |
| Technical depth | 3 | The benchmark formalism is substantial; the AML decision model remains simplified. |
| Empirical rigor | 3 | Two real public graphs and synthetic IBM regimes give useful coverage, but they do not reproduce a bank investigation workflow. |
| Statistical validity | 4 | Ten-seed blocks, corrected paired tests, and the full context-specific IBM sensitivity table address the earlier dependence concern; fixed-dataset inference still limits generalization. |
| Clarity | 5 | Prediction units, prevalence, visibility, thresholds, costs, resource blocks, and intended use are stated unusually clearly. |
| Literature positioning | 4 | Fraud detection, cost sensitivity, review feedback, graph anomaly, and AML datasets are represented. |
| Reproducibility | 5 | The evidence is prediction-backed and the exclusions are auditable. |
| Significance | 4 | Preventing protocol and metric mismatches is directly relevant to financial-model governance. |
| Limitations | 5 | The paper does not claim bank validation, fairness, or automated-decision suitability. |

## What is operationally useful

The paper makes several distinctions that AML evaluations often blur:

- Elliptic predicts illicit transaction nodes, DGraphFin predicts anomalous account nodes, and IBM predicts laundering transaction edges. The manuscript never averages these into one “AML score.”
- AUPRC is treated as a ranking metric, while F1, Precision@K, Recall@K, and cost risk require a threshold or capacity.
- IBM test prevalence is reported for each temporal window, ranging from roughly 0.058% to 0.177%; raw AUPRC is not interpreted without that base-rate context.
- Memory exhaustion and resource guards are governance outcomes, not predictive losses.
- The ethics section rules out automatic punitive decisions and calls for institution-specific validation, human review, appeal procedures, and lawful data handling.

The strongest empirical message for AML practice is that model choice depends on what investigators are optimizing. Histogram gradient boosting has the best mean AUPRC in all eight IBM baseline cells, but AUPRC and fixed-threshold F1 choose different configurations in 11 of 16 feasible contexts. In the HI-Medium late-window graph grid, the AUPRC leader is sixth by F1, while the F1 leader is sixth by AUPRC. That is a useful warning against metric-free claims of superiority.

## Fatal flaws and major concerns

### A1. There is no institutional AML validation

**Severity:** Fatal to any deployment-effectiveness claim; nonfatal for the current benchmark claim.  
**Status:** Unresolved external-validity boundary.

Elliptic and DGraphFin are public, pseudonymized or anonymized graph tasks with labels and entities that do not reproduce a bank's case-management process. IBM AML-Data is synthetic. None of the datasets contains investigator decisions, escalation outcomes, customer harm, delayed adjudication, sanctions-screening interactions, or feedback from prior alerts. The manuscript states this correctly. It must not describe review-budget metrics as evidence of real investigative value.

### A2. The IBM decision comparison is now explicitly threshold-specific

**Severity:** Major if generalized beyond its operating point.  
**Status:** Repaired by scope.

IBM F1 uses a fixed threshold of 0.5. The abstract calls the result a fixed-threshold disagreement, and the main and supplement state that 11 of 16 is specific to threshold 0.5 and is not threshold-invariant. This repairs the claim boundary. The supplement's 0.5%, 1%, and 2% review-budget analysis remains limited to the GraphSafe case on Elliptic and DGraphFin, so the IBM result should still not be read as capacity-policy evidence.

### A3. The cost model is illustrative, not economic validation

**Severity:** Major if overgeneralized; currently contained.  
**Status:** Repaired by scope.

The 1:5 false-positive/false-negative risk is explicitly hypothetical. It omits investigation time, alert aggregation into cases, heterogeneous transaction values, recovery, regulatory exposure, and customer friction. The manuscript correctly calls it a scenario. It should remain in the GraphSafe case study rather than serve as evidence that one policy reduces bank loss.

### A4. IBM's early-to-late feature visibility weakens temporal interpretation

**Severity:** Major construct limitation.  
**Status:** Disclosed; not empirically repaired.

The early-to-late classifier trains on labels from the first 50%, but its node-history features cover the first 60%. That 50--60% interval is label-free and does not contain test labels, but it represents transductive covariate access into the validation period. This is acceptable as a declared contract. It prevents interpreting the early-to-late comparison as a clean first-half training deployment.

### A5. Synthetic scale is not realism

**Severity:** Major external-validity limitation.  
**Status:** Unresolved.

IBM Medium contains more than 31 million transactions and is useful for compute and rarity studies. Scale alone does not establish realistic laundering typologies, adversarial adaptation, institutional process, or label noise. The paper appropriately calls IBM synthetic, but statements about AML behavior should be tied to “IBM AML-Data regimes,” not AML systems generally.

### A6. The resource story is incomplete but honestly bounded

**Severity:** Nonfatal.  
**Status:** Resource-blocked.

Large is unmeasured, Medium GINE exhausts T4 memory, and the fixed DGraphFin GAT cell is also OOM. These results matter to constrained practitioners, but feasibility on T4 does not determine feasibility on another accelerator or with sampling. The manuscript correctly avoids hardware-general performance claims.

### A7. Fairness and disparate impact are not evaluated

**Severity:** Major for deployment; nonfatal for a public benchmark paper if explicit.  
**Status:** Unresolved evidence gap, clearly disclosed.

No validated artifact supports demographic fairness analysis. The paper correctly says that absence of analysis is not evidence of fairness. This limitation should appear in any release card and intended-use statement.

## Required revisions

1. Keep “deployment contract” as a description of evaluation assumptions, not a claim of deployment validation.
2. Preserve the explicit statement that IBM's 11-of-16 disagreement is specific to threshold 0.5; a full IBM capacity surface would be a strengthening, not evidence already claimed.
3. Retain the exact 50/60 covariate-visibility statement and synthetic-data qualifier.
4. Do not translate the 1:5 risk scenario into monetary, investigator-productivity, or regulatory-benefit claims.
5. Keep fairness, feedback loops, and institution-specific calibration as explicit unresolved requirements.

## Decision rationale

I would reject a version claiming that FraudShiftBench validates an AML deployment or proves that one model reduces investigator cost. The revised manuscript avoids those claims, makes the fixed-threshold boundary impossible to miss, and offers useful governance discipline. The unresolved institutional, fairness, and workflow gaps are real but accurately classified as limits on deployment inference rather than hidden benchmark defects.
