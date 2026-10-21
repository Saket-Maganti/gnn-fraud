# Reviewer 2: Graph ML and Temporal GNNs

## Recommendation

**Major revision / weak reject.** The paper asks an important evaluation question and the Elliptic visibility reversal is cleanly demonstrated. The graph-ML coverage, however, is narrower than the title may lead readers to expect: the principal real-graph experiment contains MLP, GCN, and mean GraphSAGE, while the edge-conditioned GINE result exists only on IBM Small. No temporal message-passing model or modern transformer is part of the locked primary comparison. The paper can succeed as an evaluation-framework study, but not as a broad empirical verdict on temporal graph fraud architectures.

Scores use a 1--5 scale.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Novelty | 3 | Novelty lies in evaluation contracts and support semantics, not a graph-learning method. |
| Technical depth | 3 | The protocol intervention and construction controls are thoughtful, but the learned models are mostly established and several IBM “graph” models use fixed one-hop summaries. |
| Empirical rigor | 3 | Ten seeds and matched controls are strong; architecture breadth and feasible scale coverage are limited. |
| Statistical validity | 4 | Real-data effects are paired and Holm-corrected. IBM fixed contexts are now averaged within seed for n=10 inference, with all 208 context-specific effects retained. |
| Clarity | 4 | The manuscript now distinguishes full GNNs, fixed-summary edge classifiers, aliases, and blocked configurations. |
| Literature positioning | 4 | Relevant graph fraud, graph anomaly, temporal benchmark, and directed/edge-aware methods are discussed. |
| Reproducibility | 5 | Predictions, manifests, locked result families, exact hyperparameters, and resource outcomes are exposed. |
| Significance | 3 | The evaluation lesson is significant; the empirical architecture conclusions are local. |
| Limitations | 5 | The authors report negative results, OOM cells, task differences, and mechanism limits directly. |

## Strengths

- The strict-versus-isolated intervention holds masks, features, training data, and seeds fixed. The unchanged MLP is a useful negative control.
- The Elliptic result is large and unambiguous within the grid: GraphSAGE changes from 0.6114 to 0.4797 mean AUPRC, while GCN changes from 0.4666 to 0.5605. The leader therefore changes under graph visibility.
- DGraphFin supplies a necessary counterexample: GraphSAGE remains the leader, although its AUPRC falls by 21.5% relative. The paper does not universalize the reversal.
- IBM matched controls are better designed than a winner table. NoEdge and ShuffledEdge test transaction-attribute contribution, DegreeOnly changes the information source, DegreeCap changes hub exposure, and RecentWindow changes recency and computation.
- The manuscript correctly calls the IBM h32/h64 models “GraphSAGE-derived edge classifiers” over fixed one-hop summaries. It does not pretend that these are generic end-to-end GraphSAGE implementations.
- GINE's Medium OOM is not converted into underperformance, and the reduced DGraphFin GAT diagnostic is not substituted for the fixed h64/l2 cell.

## Fatal flaws and major concerns

### G1. The architecture sample does not support broad temporal-GNN conclusions

**Severity:** Potentially fatal if the paper is read as an architecture benchmark.  
**Status:** Unresolved evidence limitation.

The real-data visibility grid includes no TGN-style memory model, temporal attention model, heterogeneous GNN, transformer, or sampling-based large-graph model. The primary learned graph comparison is GCN versus mean GraphSAGE. GINE appears only in four Small IBM contexts, and its Medium cells are resource-blocked. The study therefore demonstrates that graph visibility can change conclusions for the tested architectures; it does not characterize modern temporal GNNs as a class.

This cannot be repaired by prose into broader evidence. Either additional locked runs are needed, or the title, abstract, and conclusion must remain centered on evaluation methodology and the evaluated grid. The current conclusion is mostly scoped correctly.

### G2. “Temporal graph” here often means static architectures under chronological contracts

**Severity:** Major positioning issue.  
**Status:** Partly repaired by definitions; unresolved coverage risk.

The experimental temporal structure is primarily in chronological masks, visible subgraphs, and history construction. The main models themselves are not temporal-state models. That is a legitimate temporal evaluation study, but readers may expect temporal representation learning. The paper should state this distinction near the contribution list, not leave it to model details.

### G3. The visibility experiment establishes sensitivity, not mechanism

**Severity:** Fatal if causally overinterpreted; repaired in the manuscript.  
**Status:** Repaired by scope.

Removing cross-period edges changes degree, connectivity, message paths, and possibly neighborhood label composition at once. The manuscript now says that the intervention does not identify homophily, oversmoothing, degree shift, or another mechanism. This is the correct conclusion. A future mechanism study should measure degree-conditioned effects, component fragmentation, neighborhood-label shift, and message-path availability, but those analyses are not prerequisites for retaining the current sensitivity claim.

### G4. IBM graph construction is not protocol-symmetric

**Severity:** Major internal-validity limitation.  
**Status:** Disclosed, not empirically repaired.

Both IBM protocols reuse a node-history matrix computed from the first 60% of transactions. Early-to-late labels stop at 50%, so its features use an additional 10% label-free covariate interval. This is not test-label leakage, but it entangles temporal-label partition and covariate visibility. Comparisons across the two IBM protocols should remain descriptive and should not be interpreted as isolated time-window effects.

### G5. The GINE result is a feasible-set result, not evidence of dominance

**Severity:** Fatal if widened; repaired by resource-aware wording.  
**Status:** Repaired with an unresolved resource boundary.

GINE leads mean AUPRC in all four measured Small graph-grid contexts and improves mean Small AUPRC by 0.01190 versus the h64 reference. Its mean F1 is lower by 0.00206, but the repaired seed-blocked Holm test is not significant ($p=0.252$). It averages about nine times the runtime. Both Medium GINE lanes exhausted T4 memory. The current text correctly confines the positive result to Small AUPRC and does not infer Medium performance.

### G6. GraphSafe is not supported as a graph-learning contribution

**Severity:** Fatal if presented as universal method improvement; repaired.  
**Status:** Repaired by demotion and negative reporting.

GraphSafe uses saved branch scores and validation-fitted switching. DGraphFin descriptive means favor it over simple averaging, Elliptic favors simple averaging, and none of the 48 declared GraphSafe-versus-average tests has Holm-adjusted p below 0.05. The paper appropriately uses this as a claim-discipline example. It should not appear in the title or contribution hierarchy as a generally validated method.

### G7. IBM matched inference now uses seed blocks

**Severity:** Former major statistical issue.  
**Status:** Repaired.

The revised construction analysis averages the four fixed variant--protocol differences within each seed before inference, giving $n=10$ seed blocks rather than 40 exchangeable rows. `IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv` reports all 208 context-specific ten-seed effects, and Supplement Table S13b renders them. The repair materially changes interpretation: NoEdge Small AUPRC no longer survives Holm correction ($p=0.082$), and the GINE F1 reduction is also non-significant ($p=0.252$). The paper updates those claims rather than retaining the more favorable pooled results.

## Required revisions

1. State explicitly that this is temporal evaluation of mostly static graph architectures, not an exhaustive temporal-GNN benchmark.
2. Keep every architecture claim local to MLP/GCN/GraphSAGE or the feasible IBM graph grid.
3. Preserve the repaired seed-blocked IBM inference, context table, GINE feasibility boundary, and GraphSafe negative comparisons.
4. If space permits, include a compact topology diagnostic for strict versus isolated graphs; otherwise identify it as a targeted follow-up rather than imply a mechanism.

## Decision rationale

The paper contains a strong evaluation result, but a graph-ML reviewer can reasonably ask whether the model grid is representative enough for the venue claim. I would support a revised paper that presents the graph experiments as carefully controlled demonstrations of the contract framework. I would not support a version that markets three standard node models and Small-only GINE as a comprehensive temporal-GNN benchmark.
