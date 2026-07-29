# Figure Audit

## Scope and standard

The rebuild replaces every figure from the superseded draft. Each replacement
has a PDF, PNG preview, source-data CSV, generation-script pointer, filters,
seed scope, and SHA-256 recorded in `FIGURE_DATA_PROVENANCE.csv`. Figures were
generated at their intended final width with Matplotlib TrueType embedding
(`pdf.fonttype=42`), rendered to raster previews, and inspected for clipping,
legibility, ambiguous missing values, and claims not present in the source data.

## Replacement figures

| ID | file | role | visual/scientific audit |
|---|---|---|---|
| F01 | `fig01_deployment_contract.pdf` | Main: framework | Six separately recorded, non-substitutable contract coordinates are visually distinct from the evidence unit, typed claim, support test, and statuses. The lower status row was respaced after the first render clipped text. |
| F02 | `fig02_protocol_architecture_effects.pdf` | Main: RQ1--RQ2 | Shows paired effects and actual strict/isolated means with intervals. It never imputes missing evidence as zero. Labels and panel margins were repaired after the first render. |
| F03 | `fig03_ibm_metric_scale_construction.pdf` | Main: RQ3 | Uses configuration-specific means and ten-seed bootstrap intervals. Small and Medium feasibility sets remain visibly separate. |
| F04 | `fig04_rank_decision_divergence.pdf` | Main: RQ4 | Reports rank correlations and a concrete AUPRC/F1 reversal; it does not substitute categorical winner counts for metric values. |
| F05 | `fig05_matched_ablation_effects.pdf` | Main: RQ3 | Uses only paired, matched cells against the h64 reference. Four fixed contexts are averaged within each seed before interval estimation and testing, giving ten seed blocks; all 208 context-specific sensitivity rows remain in the supplement. Small-only GINE is not averaged with Medium configurations. Long labels were reflowed after the first render. |
| F06 | `fig06_runtime_resource_pareto.pdf` | Main: RQ5 | Plots each configuration's own runtime and AUPRC. Pareto flags are within compatible cells; blocked Medium GINE is displayed as a status rather than a numerical point. |
| F07 | `fig07_review_budget_analysis.pdf` | Supplement/case study | Aggregates within six protocol--model contexts and then across ten seed blocks. It is retained as a bounded operational analysis, not evidence of universal GraphSafe dominance. |
| F08 | `fig08_claim_support_validation.pdf` | Main: RQ6 | Shows observed status transitions for fourteen validator mutations. The diagram records validation outcomes rather than treating hand-authored claim counts as a scientific result. |

## Disposition of superseded assets

The earlier heat map filled absent cells with zero, the runtime plot reused a
cell winner's performance for every configuration, the ablation plot mixed
unequal feasibility sets, and the claim-count graphic visualized administrative
labels. Those files remain in the dirty research checkout for historical
reproducibility but are not referenced by either rebuilt document and are
excluded from the curated release packages. They also retain Type 3 fonts; all
eight replacement PDFs use embedded CID TrueType fonts.

## Final embedded-PDF check

The final 14-page main PDF and 47-page supplement were rendered after the last
source change and inspected in page order. `TABLE_AND_FLOAT_PLACEMENT_AUDIT.md`
records page placement, effective width, readability, and main-paper disposition.
No embedded figure is clipped, overlapped, illegible, or backed by a Type 3
font. A successful standalone render was not treated as sufficient without this
in-document inspection.
