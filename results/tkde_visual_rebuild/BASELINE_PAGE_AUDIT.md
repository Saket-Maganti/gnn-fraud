# Baseline page audit

## Method

The frozen 14-page main paper and 47-page supplement were rendered at 200 dpi. Every page was inspected in full-page contact sheets, at original rendered resolution, and in grayscale. Utilization is an approximate share of the printable region containing meaningful content; it is not a scientific metric. The purpose is to identify design and pagination defects before reconstruction.

## Main paper

| Page | Primary content | Utilization | Print/grayscale finding | Disposition |
| ---: | --- | ---: | --- | --- |
| 1 | Title, abstract, introduction | 96% | Dense but mechanically readable; abstract is long and visually heavy. | Minor local prose/caption integration only. |
| 2 | Introduction, contributions, related work | 97% | Normal IEEE density; no visual defect. | Keep text architecture. |
| 3 | Related work, task notation, contract definitions | 98% | Display equations are readable; no visual object needs enlargement. | Keep equations; tighten transitions. |
| 4 | Related-work table, deployment-contract figure, contract prose | 94% | Table uses small type; Figure 1 is slide-like, pastel, prose-heavy, and still relies on colored status boxes in grayscale. | Redesign both objects; reduce Figure 1 height. |
| 5 | Dataset/protocol/model narrative | 98% | Text-only page is readable but delays the tables that define the study. | Integrate new protocol matrix nearby. |
| 6 | Dataset and model/construction tables, experiment design | 96% | Both full-width tables use small type and answer multiple questions; the model table belongs in detailed cards. | Redesign dataset table; move model details to supplement; add main protocol matrix. |
| 7 | Statistical design and start of RQ1/RQ3 prose | 98% | Primary IBM discussion begins here but its table is delayed to page 9. | Rebuild float order. |
| 8 | Protocol-effect table and figure; later RQ text | 91% | The forest/slope plot is the strongest main visual, but table and plot labels are below the V2 type floor. The figure is legible in grayscale. | Restyle and retain; shorten captions. |
| 9 | Oversized IBM Table V and resource Table VI | 91% | IBM table mixes baseline and graph questions, repeats winner columns, and requires zoom. Resource table contains administrative file counts. | Split IBM baseline/construction; simplify resource table. |
| 10 | Six-panel IBM grid, GraphSafe table, implications | 90% | Six panels are crowded at print scale; configuration identity is color-dependent; GraphSafe caption dominates its table. | Replace IBM plot with a readable baseline comparison; restyle GraphSafe. |
| 11 | Matched-ablation and rank-divergence figures | 88% | Ablation shows AUPRC only despite prose discussing three metrics; the right panel title is clipped. Rank plot is useful but uses many categorical colors. | Redesign ablation as multi-metric forest; repair/restyle rank plot. |
| 12 | Four-panel runtime plot, validity prose | 92% | Points are crowded, legend is large, and configuration identity depends on color; blocked GINE is correctly nonnumeric. | Simplify encodings, preserve cell-specific runtime/performance. |
| 13 | Claim-mutation flowchart, conclusion, start of references | 92% | Figure is administrative, depicts only a subset of 14 cases, and lands five pages after first RQ6 discussion. | Replace with data-driven confusion/transition summary and move near discussion. |
| 14 | References | 78% | Readable two-column reference tail; unused lower region is normal. | Keep unless repagination changes tail. |

Main-paper summary: all scientific content is intact, but 13 of 14 floats are double-column, every table uses `\scriptsize`, and several objects land pages after their first discussion. The critical redesigns are the deployment framework, IBM baseline/construction split, resource table, multi-metric construction effects, runtime view, and validator display.

## Supplement

| Page | Primary content | Utilization | Print/grayscale finding | Disposition |
| ---: | --- | ---: | --- | --- |
| 1 | Title, abstract, first contents page | 86% | Readable; abstract is administrative and the navigation starts without a reader guide. | Rewrite navigation abstract locally. |
| 2 | Contents continuation | 62% | Large unused lower region; expected from a long TOC but signals over-fragmentation. | Rebuild 20-part reviewer path and repaginate. |
| 3 | Scope, status vocabulary, notation, contract start | 90% | Readable body; status list is useful. | Keep with navigation edits. |
| 4 | Contract diagram and formal definitions | 86% | Duplicate slide-like figure; its pastel status encoding weakens in grayscale. | Use compact redesigned main figure or remove duplication. |
| 5 | Propositions and complete 22-claim ledger | 82% | Claim ledger is microscopic and visually administrative. | Move to thematic claim tables later in supplement. |
| 6 | Dataset descriptions and protocol definitions | 91% | Strong explanatory prose. | Retain and expand into dataset cards. |
| 7 | Dataset/protocol tables and model equations | 88% | Tables are tiny; multiple scientific roles share the page. | Split dataset cards and protocol visibility matrix. |
| 8 | Model and construction definitions | 83% | Readable prose; a modest unused lower region. | Keep and add readable role-specific model cards. |
| 9 | Model/training landscape tables | 35% | Severe underutilization; two tiny tables occupy a narrow strip near the top. | Remove landscape; split into portrait model/training cards. |
| 10 | Metrics and statistical procedure | 92% | Readable equations and prose. | Keep with minor clarification. |
| 11 | Node-protocol interpretation and figure | 83% | Figure repeats the main visual; explanatory prose is useful. | Retain prose; remove redundant figure if aggregate table suffices. |
| 12 | RB09 seed rows | 54% | Raw 180-row CSV-style table begins; text is microscopic. | Move all rows to artifact; show schema/completeness only. |
| 13 | RB09 seed rows | 54% | Same defect. | Move to artifact. |
| 14 | RB09 rows/effects, V24 summary | 55% | Multiple unrelated tables stacked at tiny type. | Keep aggregate effects and V24 summary in focused portrait tables. |
| 15 | V22 lane tables | 31% | Very sparse landscape page with tiny tables centered at top. | Replace with one readable completeness/negative-control summary. |
| 16 | V22 statistical rows | 56% | Raw 198-test dump; microscopic. | Move full rows to artifact; retain family summary. |
| 17 | V22 statistical rows | 56% | Same defect. | Move to artifact. |
| 18 | V22 statistical rows | 48% | Same defect and large unused lower region. | Move to artifact. |
| 19 | IBM evidence, baseline, ranks, construction prose | 78% | Strong technical explanation with excessive white space caused by pending floats. | Retain prose; place curated tables adjacent. |
| 20 | Matched IBM ablation figure | 36% | Sparse figure-only page; plot is legible but duplicates the main AUPRC view. | Replace with denser curated multi-metric/context summary. |
| 21 | IBM seed rows | 50% | First of twelve pages of 840-row dump; unreadable without zoom. | Move to artifact. |
| 22 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 23 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 24 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 25 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 26 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 27 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 28 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 29 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 30 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 31 | IBM seed rows | 50% | Raw dump. | Move to artifact. |
| 32 | End of IBM seed rows and cell aggregates | 50% | Raw/aggregate tables still microscopic. | Move seeds to artifact; split aggregates by question. |
| 33 | Cell aggregates, rank table, ablation table | 62% | Three distinct questions are stacked in tiny type. | Replace with focused baseline, rank, and matched-effect tables. |
| 34 | Aggregate and context-specific effects | 59% | Context rows begin as a raw statistical dump. | Replace with direction/heterogeneity summary; retain all rows in artifact. |
| 35 | Context-specific effects | 62% | Raw 208-row dump. | Move to artifact. |
| 36 | Context-specific effects | 62% | Raw dump. | Move to artifact. |
| 37 | Context-specific effects | 35% | Final sparse raw-table page. | Move to artifact. |
| 38 | Runtime/feasibility rows | 44% | Tiny table and large unused region. | Replace with concise Pareto/resource summary and case prose. |
| 39 | GraphSafe policy and aggregate interpretation | 78% | Useful prose; no major mechanical defect. | Retain and add comparator table nearby. |
| 40 | GraphSafe aggregate/tests/budget rows | 55% | Several tables stacked in microscopic type. | Split by scientific question; summarize full test family. |
| 41 | Review-budget rows | 33% | Sparse table-only page. | Replace with readable aggregate table/figure. |
| 42 | Review-budget figure and resource section/table | 82% | Figure is useful but relies on four colors; resource table is too small. | Restyle grayscale-first; turn resources into case cards. |
| 43 | Validator flowchart and surrounding prose | 61% | Duplicate administrative diagram and large unused region. | Remove duplicate; use controlled-case table and interpretation. |
| 44 | Validation and false-promotion tables | 40% | Tiny administrative tables on a sparse page. | Retain as readable, focused controlled-validation summaries. |
| 45 | Evidence-family/lock table and reproduction prose | 78% | Table exposes long internal paths at tiny type. | Move exhaustive paths to artifact; keep concise provenance map. |
| 46 | Commands, checksums, packaging, limitations | 86% | Command block is readable; raw hash list is not useful in the PDF. | Keep reproduction command and move exhaustive hashes to artifact index. |
| 47 | Limitations, ethics, references | 87% | Readable, balanced ending. | Retain; repaginate after curation. |

Supplement summary: pages 12--18 and 21--38 are dominated by raw rows, while pages 9, 15, 20, 37, 38, 41, 43, and 44 use less than roughly half of the printable area. The grayscale issue is secondary to scale: the principal failure is that machines, not reviewers, are the appropriate interface for these exhaustive rows.

## Baseline decision

The frozen PDFs remain valid scientific references and are preserved under `results/tkde_visual_rebuild/before/`. The visual pass will not repair them in place by shrinking fonts. It will rebuild the object allocation, preserve complete rows in the evidence archive, and regenerate both human-readable PDFs from curated sources.
