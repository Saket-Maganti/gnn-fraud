# Final visual QA report

Verdict: **PASS**

## Inspected surface

- 14 main-paper pages and 30 supplement pages rendered at 200 dpi.
- Every page inspected at full-page and print scale in color and grayscale.
- Four main contact sheets and eight supplement contact sheets inspected in both color and grayscale.
- Seven main-paper figures, eight main-paper tables, six supplement figure instances, and 43 curated supplement tables inspected in their rendered context.
- All eight generated figure assets inspected separately at native resolution.

## Final findings

- No clipped label, tick, caption, rule, or table cell remains.
- No active `tiny`, `scriptsize`, `resizebox`, landscape, or sidewaystable construct remains.
- No active table is a raw CSV dump; all table bodies use footnotesize or larger.
- Every substantive supplement table is introduced and interpreted in prose and states a scope or reading limit.
- No missing or resource-blocked cell is numeric, ranked, averaged, or placed on a performance axis.
- No figure relies on color alone. Model, policy, significance, feasibility, and status distinctions also use markers, fills, line styles, direct labels, or explicit text.
- Figure 2 remains readable in grayscale through circle/square/triangle markers and solid/dashed lines.
- Figure 4 retains Holm status through filled/hollow points and keeps Medium GINE in an off-scale `T4 OOM` box.
- Figures 3, 5, 6, and 7 retain family, winner, Pareto, and validator distinctions through marker shape, fill, labels, or paired symbols.
- Page rhythm is stable: no blank page, stranded table, sparse landscape page, unresolved float backlog, or conclusion split by a late float.

## Repairs made during final inspection

1. Wrapped the overflowing `Deployment contract` label inside Fig. 1.
2. Increased the Fig. 2 forest margin to restore all DGraphFin labels.
3. Widened the Fig. 5 rank bounds to restore the left `DegCap` label.
4. Increased vertical separation and right crop margin in Fig. 4 to remove a panel-title/x-label collision and protect the final tick.
5. Balanced the final main-paper bibliography page across two columns.
6. Recompiled and rerendered both PDFs after every repair.

## Automated corroboration

- Active-table audit: 51 tables, 0 errors, 0 warnings.
- PDF/layout audit: 44 pages, 0 errors, 0 warnings.
- Main PDF: 34 of 34 fonts embedded; 0 Type 3 fonts.
- Supplement PDF: 30 of 30 fonts embedded; 0 Type 3 fonts.
- LaTeX logs: no overfull box, undefined citation/reference, duplicate label, missing character, or rerun warning.
- Scientific-delta gate: `ZERO_SCIENTIFIC_DELTAS`.

The human inspection closes the automated audit's explicit review requirement. It establishes publication-design quality for the current PDFs, not scientific truth, external artifact validation, or venue compliance.
