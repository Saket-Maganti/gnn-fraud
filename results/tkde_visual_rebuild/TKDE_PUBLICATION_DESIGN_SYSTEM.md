# FraudShiftBench TKDE publication design system

## Purpose

This is the authoritative visual configuration for the main paper, supplement, generated figures, and generated tables. It preserves the frozen scientific record while separating decisive communication (main), readable technical evidence (supplement), and exhaustive rows/provenance (artifact).

## Typography

- IEEEtran body typography remains unchanged.
- Main-figure base text: 8.0 pt at final physical size; tick labels never below 7.5 pt.
- Supplement-figure base text: 8.0 pt at final physical size; tick labels never below 7.0 pt.
- Main-table body: `\footnotesize` (approximately 8 pt). No `\scriptsize`, `\tiny`, or `\resizebox` is permitted for a substantive table.
- Supplement-table body: `\footnotesize`; notes may use `\small`/`\footnotesize` but never `\scriptsize` or `\tiny`.
- Panel labels are bold, lowercase letters in parentheses, aligned to the upper-left of each panel.
- Mathematical symbols use LaTeX math typography. Dataset/model names stay roman; variables and estimands stay italic.
- Abbreviations are expanded at first local use. Table headers use short forms only when a note defines them.
- Decimal precision follows the evidence surface: four decimals for AUPRC/AUROC/F1/effects, three for standardized effects and runtimes when finer precision is not scientifically meaningful, and explicit inequality notation for very small p-values.
- Ranges use LaTeX en dashes (`--`); negative values use mathematical minus signs.

## Color and accessibility

- Grayscale is the base. Text, axes, rules, and reference marks use neutral grays.
- Two scientific accents are permitted: blue (`#0072B2`) and vermilion (`#D55E00`). A muted teal (`#009E73`) may identify a reference or feasible construction only when necessary.
- Categorical status is never encoded by color alone. Labels, open/filled markers, hatching, or explicit status tags carry the same meaning.
- Blocked and unmeasured cells use a labeled nonnumeric glyph (`×` or an outlined status box), never a plotted zero.
- Every key figure must remain interpretable after conversion to grayscale.

## Figure geometry

- One-column canvas: 3.45 in wide, normally 2.0--3.2 in high.
- Two-column canvas: 7.16 in wide, normally 2.3--4.4 in high.
- Figures are saved at exact final dimensions with `bbox_inches=None`; physical scaling may not depend on tight bounding-box heuristics.
- Minimum plotted line width: 0.9 pt. Minimum marker diameter: 4.5 pt. Error-bar caps: at least 2.0 pt.
- Confidence intervals are whiskers or low-opacity bands. Point estimates remain visible without color.
- Direct labels are preferred when six or fewer series are shown. Legends must not obscure data and should be outside the data region when practical.
- Small multiples share axes only when metrics and scales are comparable. AUPRC, AUROC, and F1 are never forced onto a common numerical axis.
- Missing cells are omitted with an explicit label; resource-blocked cells are displayed outside the predictive scale.
- Every figure has a source-data CSV and a provenance/checksum entry.

## Table system

- Use `booktabs`, `tabularx`, and `siunitx` numeric alignment where it materially improves scanning.
- No vertical rules, boxed cells, or color-filled winner grids.
- A table answers one primary scientific question. Unrelated questions are split or moved.
- Winner columns are avoided when boldface and paired deltas provide the same information.
- Main tables contain aggregates, paired effects, uncertainty, feasibility, and concise notes. Raw seed rows, path lists, checksums, and exhaustive test families stay in the artifact.
- Supplement tables provide readable aggregate detail plus explicit scope, seed count, correction family, and feasibility treatment. Representative schema rows may be shown, but exhaustive row dumps are prohibited.
- Longtable is reserved for genuinely human-readable multi-page material with repeated useful headers. Landscape and sidewaystable require a documented exception.
- Table notes define symbols, deterministic-baseline semantics, correction families, and blocked-cell treatment once per table, not in every cell.

## Captions and local prose

- Captions state the scientific question, what is plotted/tabulated, the inferential unit, and the principal takeaway in at most 55 words for main objects and 75 words for supplement objects unless a documented exception is necessary.
- Captions do not repeat all axis labels, methods, or the full methods section.
- Every substantive object is introduced before placement and followed by interpretation.
- The local paragraph names the matched scope, gives the central quantitative result, explains why it matters, and states the feasibility or construct limit.
- Formal support-language is used only when support status is itself the result. Ordinary empirical discussion uses natural scholarly prose.

## Float allocation

- Main paper target: 5--7 figures and 5--7 tables; no more than 6--7 double-column floats. A small one-column comparator table is permitted when it prevents prose overload.
- Full-width floats are reserved for comparisons that need the width. Compact tables and figures should use a single column.
- No unconditional `\clearpage` or `\newpage` is permitted inside scientific sections.
- Supplement objects are placed in the narrative flow. Related tables may share a page; landscape is avoided unless readable portrait layout is impossible.

## Provenance and regeneration

- `scripts/tkde_visual_rebuild/publication_style.py` is the shared machine-readable style authority for figure dimensions, font sizes, colors, precision, caption thresholds, and blocked-value semantics.
- Generators must fail on missing required inputs, nonnumeric blocked cells represented numerically, unmatched feasibility pooling, or missing provenance.
- Regeneration must reproduce the curated main and supplement surfaces without restoring raw-row tables or legacy styling.
