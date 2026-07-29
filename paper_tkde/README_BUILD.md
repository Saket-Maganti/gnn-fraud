# TKDE manuscript build

The LaTeX sources in this directory are authoritative. The retired
`scripts/paper_figures/build_tkde_manuscript.py` generator belongs to the
superseded draft and intentionally refuses to overwrite this rebuild unless an
explicit historical-reconstruction flag is supplied.

The scientific TKDE rebuild is frozen at `PROFESSOR_REVIEW_READY`. Do not rerun
the evidence reconstruction or statistical analysis for a publication-design
rebuild. Regenerate the reviewer-facing assets from the repository root:

```bash
gnn_env/bin/python scripts/tkde_rebuild/make_figures.py
gnn_env/bin/python scripts/tkde_visual_rebuild/build_main_tables.py
gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py
gnn_env/bin/python scripts/tkde_rebuild/build_bibliography.py
```

The compatibility entry point `scripts/tkde_rebuild/build_tables.py` delegates
to the same two V2 table builders and cannot restore the retired raw-row tables.

Build the main paper with a strict BibTeX cycle:

```bash
cd paper_tkde
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Build the supplement independently:

```bash
cd paper_tkde/supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

All regeneration commands consume imported, locked evidence. They do not train
models, request a GPU, download datasets, or promote resource-blocked cells to
performance evidence.
