# FraudShiftBench TKDE visual-rebuild release

This namespaced release is a publication-design delta over the frozen
`PROFESSOR_REVIEW_READY` scientific baseline. It does not replace or mutate the
baseline archives. The expected successful automated verdict is
`TKDE_VISUAL_REBUILD_COMPLETE_PROFESSOR_REVIEW_READY`; venue policy, author
metadata, and human scientific review remain external gates.

## Package separation

- `tkde_visual_manuscript_package.zip`: final main/supplement PDFs, active
  LaTeX dependency closure, bibliography, referenced figure/table assets, and
  publication-design audit reports.
- `tkde_visual_source_analysis_package.zip`: everything above plus all visual
  generators, every declared/frozen generator input, figure/table source CSVs,
  exhaustive RB09/V22/IBM rows moved out of the PDF, environment files, and
  provenance manifests.

Raw datasets, per-example prediction payloads, imported workspaces, caches,
private paths, credentials, LaTeX auxiliaries, and stale assets are excluded.
Identity-bearing scanner configuration is also excluded from anonymous packages.
Resource-blocked cells remain nonnumeric and excluded from rankings.

## Clean-room regeneration

From the extracted source-analysis package, install the declared Python and
LaTeX dependencies, then run:

```bash
python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --strict --skip-baseline-archives
python3 scripts/tkde_visual_rebuild/build_main_tables.py
python3 scripts/tkde_visual_rebuild/build_curated_supplement_tables.py
python3 scripts/tkde_rebuild/make_figures.py
python3 scripts/tkde_rebuild/build_bibliography.py
bash scripts/tkde_rebuild/compile_papers.sh
python3 scripts/tkde_visual_rebuild/audit_table_readability.py --root . --strict
python3 scripts/tkde_visual_rebuild/audit_pdf_layout.py --root . --strict
python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --strict --skip-baseline-archives
```

The publisher executes this sequence in a newly extracted temporary directory
before publishing either ZIP. The baseline archive hashes are verified in the
full repository immediately before and after packaging.
