# Clean-room visual release build

Verdict: **PASS**

- Source archive SHA-256: `2c2274a1a3a6dc6dd07139631c20305370619298beb151eb5e4f380cd7c6d6cf`
- Extraction: new temporary directory with path-safe member validation
- Regeneration: main tables, curated supplement tables, figures, and bibliography
- Compilation: strict main and supplement BibTeX cycles
- Post-build gates: table readability, PDF/log/font/layout, and scientific delta

## Commands

| Command | Exit | Last output line |
| --- | ---: | --- |
| `python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --report-dir results/tkde_visual_rebuild/clean_room_audits --strict --skip-baseline-archives` | 0 | scientific_delta_errors=0 warnings=0 frozen_rows=36 |
| `python3 scripts/tkde_visual_rebuild/build_main_tables.py` | 0 | generated_main_tables=8 |
| `python3 scripts/tkde_visual_rebuild/build_curated_supplement_tables.py` | 0 | Wrote 43 curated portrait table fragments; verified 22 claim scopes and raw row counts 180/198/840/208; manifest=results/tkde_visual_rebuild/CURATED_SUPPLEMENT_TABLE_MANIFEST.csv |
| `python3 scripts/tkde_rebuild/make_figures.py` | 0 | provenance: results/tkde_visual_rebuild/FIGURE_DATA_PROVENANCE.csv |
| `python3 scripts/tkde_rebuild/build_bibliography.py` | 0 | wrote 50 verified BibTeX entries |
| `bash scripts/tkde_rebuild/compile_papers.sh` | 0 | Main and supplement compiled with clean citation/cross-reference logs. |
| `python3 scripts/tkde_visual_rebuild/audit_table_readability.py --root . --report-dir results/tkde_visual_rebuild/clean_room_audits --strict` | 0 | tables=51 errors=0 warnings=0 |
| `python3 scripts/tkde_visual_rebuild/audit_pdf_layout.py --root . --report-dir results/tkde_visual_rebuild/clean_room_audits --strict` | 0 | documents=2 pages=44 errors=0 warnings=0 |
| `python3 scripts/tkde_visual_rebuild/scientific_delta_gate.py --root . --report-dir results/tkde_visual_rebuild/clean_room_audits --strict --skip-baseline-archives` | 0 | scientific_delta_errors=0 warnings=0 frozen_rows=36 |
