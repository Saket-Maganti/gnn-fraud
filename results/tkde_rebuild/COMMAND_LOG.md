# TKDE Rebuild Command Log

All commands were run from the repository root unless a command explicitly changes directory. This is a saved-evidence rebuild: no full model-training sweep, Kaggle job, dataset download, or GPU experiment was launched.

## Repository protection and discovery

| Operation | Status | Result |
| --- | --- | --- |
| Record `git status --short --branch` and inventory the dirty, unborn worktree | PASS | Existing user-owned changes were preserved |
| Copy the prompt, paper tree, paper-generation scripts, Git status, and file manifest to the sibling safety backup `gnn-fraud_tkde_rebuild_backup_20260710_110548_IST` | PASS | 1,305 manifest rows; backup manifest SHA-256 `ff0001539c4afc9dd2855f76c9b0851afd56d11899f5a4989856eb847bb0b4c4` |
| Hash the governing prompt | PASS | SHA-256 `de28abda82186bc6825f39ada4951dbb6583e963c1997110df3178489cb61b98` |
| Guard the retired manuscript generator | PASS | It now refuses overwrite unless `--legacy-overwrite` is supplied |

## Deterministic evidence and manuscript regeneration

The final sequence was:

```bash
gnn_env/bin/python scripts/tkde_rebuild/build_evidence_inventory.py
gnn_env/bin/python scripts/tkde_rebuild/build_claim_ledger.py
gnn_env/bin/python scripts/tkde_rebuild/build_literature_assets.py
gnn_env/bin/python scripts/tkde_rebuild/compute_analysis.py
gnn_env/bin/python scripts/tkde_rebuild/validate_support_relation.py
gnn_env/bin/python scripts/tkde_rebuild/make_figures.py
gnn_env/bin/python scripts/tkde_rebuild/build_tables.py
gnn_env/bin/python scripts/tkde_rebuild/build_bibliography.py
```

| Command | Exit | Final result |
| --- | ---: | --- |
| `build_evidence_inventory.py` | 0 | 247 evidence rows |
| `build_claim_ledger.py` | 0 | 22 typed claims |
| `build_literature_assets.py` | 0 | 50 verified works; 12 novelty rows |
| `compute_analysis.py` | 0 | 6,796 scalar provenance records, including dependence-aware IBM analysis |
| `validate_support_relation.py` | 0 | 14/14 expected support cases; 36/36 deterministic hash checks |
| `make_figures.py` | 0 | 8 PDF/PNG vector-figure pairs regenerated |
| `build_tables.py` | 0 | 7 main and 23 supplementary tables regenerated |
| `build_bibliography.py` | 0 | 50 verified BibTeX records written |

The IBM sensitivity repair changed the inferential unit from 40 context--seed rows to ten seed blocks after averaging four fixed contexts within seed. It also generated `IBM_MATCHED_ABLATION_CONTEXT_EFFECTS.csv` with 208 context-specific rows. No evidence file was promoted or fabricated during this repair.

## LaTeX compilation

The strict wrapper executes the full bibliography cycle for each document:

```bash
scripts/tkde_rebuild/compile_papers.sh
```

Equivalent explicit cycles:

```bash
cd paper_tkde
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

cd paper_tkde/supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
```

| Document | Exit | Pages | Undefined citations/refs | Overfull boxes | Type 3 fonts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `paper_tkde/main.pdf` | 0 | 14 | 0 | 0 | 0 |
| `paper_tkde/supplement/supplement.pdf` | 0 | 47 | 0 | 0 | 0 |

## Tests and audits

| Command | Exit | Result |
| --- | ---: | --- |
| `python3 -m compileall -q scripts tests aaai_upgrade` | 0 | PASS |
| `gnn_env/bin/python -m pytest -q` | 0 | PASS; 941 tests collected, 100% completed |
| `gnn_env/bin/python -m unittest discover -s tests -t . -p 'test_*.py'` | 0 | PASS; 831 tests in 235.460 s |
| `gnn_env/bin/ruff check .` | 0 | PASS |
| `gnn_env/bin/python scripts/check_claim_gates.py` | 0 | PASS; `ok=True`, zero issues |
| `gnn_env/bin/python scripts/safety_check_no_heavy_defaults.py` | 0 | PASS; `ok=True`, zero issues |
| `gnn_env/bin/python scripts/audit_claim_language.py` | 0 | PASS; zero findings |
| `gnn_env/bin/python scripts/tkde_rebuild/audit_manuscript.py` | 0 | PASS; 50/50 citations used, no source-fatal finding |
| LaTeX log fatal-pattern scan | 0 | PASS; zero undefined citations/references, rerun warnings, or overfull boxes |
| `pdffonts` Type 3 scan | 0 | PASS; zero Type 3 fonts in both PDFs |

Two expected warnings remain nonfatal and documented: the installed PyG emits a deprecation warning for `torch_geometric.distributed`, and one near-identical-pairs test emits SciPy's precision-loss warning. Neither warning affects the rebuilt manuscript analyses. A test-side unclosed-file warning was repaired with a context manager.

### Repaired validation failures

The following failures were not hidden:

1. `gnn_env/bin/ruff check .` initially found one undefined `Mapping` name in `aaai_upgrade/tools/build_t4x2_stretch_upload_bundle.py`. Adding the missing import repaired the issue; the final full lint passes.
2. The workflow's original `python -m unittest discover -s tests -p 'test_*.py'` invocation imported nested tests as unresolved top-level `tkde_max.*` modules and produced 78 collection errors. Setting `-t .` in `.github/workflows/ci.yml` repaired collection; the corrected command passes 831 tests.

## PDF visual inspection

```bash
pdftoppm -png -r 95 paper_tkde/main.pdf tmp/pdfs/submission_main_pages/page
pdftoppm -png -r 85 paper_tkde/supplement/supplement.pdf tmp/pdfs/submission_supp_pages/page
```

All 14 main pages and all 47 supplement pages were inspected after the final source change. `TABLE_AND_FLOAT_PLACEMENT_AUDIT.md` records the page-level result. Temporary rendered PNGs are excluded from the release.

## Release terminal gate

The following commands are intentionally the last, self-referential packaging step after this log and the readiness report exist:

```bash
python3 scripts/tkde_rebuild/build_release.py
python3 scripts/tkde_rebuild/build_release.py --check-only
```

The script performs deterministic double-build comparison, member checksums, ZIP listing/CRC validation, private-path and credential scans, and exclusion checks. Exact final ZIP hashes live only in `release/tkde_artifact_manifest.csv`, outside the ZIP members, to avoid a self-referential archive hash.
