# TKDE V2 visual-rebuild command log

All commands were run from the repository root. This pass was restricted to publication design, table/figure generation from frozen inputs, LaTeX compilation, visual inspection, auditing, and packaging. It did not launch model training, dataset acquisition, a Kaggle/provider job, or a new evidence/statistical analysis.

## Baseline protection

| Operation | Status | Result |
| --- | --- | --- |
| Read the complete V2 governing prompt before edits | PASS | The staged repository copy was used because the original Downloads attachment was no longer present. |
| Record the dirty worktree and preserve user-owned changes | PASS | No unrelated change was reverted or cleaned. |
| Create the sibling safety backup `gnn-fraud_tkde_visual_backup_20260710_224421_IST` | PASS | Paper, generators, reports, prompt, and repository state were preserved before the delta. |
| Verify frozen manuscript ZIP | PASS | SHA-256 `234b164d0553fd4d19b1e850c19e4b0924bfeee1a0342201f326d24d2faa2ba0`. |
| Verify frozen source/table ZIP | PASS | SHA-256 `528780b5666dd1b97f0a12be39e20957af4a08f25de8d7b334179727be4d919c`. |

## Publication-asset regeneration

The final generator sequence was:

```bash
gnn_env/bin/python scripts/tkde_visual_rebuild/build_main_tables.py
gnn_env/bin/python scripts/tkde_visual_rebuild/build_curated_supplement_tables.py
gnn_env/bin/python scripts/tkde_rebuild/make_figures.py
gnn_env/bin/python scripts/tkde_rebuild/build_bibliography.py
bash scripts/tkde_rebuild/compile_papers.sh
```

| Command | Exit | Final result |
| --- | ---: | --- |
| `build_main_tables.py` | 0 | Eight readable main-paper table fragments and their source/provenance CSVs. |
| `build_curated_supplement_tables.py` | 0 | Forty-three portrait supplementary tables; 22 claim scopes and raw row totals 180/198/840/208 verified. |
| `make_figures.py` | 0 | Eight PDF/PNG figure pairs, eight source CSVs, and a visual-rebuild provenance manifest. |
| `build_bibliography.py` | 0 | Fifty verified BibTeX entries. |
| `compile_papers.sh` | 0 | Strict BibTeX cycles for main and supplement with clean citation and cross-reference logs. |

The legacy `scripts/tkde_rebuild/build_tables.py` entry point was hardened to delegate to the V2 table builders, so a future invocation cannot restore the retired raw-row/tiny-table layout.

## Object, table, PDF, and scientific-delta audits

```bash
gnn_env/bin/python scripts/tkde_visual_rebuild/audit_visual_objects.py --strict
gnn_env/bin/python scripts/tkde_visual_rebuild/audit_table_readability.py --strict --warnings-as-errors
gnn_env/bin/python scripts/tkde_visual_rebuild/audit_pdf_layout.py --strict --warnings-as-errors
gnn_env/bin/python scripts/tkde_visual_rebuild/scientific_delta_gate.py --strict
```

| Command | Exit | Final result |
| --- | ---: | --- |
| `audit_visual_objects.py --strict` | 0 | 72 frozen objects reconciled; 0 problems. |
| `audit_table_readability.py --strict --warnings-as-errors` | 0 | 51 active tables; 0 errors; 0 warnings. |
| `audit_pdf_layout.py --strict --warnings-as-errors` | 0 | 2 documents, 44 pages; 0 errors; 0 warnings. |
| `scientific_delta_gate.py --strict` | 0 | 36 frozen-hash rows checked; 0 errors; 0 warnings; `ZERO_SCIENTIFIC_DELTAS`. |

The scientific-delta gate also reconciled 72 allocation rows, 8 figure-provenance rows, 51 table-provenance rows, 22 typed claims, 50 verified/cited references, 41 blocked-semantics CSVs, and both frozen archives.

## Tests and code-quality gates

```bash
gnn_env/bin/python -m pytest -q
gnn_env/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
gnn_env/bin/python -m pytest -q tests/test_tkde_visual_release_audits.py
gnn_env/bin/ruff check scripts/tkde_visual_rebuild scripts/tkde_rebuild/make_figures.py tests/test_tkde_visual_release_audits.py
gnn_env/bin/python -m compileall -q scripts/tkde_visual_rebuild scripts/tkde_rebuild/make_figures.py tests/test_tkde_visual_release_audits.py
gnn_env/bin/python scripts/audit_claim_language.py
gnn_env/bin/python scripts/check_claim_gates.py
gnn_env/bin/python scripts/safety_check_no_heavy_defaults.py
gnn_env/bin/python scripts/check_anonymization.py --scan-root paper_tkde --json-output results/tkde_visual_rebuild/validation/anonymization.json --markdown-output results/tkde_visual_rebuild/validation/anonymization.md --fail-on-paper-leak
pdfinfo paper_tkde/main.pdf
pdfinfo paper_tkde/supplement/supplement.pdf
```

| Gate | Exit | Final result |
| --- | ---: | --- |
| Canonical Pytest suite | 0 | 954 tests passed; two pre-existing nonfatal warnings (PyG deprecation and SciPy precision loss in a near-identical fixture). |
| Corrected unittest discovery | 0 | 831 tests in 234.384 s; OK. |
| V2 release/audit tests | 0 | 13 tests passed. |
| Ruff | 0 | PASS. |
| Compileall | 0 | PASS. |
| Claim-language audit | 0 | PASS; no findings. |
| Claim gates | 0 | PASS; zero issues. |
| Heavy-default safety | 0 | PASS; zero issues. |
| Anonymization report | 0 | `double_blind_ok=true`; zero high-severity paper-source findings. PDF metadata was also checked manually because the report-only scanner lacked a PDF metadata library. |

One noncanonical unittest invocation without `-t .` produced 78 import-collection errors for nested `tkde_max.*` modules. The corrected repository-root discovery command above passed all 831 tests and is the reported gate. One early focused-Pytest command used the obsolete filename `tests/test_tkde_visual_release.py`; rerunning against `tests/test_tkde_visual_release_audits.py` passed 13 tests.

## Page rendering and human visual inspection

Both PDFs were rendered at 200 dpi into `results/tkde_visual_rebuild/final_pages/`. Color and grayscale contact sheets were generated under `results/tkde_visual_rebuild/final_contact_sheets/`.

All 14 main-paper pages and all 30 supplement pages were inspected at full-page, print-scale, and contact-sheet views in color and grayscale. The final repairs addressed a wrapped contract label, two cropped forest/rank labels, one panel-title/x-label collision, a protected right tick, and bibliography-column balance. Both documents were recompiled and rerendered after the last repair.

## Release and clean-room terminal gate

```bash
gnn_env/bin/python scripts/tkde_visual_rebuild/build_visual_release.py --python "$PWD/gnn_env/bin/python"
gnn_env/bin/python scripts/tkde_visual_rebuild/build_visual_release.py --check-only --python "$PWD/gnn_env/bin/python"
unzip -tqq release/tkde_visual_rebuild/tkde_visual_manuscript_package.zip
unzip -tqq release/tkde_visual_rebuild/tkde_visual_source_analysis_package.zip
```

The first packaging probe passed a relative interpreter (`gnn_env/bin/python`) into the extracted clean-room working directory, where that relative path did not exist. The final invocation passed a repository-resolved interpreter, while the publisher was adjusted to record host-neutral `python3` commands in the clean-room report. The final build and `--check-only` pass both succeeded.

A final member-content inspection found that the JSON output of the local anonymization scanner included the identity tokens it had searched for. The release builder was hardened to reject identity-bearing content and to exclude that scanner-configuration JSON from both anonymous archives. The identity-free Markdown result and the manual PDF-metadata outcome remain available; focused release tests and the complete package validation were rerun after this repair.

| Gate | Exit | Final result |
| --- | ---: | --- |
| Deterministic namespaced packaging | 0 | Manuscript and source-analysis ZIPs created. |
| New-directory clean-room rebuild | 0 | Generators, bibliography, strict compilation, table/PDF audits, and scientific-delta gate all PASS. |
| Archive `--check-only` | 0 | ZIPs match manifests; frozen archives unchanged. |
| ZIP CRC tests | 0 | Both archives PASS. |
| Release hygiene | 0 | No credentials, private absolute paths, raw datasets, prediction payloads, caches, macOS metadata, auxiliary files, or stale assets. |

Exact current package, PDF, report, and manifest hashes are recorded outside the ZIP members in `release/tkde_visual_rebuild/CHECKSUMS.sha256`, avoiding archive self-reference.
