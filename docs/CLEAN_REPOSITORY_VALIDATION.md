# Clean repository validation

Validation was performed from the curated checkout on 2026-07-29. No model
training, dataset download, GPU job, or mutation of locked evidence occurred.
`PUBLICATION_PY` below denotes the compatible Python environment used for the
publication stack; the machine-specific interpreter path is intentionally not
recorded in this public file.

## Results

| Check | Command | Exit | Result |
| --- | --- | ---: | --- |
| Python syntax | `make PY="$PUBLICATION_PY" compile` | 0 | Selected source trees compiled. |
| Curated pytest | `make PY="$PUBLICATION_PY" test` | 0 | 27 passed. |
| Corrected unittest discovery | `make PY="$PUBLICATION_PY" unittest` | 0 | 7 unittest-style tests passed. |
| Frozen support relation | `make PY="$PUBLICATION_PY" support` | 0 | 14 support cases and 36 saved hash checks passed. |
| Claim-language audit | `"$PUBLICATION_PY" scripts/audit_claim_language.py` | 0 | Zero findings. |
| Heavy-default safety | `"$PUBLICATION_PY" scripts/safety_check_no_heavy_defaults.py --output-dir results/tkde_visual_rebuild/validation/safety` | 0 | Zero issues. |
| Figure regeneration | `make PY="$PUBLICATION_PY" figures` | 0 | Eight figure/source pairs regenerated from curated aggregates. |
| Table and bibliography regeneration | `make PY="$PUBLICATION_PY" tables` | 0 | Eight main tables, 43 supplement tables, and 50 verified bibliography entries regenerated. |
| Paper build | `make paper` | 0 | Main and supplement built successfully: 14 and 30 pages. |
| Manuscript source/PDF audit | `"$PUBLICATION_PY" scripts/tkde_rebuild/audit_manuscript.py` | 0 | 50/50 references cited; no missing citations, undefined references, duplicate labels, forbidden source patterns, or Type 3 fonts. |
| Visual-object audit | `"$PUBLICATION_PY" scripts/tkde_visual_rebuild/audit_visual_objects.py --strict` | 0 | 72 objects, zero problems. |
| Table readability | `"$PUBLICATION_PY" scripts/tkde_visual_rebuild/audit_table_readability.py --strict --warnings-as-errors` | 0 | 51 tables, zero errors, zero warnings. |
| PDF layout | `"$PUBLICATION_PY" scripts/tkde_visual_rebuild/audit_pdf_layout.py --strict --warnings-as-errors` | 0 | Two documents, 44 pages, zero errors, zero warnings. |
| Baseline scientific delta | `"$PUBLICATION_PY" scripts/tkde_visual_rebuild/scientific_delta_gate.py --strict` in the authoritative source tree | 0 | Both frozen baseline ZIP hashes and all scientific invariants passed. |
| Public-package scientific delta | `make PY="$PUBLICATION_PY" delta` | 0 | 36 frozen rows, 22 typed claims, 50 cited references, 72 objects; `ZERO_SCIENTIFIC_DELTAS`. |
| Final-PDF identity | `shasum -a 256 paper/pdf/*.pdf <authoritative-final-pdfs>` | 0 | Both curated PDF hashes equal the authoritative final PDF hashes. |
| Public-tree safety | `make PY="$PUBLICATION_PY" public-audit` | 0 | Zero secret, identity, private-path, raw-data, prediction, archive, cache, and size findings. |

The final public-tree and staged-object audits are rerun immediately before
push. Their post-push outcome is recorded in
`results/github_publish/GITHUB_PUSH_REPORT.md` in the authoritative working
copy.

## Honest boundaries

- Full experiment reruns require provider datasets and, for some cells, GPUs.
  They were not run.
- Full `compute_analysis.py` reconstruction depends on excluded raw/imported
  payloads. Public validation instead checks the frozen deterministic
  regeneration audit against the included canonical aggregates.
- The two baseline release ZIPs are deliberately excluded from Git. They were
  hash-verified in the authoritative source tree before the public-package
  scientific-delta check used `--skip-baseline-archives`.
- The full TeX build is validated locally but omitted from hosted CI to keep CI
  lightweight and independent of a large TeX installation.
- Project-wide licensing remains unresolved. `LICENSE_REVIEW_REQUIRED.md`
  grants no new permission.

