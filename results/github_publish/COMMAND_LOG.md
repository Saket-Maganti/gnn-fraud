# GitHub curation command log

Date: 2026-07-29

This is a command-level record of the curation. Machine-specific paths,
authentication material, and secret-scan match values are intentionally
redacted. No training command was run.

| Phase | Command or operation | Exit | Outcome |
| --- | --- | ---: | --- |
| Remote auth | `gh auth status` | 0 | Authenticated account confirmed as `Saket-Maganti`. |
| Remote identity | `gh repo view Saket-Maganti/gnn-fraud --json …` | 0 | Existing public repository; default branch `main`; no visibility change. |
| Remote history | `git ls-remote …` and GitHub metadata queries | 0 | `origin/main` fixed at `9e5f68f795f4cfce939ab2965702fd2dc7eedf08` before curation. |
| Source inventory | `python3 scripts/github_publish/build_audit_reports.py …` | 0 | 1,595 logical source entries classified. |
| Candidate construction | `python3 scripts/github_publish/sync_curated_checkout.py …` | 0 | Explicit allowlist copied into a clean clone; final PDFs mapped to `paper/pdf/`. |
| Candidate safety | `python3 scripts/github_publish/validate_public_tree.py .` | 0 | Zero findings before validation. |
| Compile | `make PY="$PUBLICATION_PY" compile` | 0 | Passed. |
| Pytest | `make PY="$PUBLICATION_PY" test` | 0 | 27 passed. |
| Unittest | `make PY="$PUBLICATION_PY" unittest` | 0 | 7 passed. |
| Initial support attempt | `"$PUBLICATION_PY" scripts/tkde_rebuild/validate_support_relation.py` | 1 | Correctly exposed excluded raw/imported input dependency; no fallback fabricated. |
| Frozen support | `make PY="$PUBLICATION_PY" support` | 0 | 14 cases and 36 hashes passed. |
| Initial claim-gate attempt | `"$PUBLICATION_PY" scripts/check_claim_gates.py` | 1 | Expected historical documents are excluded; the final frozen PASS report was preserved. |
| Claim/safety checks | `make PY="$PUBLICATION_PY" claims` | 0 | Claim language and no-heavy-default checks passed. |
| Figures | `make PY="$PUBLICATION_PY" figures` | 0 | Eight figure/source pairs regenerated. |
| Tables/bibliography | `make PY="$PUBLICATION_PY" tables` | 0 | 8 main tables, 43 supplement tables, 50 bibliography entries. |
| Paper | `make paper` | 0 | Main 14 pages; supplement 30 pages. |
| Initial manuscript audit | `"$PUBLICATION_PY" scripts/tkde_rebuild/audit_manuscript.py` | 1 | Found stale audit-only table filenames; no manuscript or evidence defect. |
| Final manuscript audit | same command after audit inventory repair | 0 | 50 cited entries and clean source/PDF checks. |
| Visual audits | strict visual-object, table-readability, and PDF-layout commands | 0 | 72 objects and 44 PDF pages; zero errors/warnings. |
| Baseline delta | `scientific_delta_gate.py --strict` in source tree | 0 | Both excluded baseline ZIP hashes passed. |
| Public delta | `make PY="$PUBLICATION_PY" delta` | 0 | `ZERO_SCIENTIFIC_DELTAS`. |
| Final PDF hashes | `shasum -a 256 …` | 0 | Public copies equal authoritative final PDFs. |
| Post-test working-tree audit | `make PY="$PUBLICATION_PY" public-audit` | 1 | Correctly flagged ignored caches created by compile/test; CI was ordered to audit the clean checkout before tests. |
| Cleaned-tree audit | same command after moving generated caches out of the checkout | 0 | Zero findings. |

Final staged-object, push, PR, and post-push verification entries are recorded
in the authoritative `GITHUB_PUSH_REPORT.md` after publication.
