# CoReGraph independent-audit command log

All commands are run from the repository root. This log records only the
deterministic repair and validation pass; forbidden heavy actions are not run.

| Phase | Command or check | Result |
|---|---|---|
| Preflight | Read all 480 lines of `COREGRAPH_INDEPENDENT_AUDIT_METHOD_AND_PILOT_REPAIR.md` | PASS; treated as governing specification |
| Preflight | Read GitHub publish workflow and inspect `git status -sb`, branch, tip, remote | PASS; clean branch at specified tip |
| Preflight | `gh --version`; `gh auth status`; `gh pr view 2` | PASS; authenticated; PR #2 open and draft |
| Frozen boundary | `.venv/bin/python scripts/coregraph/hash_frozen_assets.py --verify` | PASS; `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |
| Regression wave | Focused independent-audit tests | PASS; contracts, availability, routing, score/abstention/objective, manifests/pilot/statistics, theory/support, and synthetic repairs |
| Full tests | `.venv/bin/python -m pytest -q` | PASS; 158 tests |
| Anonymous package tests | Build package, then run `pytest` inside the package through `audit_anonymous_release.py` | PASS; 131 tests |
| Coverage | `coverage run --source=coregraph -m pytest -q`; separate fail-under reports | PASS; all coregraph 85%; contracts 96%; routing 96%; objectives 86%; pilot 90%; support 88%; splits 96% |
| Typed core | Expanded mypy surface across 50 scientific-core and pilot/statistics modules | PASS; zero issues |
| Lint | `ruff check coregraph scripts/coregraph tests/coregraph` | PASS |
| Analysis freeze | `python scripts/coregraph/freeze_analysis_plan.py` | PASS; V2 outcome families frozen before any empirical outcomes |
| Theory | Numerical checker and theorem-status validator | PASS; exact formula, near-tight case, above-bound rejection, XOR counterexample, three proved scoped results |
| Synthetic | `python scripts/coregraph/run_synthetic_method_checks.py` | PASS; ten required deterministic qualitative scenarios; no real data or multi-seed run |
| CPU smoke | `python scripts/coregraph/run_coregraph_smoke.py` | PASS; one epoch, 96 examples, finite gradients/predictions, no provider data |
| Notebooks | `python scripts/coregraph/validate_notebooks.py` | PASS; 12 notebooks |
| Paper | `build_iclr_paper.sh`; skeleton audit; warning scan | PASS; seven-page placeholder PDF, no TeX warnings, empirical claims remain blocked |
| Anonymous release | Build plus import/checksum/anonymity/package-test audit | PASS |
| Public tree | Validate sanitized intended tracked/untracked tree, excluding ignored local tooling | PASS; zero findings |
| Frozen boundary | `python scripts/coregraph/hash_frozen_assets.py --verify` after final local gates | PASS; `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |

## Explicit non-execution record

- Official baselines installed: **no**
- Datasets downloaded: **no**
- Real saved predictions connected: **no**
- Saved-output pilot executed: **no**
- Kaggle launched: **no**
- Multi-seed experiment executed: **no**
- Frozen FraudShiftBench/TKDE assets modified: **no**

Subsequent deterministic gate commands and exact results are appended as each
repair wave completes.
