# CoReGraph second-review command log

All commands run from the repository root. Forbidden empirical and heavy
actions are intentionally absent.

| Phase | Command or check | Result |
|---|---|---|
| Governing prompt | Read all 152 lines of `COREGRAPH_SECOND_REVIEW_PILOT_SEMANTICS_CLOSURE.md` | PASS |
| Repository | `git status -sb`; `git rev-parse HEAD`; fetch and compare origin | PASS; clean and synchronized at specified tip |
| GitHub | `gh auth status`; `gh pr view 2` | PASS; authenticated; PR #2 open and draft |
| Frozen boundary | `.venv/bin/python scripts/coregraph/hash_frozen_assets.py --verify` | PASS; `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |
| Audit trail | Create findings, repair plan and command log before implementation edits | PASS |

## Explicit non-execution record

- Official baselines installed: **no**
- Datasets downloaded: **no**
- Real prediction manifests created or connected: **no**
- Saved-output pilot executed: **no**
- Kaggle launched: **no**
- Multi-seed experiment executed: **no**
- PR #2 merged: **no**
- Frozen FraudShiftBench/TKDE assets modified: **no**

Deterministic commands and exact results will be appended as the repair waves
complete.

## Regression-first record

| Phase | Command or check | Result |
|---|---|---|
| Red tests | Focused second-review pilot, gate, and documentation regressions before fixes | EXPECTED FAIL; 14 defects reproduced |
| Focused repairs | Second-review and affected independent-audit tests | PASS; 36 focused tests |
| Static checks | Ruff and expanded mypy over 50 source files | PASS; zero findings |
| Full suite | `.venv/bin/python -m pytest -q` | PASS; 176 tests |
| Critical coverage | `make PY=.venv/bin/python coregraph-coverage` | PASS; every declared surface at least 85% |
| Full deterministic gates | `make PY=.venv/bin/python coregraph-local-gates` | First run correctly exposed missing release specifications; repaired allowlist; final run PASS |
| Paper | Build placeholder PDF, skeleton audit, and clean log audit | PASS; 8 pages, 8 empirical claims blocked, no warnings |
| Anonymous package | Build and audit history-free package | PASS; 242 files and 149 CoReGraph tests |
| Public tree | Validate the exact staged/index tree in an isolated temporary directory | PASS; zero findings |
| Analysis freeze | `scripts/coregraph/freeze_analysis_plan.py` | PASS; aggregate `3aa453c2563e768a53b16d8d053e51011eb852d20b411c88a7424e0d2e546614` |
| Frozen boundary | `scripts/coregraph/hash_frozen_assets.py --verify` after repairs | PASS; `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |

Repair commits:

- `8659866`: frozen decision, oracle, budget, capacity, availability, baseline
  and seed semantics;
- `5dc99d5`: exact coverage, corrected inference, meaningful ablations and
  matched-contract worst-case gate.

Draft-PR CI and the PR-body update are pending the first normal push.

## First CI feedback

The first normal push started duplicate push/PR workflows. Both audit jobs
passed. One typed-core job exposed a pre-existing platform-sensitive exact
float assertion (`1.0000001192092896 == 1`) in the router mask test. The
assertion was repaired to use numerical tolerance; no scientific computation
or acceptance threshold changed.

The repair was revalidated locally, committed as `c81bb7f`, and pushed by
normal fast-forward. Both audit and both typed-core jobs then passed. PR #2
remained open and draft.
