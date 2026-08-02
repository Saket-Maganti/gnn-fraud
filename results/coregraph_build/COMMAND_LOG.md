# CoReGraph build command log

This log records commands that can affect reproducibility or validate a build
gate. Read-only source inspection commands are summarised rather than copied
verbatim. No provider download, full dataset run, GPU run, Kaggle execution, or
multi-seed campaign is permitted.

| Time (Asia/Kolkata) | Command or action | Purpose | Result |
|---|---|---|---|
| 2026-07-29 | Read all 2,206 lines of the governing master prompt | Establish controlling specification | Complete |
| 2026-07-29 | `git status --short`; branch/remotes/log inspection in authoritative checkout | Protect user work and identify base | Authoritative checkout dirty and on unborn branch; left untouched |
| 2026-07-29 | `git fetch --all --prune` | Obtain curated branch reference | Complete |
| 2026-07-29 | Verify commit `2dec25eac1d7a8951f9d4639f49e889c4c9ca486` | Confirm immutable base | Commit exists on `origin/codex/curated-fraudshiftbench-2026` |
| 2026-07-29 | `git worktree add -b codex/coregraph-iclr-buildout-2026 <COREGRAPH_WORKTREE> 2dec25e...` | Isolate CoReGraph development | Complete |
| 2026-07-29 | Read-only inventory and targeted source inspection | Audit protocols evidence loaders models harness metrics CI and paper boundary | Complete; issues recorded in risk register |
| 2026-07-29 | `.venv/bin/python -m pytest -q` before CoReGraph edits | Establish frozen regression baseline | 27 passed |
| 2026-07-29 | `scripts/coregraph/hash_frozen_assets.py --write` | Freeze TKDE scientific boundary | 249 tracked files hashed |
| 2026-07-29 | `git ls-remote` and official repository/licence inspection | Pin external baseline revisions | Six exact revisions recorded; two licence blockers identified |
| 2026-07-29 | `scripts/coregraph/generate_run_matrices.py` | Materialise all future campaigns | 1,770 master; 1,400 screening; 1,050 final; dedicated ablation/synthetic/GOOD/fraud/resource grids |
| 2026-07-29 | `scripts/coregraph/freeze_analysis_plan.py` | Freeze matrices and statistical families | Aggregate SHA-256 recorded |
| 2026-07-29 | `scripts/coregraph/generate_notebooks.py`; `validate_notebooks.py` | Generate and statically validate orchestration notebooks | 9 Kaggle T4x2 and 3 local notebooks; pass |
| 2026-07-29 | `scripts/coregraph/run_saved_output_pilot.py` (without `--execute`) | Dry-run saved-output discovery | Planned; zero manifests; no result computed |
| 2026-07-29 | `scripts/coregraph/evaluate_pilot_gate.py` | Evaluate preregistered pilot readiness | `BLOCKED_INCOMPLETE_METHOD_ROWS` |
| 2026-07-29 | `scripts/coregraph/validate_dataset_manifests.py --all` | Check real-data prerequisites without downloads | `BLOCKED`: no provider manifests staged |
| 2026-07-29 | `scripts/coregraph/validate_baseline_registry.py` | Check pins licences and parity state | Valid registry with six headline blockers |
| 2026-07-29 | `scripts/coregraph/install_official_baselines.sh --dry-run` | Print exact reviewed acquisition commands | Complete; no repository installed |
| 2026-07-29 | `scripts/coregraph/build_iclr_paper.sh` | Compile result-gated anonymous skeleton | 7-page local placeholder PDF built |
| 2026-07-29 | Poppler render and page-by-page inspection | Visual PDF QA | 7/7 pages pass |
| 2026-07-29 | `make coregraph-local-gates PY=.venv/bin/python` | Execute complete no-heavy local gate | Pass |
| 2026-07-29 | `coverage run --source=coregraph -m pytest -q` | Measure implementation coverage | 94 tests passed; 77% statement coverage |
| 2026-07-29 | Anonymous package build/audit and packaged test run | Validate history-free release | Identity audit pass; 67 packaged tests pass |
| 2026-07-29 | `scripts/coregraph/hash_frozen_assets.py --verify` | Recheck frozen boundary | `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |
| 2026-07-29 | Exported-tree public-safety audit and narrow scanner/path repair | Remove private worktree paths and distinguish source modules from prediction payloads | Zero public-tree findings after repair; local gates and frozen boundary pass |
