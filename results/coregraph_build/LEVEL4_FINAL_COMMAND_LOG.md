# Level-4 final command log

All paths below are repository-relative or environment-variable based.

| Gate | Command family | Outcome |
|---|---|---|
| Git/authority preflight | `git status`, `rev-parse`, `remote`, `fetch`, `fsck` | PASS; independent checkout |
| Evidence full audit | `validate_level4_evidence_cache.py` | PASS; 6 archives, 180 members, zero extraction |
| Row-scope audit | `validate_level4_row_scopes.py` | PASS; 20 dataset-seed groups |
| Artifact construction | `build_level4_artifacts.py` | PASS; 180/60/540 |
| Pilot input validation | `orchestrate_level4.py pilot-validate` | READY; no execution |
| Full planning | `orchestrate_level4.py full-plan` | PASS; 1,680 plan rows, no jobs |
| Compile/lint/type | `compileall`, `ruff check`, `mypy` | PASS; 97 typed source files |
| Tests | `pytest -q` through coverage | PASS; 237 tests |
| Coverage | `make coregraph-coverage` | PASS; minimum 85% |
| Theory | numeric/status/standalone executable checks | PASS |
| Tiny synthetic suite | `run_synthetic_method_checks.py` | PASS; no real data |
| CPU smoke | `run_coregraph_smoke.py` | PASS; one epoch, 24 graph nodes |
| Notebook audit | `validate_notebooks.py` | PASS; 12 static, 0 executed |
| Paper/claims | skeleton, overlap, build, page QA | PASS_RESULTS_BLOCKED |
| Frozen inherited files | `hash_frozen_assets.py --verify` | ZERO_TKDE_SCIENTIFIC_DELTAS (249 files) |
| Release checksums | `validate_level4_release.py` | PASS |
| Clean room | `validate_level4_release.py --cleanroom` | PASS |

The empirical analysis target was deliberately not run because no validated
pilot result import exists. No SSD access, full training, target metric/oracle,
official baseline install, Kaggle launch, force-push, or PR merge occurred.
