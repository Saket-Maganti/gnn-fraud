# V5 final-repair command log

Status: `PUBLISHED_IMPLEMENTATION_CI_PASS_PENDING_FINAL_PROVENANCE_TIP`

## Preflight

- Read all 983 lines of `COREGRAPH_V5_FINAL_REPAIR_AND_REAL_RUN_READINESS_PROMPT.md` before edits.
- `git status --short --branch`: clean expected branch.
- `git rev-parse HEAD` and remote-tip check: both `0b9839a3de095194107dd50a08cb288c70a9e166`.
- `git remote -v` and `git fetch origin --prune`: pass.
- `gh pr view 2`: open, draft, unmerged, mergeable, correct base/head.
- `git ls-files '*.zip'`: only two source-snapshot ZIPs; no canonical evidence archive.
- `python scripts/coregraph/hash_frozen_assets.py --verify`: `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.

## Implementation and focused checks

- Inspected the V5 config, preregistration, runner, types, executor, output/resume layer, paper, release builder, runbooks, and existing tests with `rg`, `sed`, and `nl`.
- Applied focused patches for matched-action regret, v5.1 preregistration, metric/output schema v2, effective execution identity, exact package validation, runner escape-hatch removal, documentation, paper, release, and tests.
- `PYTHONPATH=. pytest -q tests/coregraph/test_v5_final_repair.py tests/coregraph/test_v5_pilot_executor.py`: PASS, 91 focused cases.
- `coverage run --timid ...` plus per-file reports: PASS; executor 95%, outputs 97%, loader 90%, package validator 100%, regret evaluator 100%.

## Permitted structural and synthetic validation

- Canonical `--validate-only` against `${COREGRAPH_EVIDENCE_CACHE}`: PASS; 6 archives, 180 members, 180 artifacts, 60 scenarios, 540 bindings, 240 coordinates, no fit, no label access, no extraction.
- No-fit canonical assembly script: PASS for Elliptic and DGraphFin; bounded source assembly, float32 target scores, no target-label field, unopened vault, no fit/metric.
- Synthetic CLI at chunk size 3: PASS, 240/240 coordinates, four methods, zero failures; effective hash `f01e7729ffec24cdb9d45e23958f82a38a618457c8dc27ff39ee964281bc0d1c`.
- Synthetic metric/package scan: minimum corrected regret `0.07603882589690639`; old fields absent; expected coordinate-set SHA-256 `d5214851b7482dae2618d4cc4e1628a3746b2b21a46d009ed7cdd1fcedf56691`.
- Unchanged `--resume`: PASS with coordinate-tree SHA-256 `d4145245d3415a1409e8f1324899af55602a50bdcea9cb04d5225d8e251ce6af` before and after.
- Changed-effective-identity resume unit and mutation tests: PASS; stale outputs rejected.
- `run_saved_output_pilot_v5.py --package` followed by CRC and post-extraction exact validation: PASS; package regenerated after resume.

## Complete local quality surface

- `make PY=.venv/bin/python coregraph-compile coregraph-lint coregraph-typecheck`: PASS; mypy checked 104 source files.
- `make PY=.venv/bin/python coregraph-coverage`: PASS; all 328 repository tests and every pre-existing/new coverage gate.
- `check_theory_numerically.py`, `validate_theory_status.py`, `run_synthetic_method_checks.py`, and `run_coregraph_smoke.py`: PASS; provider data absent.
- `validate_notebooks.py`: PASS; 12 syntax-checked, zero executed.
- `validate_paper_skeleton.py`, `audit_cross_paper_overlap.py`, and `build_level4_paper.py`: PASS_RESULTS_BLOCKED; 14 main + 5 supplement pages, zero undefined references/citations, overfull boxes, Type 3 fonts, identity/private paths, or unsupported results.
- `build_anonymous_release.py` and `audit_anonymous_release.py`: PASS; anonymous package tests pass.
- `build_level4_release.py`, `validate_level4_release.py`, and `validate_level4_release.py --cleanroom`: PASS; deterministic snapshots, checksums, public-tree report, isolated tests/paper build, and zero target metrics/oracles.
- `hash_frozen_assets.py --verify`: `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` before and after repair.
- Authoritative runner option scan: PASS; no `--allow-dirty`, `--scenario-id`, `--method`, or `--dry-run`.
- A direct scanner invocation on the developer worktree was diagnostic-only and correctly rejected local environments/caches; the publishable source-snapshot and anonymous-release public-tree audits passed.

## Hard-boundary accounting

- Real pilot commands invoked: zero.
- Real target-label reads, target metrics, and target oracles: zero.
- Kaggle/GPU jobs: zero.
- Paper result insertions: zero.
- Force-pushes and PR merges: zero.

## Publish and hosted validation

- Created three focused commits: `13655ae` (regret/preregistration/effective identity), `05c7b25` (exact packaging/runner guards), and `7b50a37` (paper/runbooks/release evidence).
- Before push, `git fetch origin --prune`, status, local SHA, and remote SHA checks passed; the remote remained at the authorized starting tip.
- `git push origin codex/coregraph-iclr-buildout-2026`: normal fast-forward; no force-push.
- Updated draft PR #2 with corrected semantics, v5.1 hash, effective identity, exact packaging, dirty policy, 240/240 synthetic evidence, local gates, and explicit non-execution.
- Exact implementation tip `7b50a37ddd27f3e5ec83cd426411d4760efd84cd`: both push and pull-request audit jobs PASS; both push and pull-request typed-core jobs PASS, including compile/lint/type, full tests, theory, notebooks, paper, release, and frozen assets.
- PR #2 remained open, draft, mergeable, and unmerged.

The final provenance-only gate/tree commit is excluded from release snapshot self-hashes and will receive its own exact-tip CI before closure.
