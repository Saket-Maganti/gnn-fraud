# Fourth-Review Command Log

All historical evidence inspection is read-only. No training, fitting, target
metric/oracle calculation, dataset download, Kaggle launch, baseline
installation, saved-output pilot execution, PR merge, or force-push is
authorized.

| Phase | Command or check | Outcome |
|---|---|---|
| Governing prompt | `wc -l` and complete segmented `sed` reads of `COREGRAPH_FOURTH_REVIEW_CANONICAL_EVIDENCE_RECOVERY.md` | Read all 820 lines before editing. |
| Repository preflight | `git status --short --branch`, `git rev-parse HEAD`, `git branch --show-current`, `git remote -v` | Clean `codex/coregraph-iclr-buildout-2026` at `5ecc0f17869038dce872c8ab14f806163e1fe400`. |
| Remote preflight | `git fetch origin codex/coregraph-iclr-buildout-2026` and remote-tip comparison | Remote branch is also `5ecc0f17869038dce872c8ab14f806163e1fe400`; normal fast-forward work can proceed. |
| Frozen boundary | `python scripts/coregraph/hash_frozen_assets.py --verify` | `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. |
| Existing implementation inspection | Targeted `rg` and `sed` reads of converter, pilot manifest, leakage audit, prior ledgers, and V4 reports | Confirmed FR-01 through FR-08 entry points before implementation. |
| Canonical inventory | Parsed `$HISTORICAL_GNN_FRAUD_REPO/results/runs_rb09v3/{ARTIFACT_FAMILY.json,predictions_manifest.json,runs.csv}` | Exactly 180 run rows and 180 role-neutral prediction references across 2 datasets, 3 protocols, 3 experts, and seeds 1–10. |
| Evidence search | Recursive structured discovery over `$COREGRAPH_REPO` and `$HISTORICAL_GNN_FRAUD_REPO` | 1,498 prediction-index records, 5,372 result-index/JSONL records, 31 evidence locks, 147 import/alias sources, 95 validation reports, 72 result sidecars, and 6 JSONL sources inspected. |
| Archive reconciliation | Joined the RB09v3 inventory to RB15/RB16 import manifests and searched exact archive names/checksums across both roots | All 180 members map to six checksum-locked archives; zero of the six archives is locally present. |
| Raw fallback | Content/filename navigation scan after indexed discovery | 201 same-coordinate CSV candidates found; none has an authoritative checksum or alias link permitting substitution for RB09v3. |
| V5 focused tests | `.venv/bin/python -m pytest -q tests/coregraph/test_fourth_review_v5_scenarios.py tests/coregraph/test_third_review_v4_integration.py` | Passed corrected scenario, legacy compatibility, discovery, archive, runner, gate, and no-training fixtures. |
| Canonical recovery | `.venv/bin/python scripts/coregraph/recover_canonical_manifests_v5.py --historical-root "$HISTORICAL_GNN_FRAUD_REPO"` | Wrote 180 base rows, 60 scenario rows, and 540 role bindings; verdict `COREGRAPH_CANONICAL_INDEX_REFERENCES_MISSING_LOCAL_ARCHIVES`. No training, fitting, metric, oracle, or pilot path ran. |
| Static checks | `make coregraph-compile`, `make coregraph-lint`, and `make coregraph-typecheck` with `.venv/bin/python` | Compile and Ruff passed; mypy passed for 57 source files. |
| Coverage repair | First full coverage run exposed V5 gate coverage at 82% versus the existing 85% threshold | Added positive, blocked-input, and missing-input V5 gate-path tests; the next run reached 85%. |
| Full deterministic suite | `make coregraph-local-gates PY=.venv/bin/python` | 205 tests passed. Critical coverage: contracts 96%, routing 94%, objectives 86%, pilot 87%, V4 conversion 86%, V5 scenarios 81%, canonical recovery 88%, registry 86%, leakage 86%, statistics 93%, evidence 88%, contract splits 96%, gate 85%, runner 58% with the unauthorized empirical branch retained. |
| Theory/status | `check_theory_numerically.py` and `validate_theory_status.py` | Passed three proved results and the declared XOR counterexample; no empirical statistic ran. |
| Notebooks/paper | `validate_notebooks.py` and `validate_paper_skeleton.py` | 12 notebooks passed; eight empirical claims remain blocked placeholders. |
| Synthetic/smoke | `run_synthetic_method_checks.py` and `run_coregraph_smoke.py` | Ten deterministic synthetic mechanisms and one CPU epoch passed; no provider data was used. |
| Anonymous release | `build_anonymous_release.py` and `audit_anonymous_release.py` | Deterministic 257-file package passed package tests and identity/path audit; `.DS_Store` is now excluded. |
| Frozen boundary after implementation | `.venv/bin/python scripts/coregraph/hash_frozen_assets.py --verify` | `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. |
| Focused commits | Normal `git commit` on the existing branch | `4b4c01a` artifact/scenario model; `c52aaad` runner/gate integration; `6702a50` canonical recovery. No force-push. |
| PR state before handoff | `gh pr view 2 --json ...` | PR #2 is open, draft, unmerged, and targets the expected branch. |

Push, PR-body update, and exact-tip hosted CI are recorded in the fifth-review
handoff after they complete.
