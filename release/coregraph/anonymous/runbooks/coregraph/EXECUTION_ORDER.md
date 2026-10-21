# Exact execution order

Run every command from the dedicated CoReGraph worktree.

## A. Local preflight (no provider data)

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements-coregraph-lock.txt
make coregraph-local-gates
```

Stop on any failure. The command hashes frozen TKDE inputs, compiles, lints,
type-checks, runs tests, checks theory, validates notebooks, builds and audits
the anonymous package, and executes only deterministic toy/one-epoch CPU work.

## B. Saved-output pilot

```bash
.venv/bin/python scripts/coregraph/run_saved_output_pilot.py
.venv/bin/python scripts/coregraph/run_saved_output_pilot.py --execute
.venv/bin/python scripts/coregraph/evaluate_pilot_gate.py
```

`--execute` is only valid when the configured prediction roots contain aligned
manifests. If none exist, the pilot remains `BLOCKED_NO_SAVED_PREDICTIONS`.

## C. Stage data and official adapters

```bash
.venv/bin/python scripts/coregraph/validate_dataset_manifests.py --all
.venv/bin/python scripts/coregraph/validate_baseline_registry.py
bash scripts/coregraph/install_official_baselines.sh --dry-run
```

Then perform the manual provider staging and isolated installations listed by
those tools. Do not continue while a selected headline row is blocked.

## D. Freeze plans

```bash
.venv/bin/python scripts/coregraph/generate_run_matrices.py
.venv/bin/python scripts/coregraph/freeze_analysis_plan.py
git diff --exit-code -- configs/coregraph/run_matrices results/coregraph_build/ANALYSIS_PLAN_FREEZE.json
```

## E. Profile, screen, and confirm

```bash
.venv/bin/python scripts/coregraph/plan_wave.py --matrix RESOURCE_GRID.csv
.venv/bin/python scripts/coregraph/plan_wave.py --matrix SCREENING_5SEED_GRID.csv
.venv/bin/python scripts/coregraph/plan_wave.py --matrix FINAL_10SEED_GRID.csv
.venv/bin/python scripts/coregraph/plan_wave.py --matrix ABLATION_GRID.csv
```

Copy the generated lane plans into the matching Kaggle notebook. Two T4 lanes
are independent scheduler partitions, not implicit distributed training.
Final-ten-seed planning is allowed only after the frozen screening rule passes.

## F. Import, analyse, and build the paper

```bash
.venv/bin/python scripts/coregraph/import_run_outputs.py --verify-checksums
.venv/bin/python scripts/coregraph/run_statistical_analysis.py --frozen-plan results/coregraph_build/ANALYSIS_PLAN_FREEZE.json
.venv/bin/python scripts/coregraph/generate_paper_assets.py
bash scripts/coregraph/build_iclr_paper.sh
.venv/bin/python scripts/coregraph/validate_claims.py
```

## G. Release and push gate

```bash
make coregraph-local-gates
.venv/bin/python scripts/coregraph/hash_frozen_assets.py --verify
.venv/bin/python scripts/coregraph/build_anonymous_release.py
.venv/bin/python scripts/coregraph/audit_anonymous_release.py
git status --short
```

Push only when applicable gates pass and no report contains a blocker:

```bash
git push -u origin codex/coregraph-iclr-buildout-2026
```
