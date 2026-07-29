# Remote repository baseline

## Repository identity

- Repository: `Saket-Maganti/gnn-fraud`
- URL: `https://github.com/Saket-Maganti/gnn-fraud`
- Visibility: public
- Default branch: `main`
- Remote main HEAD: `9e5f68f795f4cfce939ab2965702fd2dc7eedf08`
- GitHub-reported repository disk usage: 14,606 KiB
- Current main tree: 93 files, 5,527,414 aggregate blob bytes
- Licence detected by GitHub: none
- Actions workflow runs returned by GitHub: none

## Recent commits

| Commit | Date | Summary |
| --- | --- | --- |
| `9e5f68f795f4` | 2026-04-18 | Delete gnnpaper directory |
| `8092f7ed5ea3` | 2026-04-18 | Merge origin/main: supersede stale deletion |
| `1c7bd187d830` | 2026-04-18 | Update README to match 10-seed results and current thesis |
| `efd137c236aa` | 2026-04-18 | Paper polish pass |
| `2c6d2bc2c315` | 2026-04-18 | Reviewer-grade rigor pass |
| `1a042114a2dc` | 2026-04-18 | Ten-seed rerun and honest manuscript rewrite |
| `41cf2c6e7936` | 2026-04-14 | Delete gnnpaper directory |
| `6877c699f7ae` | 2026-04-12 | Add leakage-gap experiment and paper refinement |
| `64c55a24cf59` | 2026-04-11 | Add transductive-vs-inductive experiment |
| `ddb10369bc59` | 2026-04-09 | Restructure repo for paper-aligned public release |

## Current top-level tree

`.DS_Store`, `.gitignore`, `.python-version`, `README.md`,
`colab_setup.sh`, `config.py`, `data/`, `experiments/`, `figures/`,
`logs/`, `models/`, `notebooks/`, `requirements.txt`, `run_insights.sh`,
`run_overnight.sh`, `run_upgrades.sh`, `scripts/`, `train.py`,
`train_proper.py`, and `utils/`.

The current remote is the earlier Elliptic-focused public state. It does not
contain the completed FraudShiftBench TKDE manuscript, final PDFs, typed
evidence/claim system, or final visual-rebuild reports.

## Curated branch delta before validation

The clean sibling clone was created from the exact remote HEAD on
`codex/curated-fraudshiftbench-2026`.

- 383 new non-ignored files;
- 9 modified tracked files;
- 1 removed file (`.DS_Store`);
- 83 remote files preserved unchanged.

Major additions by top-level path:

| Path | New files | Purpose |
| --- | ---: | --- |
| `results/` | 145 | Frozen aggregates, evidence/claim maps, readiness and publication audits |
| `paper_tkde/` | 102 | Authoritative active LaTeX dependency closure |
| `scripts/` | 37 | Final analysis, GraphSafe, publication, and audit code |
| `fraudshiftbench/` | 14 | Reusable benchmark contracts, metrics, evidence, and claim gates |
| `models/` | 10 | Current modern/temporal/theory/GraphSafe modules |
| `experiments/` | 10 | Current multi-dataset and protocol runners |
| `docs/` | 10 | Reader, reproduction, evidence, resource, analysis, and validation guides |
| `tests/` | 7 | Dataset-free curated tests |
| `manuscript_assets/` | 7 | Frozen V22 aggregate source tables |
| `data/` | 7 | Unified loaders and scaling code, no raw data |
| `kaggle/` | 6 | Portable runbooks and notebook generators, no generated notebooks |
| `release/` | 5 | Checksums, clean-room report, and safe release manifests |
| `paper/` | 2 | Final main and supplement PDFs |

Modified tracked files are `.gitignore`, `README.md`, `config.py`,
`data/README.md`, `data/dataset.py`,
`notebooks/colab_transductive_vs_inductive.ipynb`, `requirements.txt`,
`run_upgrades.sh`, and `utils/metrics.py`.

## Remote-only content preserved

All 28 earlier Elliptic figure PNGs under `figures/`, `logs/.gitkeep`, the
legacy addition scripts, original Elliptic experiment runners, original legacy
models/trainers, and compatible root reproduction entry points remain in the
branch. Documentation labels them as legacy `origin/main` surfaces rather than
canonical TKDE evidence.

## Merge risks

- The PR is intentionally large because it introduces the final manuscript and
  aggregate evidence closure into a previously small repository.
- The README and project framing change from the earlier single-dataset paper to
  FraudShiftBench; reviewers should verify that the preserved legacy surface is
  labeled clearly.
- The repository has no resolved project-wide licence. The branch adds
  `LICENSE_REVIEW_REQUIRED.md` and grants no new permission.
- GitHub Actions is new to the remote and must be observed on the draft PR.
- Final PDFs and the 18.6 MiB scalar provenance CSV are ordinary Git files but
  remain well below GitHub's 100 MiB limit.
- Raw data, raw predictions, checkpoints, environments, provider workspaces,
  backups, and release ZIPs are not part of the branch.
