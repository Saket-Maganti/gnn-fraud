# CoReGraph remote branch preflight

Date: 2026-07-29
Repository: `Saket-Maganti/gnn-fraud`

## Refs and ancestry

- Audited local HEAD before adding this report:
  `d9789d29796aecb8d9f05f9740da8866375281f5`
- Intended base SHA:
  `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`
- Merge base:
  `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`
- Ahead/behind relative to
  `origin/codex/curated-fraudshiftbench-2026`: `2/0`
- Base-ancestor check: `PASS`
- Remote head before publication: absent
- Expected PR base: `codex/curated-fraudshiftbench-2026`
- Expected PR head: `codex/coregraph-iclr-buildout-2026`

The two audited commits are:

1. `6163e387f1e3f33bfbf74079ce5fe30fc7f74b65` — CoReGraph pre-run
   research-system build.
2. `d9789d29796aecb8d9f05f9740da8866375281f5` — narrow public-tree
   safety repair.

## Diff scope

Before this report was added, the base-to-head diff contained 475 changed
files, 46,207 insertions, and 2 deletions. Adding this report is the only
expected additional pre-push change.

Top-level changed-file counts before this report:

| Path | Files |
|---|---:|
| `release/` | 225 |
| `coregraph/` | 83 |
| `results/` | 39 |
| `paper_iclr/` | 29 |
| `scripts/` | 23 |
| `configs/` | 22 |
| `docs/` | 18 |
| `tests/` | 12 |
| `kaggle/` | 9 |
| `runbooks/` | 5 |
| `notebooks/` | 3 |
| `external_baselines/` | 3 |
| Root/build metadata | 4 |

The `release/` count is the intentional history-free anonymous copy of the
CoReGraph package. `external_baselines/` contains only the registry, integration
status, and licence audit; no upstream source tree is present.

## Conflict and publication risk

- `git merge-tree --write-tree` completed without conflict and produced the
  audited HEAD tree.
- `git diff --check` passes.
- The intended base is a strict ancestor, so the current PR relation has no
  hidden merge conflict.
- No remote head existed at preflight, so the first push does not require a
  force-push or history rewrite.
- Provider manifests, saved predictions, official-baseline parity, and
  GraphMETRO/EERM licences remain empirical-readiness blockers. They do not
  block publication of the user-authored WIP branch under the governing
  handoff.
