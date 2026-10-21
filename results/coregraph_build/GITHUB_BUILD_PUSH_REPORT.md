# GitHub build/push report

Final verdict:
`COREGRAPH_WIP_BRANCH_PUSHED_READY_FOR_INDEPENDENT_ANALYSIS`

## Publication

- Repository: `Saket-Maganti/gnn-fraud`
- Branch: `codex/coregraph-iclr-buildout-2026`
- Audited implementation SHA, local and remote:
  `7167d9b1604d1c896559704ffb8e8e244bc89113`
- Original CoReGraph build commit:
  `6163e387f1e3f33bfbf74079ce5fe30fc7f74b65`
- Base branch: `codex/curated-fraudshiftbench-2026`
- Base SHA: `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`
- Merge base: `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`
- Draft PR: [#2](https://github.com/Saket-Maganti/gnn-fraud/pull/2)
- PR state: open, draft, mergeable; no merge was requested or performed.
- Push mode: normal fast-forward/new-branch pushes only; no force-push and no
  update to `main`.

The implementation SHA above is the immutable code/preflight revision audited
before this documentation-only publication record. Because a commit cannot
embed its own SHA, the exact final report/handoff commit is recorded in the PR
and final task response after the last push.

## Deterministic validation

- Compilation: pass.
- Ruff `F,I`: pass.
- Typed core: 23 source files, zero mypy issues.
- Repository tests: 94 passed.
- `coregraph/` coverage: 2,680/3,484 statements, 76.923% (reported as 77%).
- Anonymous-package tests: 67 passed.
- Theory: three proved-result numerical checks and theorem-status gate pass.
- Notebook validation: 12/12 pass.
- CPU smoke: one epoch; feature expert, factorised router gradient, and sampled
  GCN pass; 96 fixture examples and no provider data.
- ICLR paper: seven-page placeholder PDF builds; skeleton and claim-source
  audits pass; eight empirical claims remain blocked.
- Anonymous release: 223-file manifest, import smoke, checksums, identity/path
  audit, and packaged tests pass.
- `git diff --check`: pass.

## CI

On implementation SHA `7167d9b1604d1c896559704ffb8e8e244bc89113`:

- `curated-no-training-ci` run 6: success.
- `coregraph-no-heavy-ci` run 2: success.

The sole annotation is GitHub's non-failing notice that `actions/checkout@v4`
and `actions/setup-python@v5` declare the deprecated Node.js 20 runtime and are
currently forced onto Node.js 24. No deterministic CI failure is unexplained.

## Public safety and staged content

- Exported committed-tree scanner: zero findings after the documented narrow
  private-path/scanner repair.
- Secret and credential patterns: none.
- Private absolute paths: none.
- Raw data and raw prediction payloads: none.
- Archive, cache, environment, symlink, and forbidden metadata paths: none.
- Anonymous identity/path/checksum audit: pass.
- Tracked files over 100 MiB: zero; largest tracked file is 19,502,682 bytes.
- New Git objects over 100 MiB: zero.
- Index-level scans and staged `git diff --check`: pass before each publication
  commit.

## Licence and third-party reuse

- Mowst: pinned MIT licence file verified; no source vendored.
- CIGA: pinned MIT licence file verified; no source vendored.
- official TGN: pinned Apache-2.0 licence file verified; no source vendored.
- GOOD: pinned GPL-3.0 licence file verified; external-process boundary only
  and no source vendored.
- GraphMETRO: pinned commit resolves, but no licence file exists; remains
  `UNAVAILABLE_LICENSE`.
- EERM: pinned commit resolves, but no licence file exists; remains
  `UNAVAILABLE_LICENSE`.

The branch's added Git blobs were compared with all six pinned upstream trees.
There are no exact third-party source-file matches. The only overlapping blob
is Git's universal zero-byte blob used for empty `__init__.py`/placeholder
files. Added code contains no upstream copyright, SPDX, or copied-source
header, and `external_baselines/` contains only three metadata/audit files.
No GraphMETRO, EERM, or other external-baseline source code was copied into the
branch.

## Frozen FraudShiftBench/TKDE status

- Frozen checker: `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.
- All protected evidence, locks, results, tables, figures, manuscript sources,
  prediction manifests, and final PDFs match byte-for-byte.
- No frozen output was regenerated.

## Diff and remaining blockers

- Base-to-implementation/preflight diff: 476 files, 46,275 insertions, and 2
  deletions.
- The independent-analysis handoff adds one new documentation path; the final
  report updates this existing path.
- Provider manifests: not staged.
- Saved prediction manifests: absent.
- Saved-output pilot: not executed; prior readiness artifact remains blocked.
- Mowst/CIGA/TGN/GOOD official parity: pending.
- GraphMETRO/EERM reuse licences: unavailable.
- Final pilot go/no-go: pending.
- No baseline was installed, no dataset was downloaded, and no Kaggle,
  multi-seed, or provider experiment was launched.

Independent-analysis handoff:
`results/coregraph_build/CHATGPT_COREGRAPH_GITHUB_ANALYSIS_HANDOFF.md`
