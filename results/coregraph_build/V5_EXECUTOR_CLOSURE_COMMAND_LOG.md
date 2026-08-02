# V5 executor closure command log

This file records closure-phase commands and outcomes. It intentionally contains no credentials, provider payloads, or private cache contents.

## 2026-08-01 preflight

- Read the 1,714-line master prompt completely.
- Verified the working directory resolved to the configured `${COREGRAPH_REPO_ROOT}` authority.
- Verified clean branch `codex/coregraph-iclr-buildout-2026` at `c879c979cb5964b55d8da56919ae90d46ac8e9e1`.
- Fetched `origin --prune`; remote branch tip is the same SHA.
- Ran `git fsck --full`; only two pre-existing dangling blobs were reported.
- Verified GitHub CLI authentication and PR #2: open, draft, unmerged, base `codex/curated-fraudshiftbench-2026`, head `codex/coregraph-iclr-buildout-2026`.
- Searched the repository, project tree, and Downloads for `COREGRAPH_LEVEL4_POST_BUILD_INDEPENDENT_AUDIT.md`; no copy is present. The master prompt's reproduced diagnosis is used as immediate authority.
- Confirmed the existing V5 configuration is readiness-only and the legacy executor refuses V5 execution.

Subsequent commands and deterministic gate results are appended during implementation.

## Implementation and focused validation

- Added the strict V5 types, scenario loader, executor, output/resume layer,
  deterministic fixture builder, authoritative CLI, config, frozen
  specification, notebooks, Make targets, and runbooks.
- `python -m compileall` on the V5 surface: pass.
- Ruff on the full CoreGraph surface: pass.
- mypy on 103 source files: pass with zero issues.
- Focused V5 pytest: 20 passed.
- Focused coverage: executor/label-firewall 95%; output/resume 100%.
- Notebook static validation: 12/12 syntax checked, pass; none executed.

## Canonical no-training validation

- Plan mode: exactly 6 archives, 180 base artifacts, 60 scenarios, 540
  bindings, and 240 coordinates; no fit and no target labels.
- Validate-only mode: 6 archive and 180 member hashes pass; no extraction, fit,
  target-label read, metric, or oracle.
- Representative real-cache assembly: Elliptic and DGraphFin seed-1 strict
  targets pass; source rows are bounded, target matrices are float32, and the
  target interface has no label field. No method was fit.

## Synthetic execution and recovery

- Complete deterministic campaign: 240/240 coordinates, zero failures.
- Chunk equivalence at three rows per chunk: pass for all four methods.
- Resume rehearsal: 240 reusable complete cells and unchanged method-tree hash
  `e78f8d9ea6faaba6c1a931a0c0cfda78bf1d703ad455c4ced4706511e899bf3c`.
- Timed synthetic campaign: 12 whole wall-clock seconds on the closure machine.
- Complete-run packaging and ZIP compressed-data test: pass. Synthetic gate
  outcome `NO_GO` is diagnostic-only and is not a real pilot result.

## Repository, paper, and frozen-boundary gates

- Full collection: 257 tests; full pytest and every declared coverage threshold
  pass.
- Theory numerical/status, synthetic mechanisms, and one-epoch CPU smoke: pass.
- ICLR claim audit and build: pass/results blocked; 14 main plus 5 supplement
  pages, zero undefined references/citations, overfull boxes, Type-3 fonts, or
  identity/private-path findings.
- Cross-paper overlap: pass.
- Frozen boundary: `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` and strict
  scientific delta gate pass.
- Real pilot execution: not invoked.
