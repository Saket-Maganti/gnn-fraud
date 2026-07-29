# Canonical source map

The selected public surface follows the final TKDE command logs, frozen hashes,
active imports, release manifests, and clean-room build. It does not treat the
newest filename as canonical merely because it is newer, and it does not merge
incompatible historical implementations.

The machine-readable map is
`results/github_publish/CANONICAL_SOURCE_MAP.csv`. Its central decisions are:

- `paper_tkde/` is the authoritative manuscript source. `gnnpaper/`,
  `runs_paper/`, `fraudshiftbench_paper/`, and the Markdown drafts under
  `paper/` are historical or intermediate and are not synced as canonical
  manuscript sources.
- `scripts/tkde_rebuild/` and `scripts/tkde_visual_rebuild/` are the final
  evidence, claim, table, figure, audit, and clean-room implementation.
- `results/tkde_rebuild/` and the small lock files named in the frozen hash
  ledger are the canonical machine-readable evidence layer. Raw predictions and
  nested imported output trees are excluded.
- `fraudshiftbench/` is the reusable executable benchmark-contract surface.
  `models/registry.py` remains the model factory; legacy model and runner files
  are retained only where they reproduce the public repository's earlier work.
- The final PDFs are copied from `output/pdf/` into the GitHub-facing
  `paper/pdf/` directory. Duplicate compiled PDFs and LaTeX auxiliaries are not
  synced.
- The old public-repository figures and legacy reproduction code remain in the
  cloned history unless directly replaced by a canonical current file. They are
  documented as preserved legacy surfaces, not silently deleted or relabeled as
  final TKDE evidence.
- The repository has no resolved project-wide licence. Publication therefore
  uses `LICENSE_REVIEW_REQUIRED.md`; no permissive licence is invented.

Scientific status is unchanged: `ZERO_SCIENTIFIC_DELTAS`. This map changes only
the public repository surface and portability/documentation around it.
