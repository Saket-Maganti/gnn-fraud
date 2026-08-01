# Local cleanup report

Status: `PASS_CONSERVATIVE_RECOVERABLE_CLEANUP_COMPLETE`.

Workspace bytes removed: **37558378** across **382** reproducible files.
Targets were moved to macOS Trash and remain recoverable. The count is the
exact sum of logical file sizes immediately before cleanup and excludes the
earlier temporary independent-clone workspace, which was also moved to Trash.

Removed targets were limited to `.pytest_cache`, `.mypy_cache`, `.ruff_cache`,
`.coverage`, Python bytecode/`__pycache__`, Finder metadata, `tmp`,
`paper_iclr/build`, and root paper LaTeX intermediates. Post-cleanup discovery
found none of those targets outside the deliberately retained `.venv`.

The validated virtual environment was retained through exact-SHA CI; it is
reproducible, ignored, and excluded from both source snapshots. No dataset,
prediction, checkpoint, canonical archive, member index, report, paper PDF,
source file, user file, or historical folder was removed. The former
linked-worktree pointer remains backed up outside Git. `gnn-fraud-old` remains
read-only pending the documented user archival decision.
