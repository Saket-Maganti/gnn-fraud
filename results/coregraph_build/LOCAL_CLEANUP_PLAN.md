# Local cleanup plan

The active repository contained reproducible ignored caches and build scratch.
After the final local code, coverage, paper, and visual gates, the explicit
entries marked `REMOVED_TO_TRASH_AFTER_VALIDATION` in
`LOCAL_CLEANUP_INVENTORY.csv` were moved to Trash. Generated reports, final
PDFs, source snapshots, evidence, and committed artifacts were retained. The
validated `.venv` remains available through exact-SHA CI and is still excluded
from every snapshot and Git surface.

The parent project worktree contains unrelated staged historical content and is outside this build's mutation scope. The sibling curated checkout and `gnn-fraud-old` are read-only. Source evidence storage is read-only and was not mounted.

Anonymous/release trees are generated snapshots, not hand-edited authorities. Source-only ZIP snapshots exclude Git metadata, environments, caches, datasets, predictions, checkpoints, private paths, logs, and build intermediates.
