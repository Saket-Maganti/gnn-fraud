# Kaggle T4 x2 runbook

Attach the repository snapshot and provider data as private Kaggle datasets.
Keep internet disabled. Select GPU T4 x2. Notebook 01 validates hashes and
dependencies; notebooks 02–08 each consume a frozen CSV; notebook 09 packages
manifests, predictions, telemetry, logs, and checksums.

Rows are assigned deterministically by `rows[0::2]` and `rows[1::2]`. Each lane
uses one device unless an adapter explicitly advertises distributed support.
Never average partially complete seed sets. Download the package only after
notebook 09 verifies all completed manifests and identifies failed/resource
blocked rows.

Resume by rerunning the same notebook against the same output dataset. The
runner skips only a manifest whose config, code, dataset, schema, result, and
prediction checksums all match.
