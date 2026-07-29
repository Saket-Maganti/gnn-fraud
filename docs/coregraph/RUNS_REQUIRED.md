# Runs required

The machine-readable source of truth is `configs/coregraph/run_matrices/`.
`SCREENING_5SEED_GRID.csv` covers method and baseline screening.
`FINAL_10SEED_GRID.csv` is confirmatory and must not be reduced after looking
at outcomes. `ABLATION_GRID.csv`, `THEORY_SYNTHETIC_GRID.csv`, `GOOD_GRID.csv`,
`FRAUD_GRID.csv`, and `RESOURCE_GRID.csv` retain their own analysis families.

Each row contains a stable run key, dataset/task, source and target contracts,
method, objective, seed, access regime, hardware class, estimated resource
class, prerequisite, and status. `TBD_PROFILE` is used instead of invented
runtime. Rows marked `BLOCKED_LICENSE`, `BLOCKED_DATA`, or
`PENDING_INTEGRATION` must not be scheduled.
