# Level-4 GPU wave plan

Status: `PLAN_ONLY_NO_GPU_JOB_LAUNCHED`.

1. T4x2 bootstrap and repository/dataset/hash validation.
2. Fraud saved-output pilot validation, then separately authorised full fraud training.
3. Strong official baseline parity and fraud waves only for licensed task-valid adapters.
4. Controlled synthetic full mechanisms.
5. Primary GOOD adapter, then fallback only if its licence/data gate passes.
6. Resource profiling with fixed warmup and batch manifests.
7. Encoder, diagnostic, objective, and routing ablations.
8. Prediction regeneration only for integrity-confirmed missing artifacts.
9. Output validation, checksum, one final ZIP, and completion report.

Each coordinate is resumable and failures are explicit. OOM is `RESOURCE_BLOCKED`; no cell is silently skipped.
