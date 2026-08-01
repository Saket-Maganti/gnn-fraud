# Level-4 reproducibility plan

Each future coordinate produces a run manifest before execution, code SHA, canonical config JSON and hash, dataset/archive/member hashes, environment record, prediction hash, resource record, validation report, completeness status, exclusion reason, and claim linkage. Coordinates are idempotent; resume logic validates existing hashes before skipping work.

The source tree contains no provider data, prediction payload, checkpoint, credentials, private local path map, official baseline checkout, or unresolved-licence output. Evidence ZIPs live in an ignored external cache and are opened by checksum-verified streaming. Source snapshots exclude `.git`, environments, caches, data, predictions, checkpoints, build intermediates, private paths, and Finder metadata.

The clean-room gate extracts the source snapshot, creates or selects a fresh dependency environment, runs tiny compile/lint/type/test/theory/synthetic/notebook gates, builds main and supplement PDFs, scans text and PDFs for identity/private paths, validates checksums, and compares the frozen FraudShiftBench boundary. The gate does not download provider data or execute official baselines.

Determinism is required for registry generation, scenario bindings, mechanism fixtures, run matrices, claim gates, paper placeholders, and release manifests. Real GPU kernels may require tolerance-based reproducibility; the tolerance, deterministic flags, hardware, and excluded nondeterministic operations must be recorded before those runs.

Release publication remains blocked for data or outputs with unresolved licence terms. Anonymous and full provenance releases are separate products. Shared FraudShiftBench inputs are disclosed, while manuscript text/figure overlap is audited conservatively.
