# CoReGraph Level-4 pre-run release

Status: `VALIDATED_BUILD_ARTIFACT_RESULTS_BLOCKED`.

This release contains compact metadata and deterministic source snapshots only. It contains no provider archive, prediction payload, checkpoint, credential, empirical Level-4 metric, target oracle, or official-baseline output. The external canonical cache is referenced by checksum but never redistributed.

Run `make coregraph-cleanroom` to validate extraction, a fresh isolated environment, tiny tests, both anonymous PDFs, checksum closure, and identity/path hygiene.
