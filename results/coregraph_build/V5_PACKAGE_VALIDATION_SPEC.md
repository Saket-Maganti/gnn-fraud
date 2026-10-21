# V5 exact package-validation specification

Packaging reads `PILOT_PLAN.csv` and `RUN_MANIFEST.json` and requires exact equality among planned keys, manifest keys, method directories, checkpoints, evaluations, and COMPLETE identities. Missing, extra, duplicated, stale, mixed-mode, mixed-schema, failed, or partial coordinates fail.

Each coordinate validates scientific fields; repository, dependency, base/effective-config, preregistration, output, metric, and method-registry identities; nine archive-member bindings; policy-freeze and target-score identities; terminal checkpoint; route, evaluation, result, and COMPLETE checksums; and absence of unresolved failure/temporary files.

Before ZIP creation the runner writes `PACKAGE_VALIDATION_REPORT.json`, `PACKAGE_COORDINATE_MANIFEST.csv`, and `OUTPUT_CHECKSUMS.sha256`. It validates the ZIP CRC, extracts into a temporary directory, revalidates exact sets and checksums, deletes the extraction automatically, seals the report, rebuilds the ZIP, and validates the final extracted package again.
