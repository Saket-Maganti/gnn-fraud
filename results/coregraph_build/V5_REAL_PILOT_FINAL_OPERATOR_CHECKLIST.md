# V5 real-pilot final operator checklist

Status: `READY_FOR_LATER_AUTHORISED_RUN`; the real pilot has not been run.

Before execution, the operator must check every item in order:

1. Confirm branch `codex/coregraph-iclr-buildout-2026` and record the exact clean SHA.
2. Require an empty `git status --short` and verify local SHA equals the remote branch tip.
3. Confirm PR #2 is open, draft, unmerged, and uses the frozen base and V5 head branches.
4. Re-run cache validation: 6 archives, 180 members, 180 artifacts, 60 scenarios, 540 bindings, and 240 coordinates; no fitting or labels.
5. Verify preregistration SHA-256 `931cd9f39cec9f0d28a68f6a8c13ad3628ccc155797e0c8276b9e3f75c63b487`.
6. Run plan mode on the exact clean SHA and record its `effective_execution_config_sha256`; confirm the same value is present in every plan row and `RUN_MANIFEST.json`.
7. Confirm the real output root is a new compatible directory, outside Git or ignored, contains no synthetic manifest, and has the runner-required free space.
8. Run the exact plan command and inspect the 240-row coordinate set.
9. Run the exact validate-only command; require 6/6 archive and 180/180 member hashes with zero training and zero target-label reads.
10. Optionally run the quarantined synthetic rehearsal; never copy its metrics into the real root or paper.
11. Obtain explicit authority, then run only the real command in `V5_REAL_PILOT_EXACT_COMMANDS.md`.
12. Monitor COMPLETE and failure records without editing the output tree.
13. On interruption or failure, follow `V5_REAL_PILOT_ABORT_AND_RECOVERY.md`; resume only with the identical effective identity.
14. Package only after exactly 240 coordinates are complete with no unresolved failure.
15. Require pre-ZIP validation, CRC validation, and independent post-extraction validation to pass.
16. Give the immutable package, coordinate manifest, checksums, gate, and run manifest to an independent auditor.
17. Populate paper results only after that audit and the frozen claim gates; keep PR #2 unmerged until separately authorised.

Abort immediately on any SHA, preregistration, effective-config, archive/member, schema, coordinate-set, checksum, label-firewall, or dirty-tree mismatch.
