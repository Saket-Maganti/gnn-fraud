# V5 real-pilot abort and recovery

Stop the runner with the normal process interrupt. Do not kill storage, edit checkpoints, delete failures, forge COMPLETE markers, or copy synthetic files into the real root.

Abort and investigate if any of these occurs:

- Git, preregistration, effective-config, dependency-lock, schema, archive, member, scenario, or coordinate identity mismatch;
- target labels appear before a checksum-bound policy freeze;
- a non-finite score, regret below `-1e-12`, checksum error, partial file, duplicate coordinate, unresolved failure, or unexpected directory appears;
- the output filesystem approaches capacity or the output root is not the planned real root.

Recovery procedure:

1. Preserve the output tree read-only and record the failing coordinate, stage, exception record, traceback, local/remote SHA, manifest hashes, and available disk.
2. Verify the repository remains clean and exactly matches the SHA in `RUN_MANIFEST.json`.
3. Verify the V5.1 preregistration, base config, dependency lock, archive/member surface, and effective execution payload reproduce their recorded hashes.
4. Fix only a proven code or operational defect through normal review. A changed code SHA or effective setting creates a new identity; do not present old coordinates as reusable under it.
5. If identities are unchanged and the failure is transient, rerun the exact authorised command with `--resume`. The runner reuses only checksum-valid COMPLETE coordinates and reruns nonterminal coordinates.
6. If identities changed, choose a new empty real output root and repeat plan, validate-only, and explicit authorization. Preserve the old root as failed provenance.
7. Package only when the plan, manifest, directories, evaluations, checkpoints, and COMPLETE identities form the same exact 240-coordinate set and no failure remains.

Never recover by changing the frozen oracle, abstention cost, methods, seeds, gate thresholds, source sampling, or metrics after target outcomes are visible. Escalate persistent scientific or identity failures instead of weakening a gate.
