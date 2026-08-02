# V5 final-repair implementation report

Status: `IMPLEMENTED_AND_LOCALLY_VALIDATED_REAL_PILOT_UNEXECUTED`.

V5.1 closes all seven preflight findings:

1. Primary regret now uses matched row-wise feasible expert-or-abstain actions, enforces `-1e-12`, and reports best-fixed quantities separately.
2. Preregistration v5.0 remains preserved; v5.1 is frozen at SHA-256 `931cd9f39cec9f0d28a68f6a8c13ad3628ccc155797e0c8276b9e3f75c63b487`.
3. `coregraph_v5_metric_schema_v2` makes global AUPRC, frozen-fraction recall, selective risk/coverage, regret, diagnostics, and boolean feasibility explicit.
4. A canonical effective execution hash binds code, dependency, base config, preregistration, chunking, worker, mode, dtype, determinism, schemas, streaming, sampling, and inference policy across every resume/output/package layer.
5. Packaging validates the exact 240-coordinate set, full cross-run identity, per-coordinate files and checksums, archive/member sets, terminal state, and absence of failures/partials before and after ZIP extraction.
6. The canonical runner has no dirty-tree, method-subset, scenario-subset, or redundant dry-run escape hatch; real execution unconditionally requires the authorization token and a clean compatible output root.
7. Config, frozen gate, documentation, manuscript, tests, release, and operator handoff are aligned.

Validation evidence:

- canonical cache-only: 6/6 archives, 180/180 members, 180 artifacts, 60 scenarios, 540 bindings, 240 coordinates, zero training, zero target-label reads, zero extraction;
- representative no-fit assembly: Elliptic and DGraphFin sources/targets assembled, float32 target scores, target bundles without labels, vaults unopened, no fit or metric;
- fresh synthetic: 240/240 across four methods, zero failures, corrected regret minimum `0.07603882589690639`, exact packaging and post-extraction validation PASS;
- unchanged resume coordinate-tree SHA-256 `d4145245d3415a1409e8f1324899af55602a50bdcea9cb04d5225d8e251ce6af`; effective-config mutation rejection PASS;
- tests, coverage, paper, anonymous release, deterministic snapshots, checksums, clean room, overlap, notebooks, and frozen boundary PASS.

Synthetic results are non-scientific and quarantined outside Git. No real target outcome was observed.
