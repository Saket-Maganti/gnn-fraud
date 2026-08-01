# V5 executor independent dry-run report

Status: `PASS_NO_REAL_TRAINING`

Canonical validate-only mode passed:

- 6 archive SHA-256 and ZIP surfaces;
- 180 member SHA-256 identities;
- 180 role-neutral artifacts;
- 60 materialised scenarios;
- 540 scenario-local bindings;
- 240 unique primary coordinates;
- 0 permanent extractions;
- 0 fits and 0 target-label reads.

A representative no-training assembly used provider seed 1 with strict
inductive target protocol on both datasets. Elliptic produced two source
environments of 7,085 bounded rows each and 16,670 target-test rows. DGraphFin
produced two source environments of 8,192 bounded rows each and 170,207
target-test rows. Both target score matrices were float32 and neither target
bundle exposed a label field.

The deterministic tiny fixture exercised the same 180/60/540/240 schemas and
completed all 240 method coordinates with zero failures. Re-running with
`--resume` preserved the method-tree SHA-256
`e78f8d9ea6faaba6c1a931a0c0cfda78bf1d703ad455c4ced4706511e899bf3c`.
A separately timed synthetic campaign completed in 12 whole wall-clock seconds
on the closure machine and packaged into a ZIP whose compressed-data test
passed. This timing and its `NO_GO` gate outcome are synthetic diagnostics only;
they are not a projection or scientific result for the real campaign.

The dry-run estimate records 50,000 target rows per inference chunk, 2,550,000
bytes of numeric chunk working set excluding Python/model overhead, 610,639,200
raw target-score bytes, and 1,831,917,600 raw route-weight bytes across the
planned campaign. Real fit time remains unknown until an authorised run.
