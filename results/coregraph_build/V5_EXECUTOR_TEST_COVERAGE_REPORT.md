# V5 executor test and coverage report

Status: `PASS`

- Repository collection: 257 tests.
- Full pytest/coverage run: pass.
- V5 executor and target-label firewall module: 95% line coverage.
- V5 atomic output/resume module: 100% line coverage.
- Existing critical groups: contracts 95%, routing 94%, objectives 91%, pilot
  core 87%, manifest conversion 86%, scenario manifests 81% against its 80%
  declared exception, canonical recovery 88%, protocol registry 86%, leakage
  86%, statistics 93%, evidence 88%, diagnostics 90%, benchmarks 99%, baselines
  93%, resources 100%, theory 94%, and selective/resource/counterfactual 93%.

The V5 suite covers exact surface cardinalities and identities, role reuse,
archive/member drift, deterministic source assembly, absent target labels,
nonserialisable/single-use evaluation vault behavior, freeze/hash/row-alignment
failures, all four methods, chunk equivalence, three gate outcomes, checkpoint
corruption, every resume identity check, stale reuse, explicit failure records,
and complete output hashing.

Compileall, Ruff, and mypy pass over the expanded executor and CLI surface.
