# CoReGraph Level-4 master build report

Verdict: `COREGRAPH_LEVEL4_BUILD_COMPLETE_READY_FOR_SAVED_OUTPUT_PILOT`

## Authority and integrity

The active CoReGraph authority is the independent Git checkout on
`codex/coregraph-iclr-buildout-2026`; the exact completed SHA is the normally
pushed branch tip recorded in the final Git/PR handoff. The curated
FraudShiftBench authority remains frozen at
`2dec25eac1d7a8951f9d4639f49e889c4c9ca486`, with
`ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`. The historical folder remains
read-only pending the documented backup/uniqueness decision.

The six canonical RB09v3 archives occupy 1,004,185,299
compressed bytes in `${COREGRAPH_EVIDENCE_CACHE}/archives`; all are read-only,
ZIP-CRC valid, whole-file SHA-256 valid, and represented by 180 unique streamed
member hashes:

- `dgraphfin_10seed_inductive_isolated.zip` — `6ce0d2e37893a7a162d6d575347f6606eb63f90b7940cc3a259dc309cf88b8c8`
- `dgraphfin_10seed_strict_inductive.zip` — `e0055d3482107d16c7d52574b0a32adc1e7ae9236b67dbdf12f57327c0e6bce5`
- `dgraphfin_10seed_transductive.zip` — `6d0167aae53b681bb7ffc037b84c723869ba718f30feb955c333890bfe8783d5`
- `elliptic_10seed_inductive_isolated.zip` — `20f25a1f93604ea5eb8537c8808f9b69dae2fc82eccbccfd36c50c443aee94e8`
- `elliptic_10seed_strict_inductive.zip` — `24752f5ffdc082dc79ca5084701fccd04d2ac9588b4b15712598bdfe8daa1e4a`
- `elliptic_10seed_transductive.zip` — `99d2f7ad1ad95fd7c30c193da9003c091b0c3fdce028dccd2bd0019f35869c08`

No archive or prediction payload is in Git. No SSD source was used after the
verified local cache became sufficient, and no prediction member was
permanently extracted.

## Scientific system

- V5: 180/180 role-neutral base artifacts,
  60/60 scenarios, and 540/540
  bindings materialise; byte, schema, coordinate, chronology, known-label,
  ordering, 60-group expert alignment, and 20-group cross-protocol row-scope
  audits pass.
- Method: factorised/interaction/attention/uncertainty/latent/hybrid contract
  encoders; expert diagnostics; contract, instance, and hierarchical routing;
  resource masks; robust regret/CVaR/budget/stability/abstention objectives;
  selective, counterfactual, and resource evaluation.
- Theory: fixed-mixture impossibility, regret decomposition, compositional
  bound, and selective-risk transfer are internally proved pending external
  review; resource-mask validity is proved and reviewed. Executable finite
  checks pass, including declared failure cases.
- Benchmarks: six fraud contracts, 15 deterministic synthetic mechanisms, GOOD
  primary adapter plan, and OGB molecular fallback plan. No official benchmark
  download or training occurred.
- Baselines: 20 registered; status counts are
  `{"IMPLEMENTED_INTERNAL": 15, "OFFICIAL_AVAILABLE_NOT_INSTALLED": 3, "UNAVAILABLE_LICENSE": 2}`. Internal methods are implemented;
  official repositories remain uninstalled or explicitly licence/dependency
  blocked.
- Statistics: preregistration SHA-256 `1536ba2a645baf965b09ca428caa689b3436ee25432872ce4b5cabf4c684dd3f`; all empirical claims
  remain blocked until validated results pass the frozen gates.

## Validation and paper

- Tests: 237 passed; critical-module coverage minimum
  85% and every declared group meets the
  85% gate.
- Deterministic checks: compile, Ruff, mypy, theory, synthetic fixtures,
  one-epoch CPU smoke, notebook syntax/packaging fixture, paper claims,
  cross-paper overlap, archive offline smoke, release checksums, clean room,
  and frozen boundary pass. Final clean-room status: `PASS`.
- Runbooks: 9 Kaggle T4x2 plus
  3 local notebooks;
  none executed.
- Paper: 12 main sections, 7 supplement sections, 13 main pages, 5 supplement
  pages, 8 non-empirical figures, 7 empty result templates, and 11 tables.
  All 18 pages passed visual QA. Four empirical claim families remain blocked;
  no numerical result was invented.
- Overlap firewall: `PASS` with zero common eight-grams, zero
  exact long sentences, and zero byte-identical visual assets.
- Cleanup: 39,171,198 reproducible workspace bytes removed; no evidence,
  report, environment, historical folder, or user-owned file was deleted.

## Explicit execution boundary

No full real-data training, target metric, target oracle, official-baseline
installation, Kaggle job, empirical paper population, force-push, or PR merge
occurred. The only training-like operation was the documented one-epoch,
24-node synthetic CPU smoke (`PASS`), which used no provider data.

The next authorised action is a separately invoked saved-output pilot using
`LEVEL4_NEXT_EXECUTION_PROMPTS/01_saved_output_pilot_execution.md`. This build
is not labelled submission-ready.
