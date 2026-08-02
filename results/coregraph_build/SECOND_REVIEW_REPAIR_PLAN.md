# CoReGraph second-review pilot-semantics repair plan

Governing specification:
`COREGRAPH_SECOND_REVIEW_PILOT_SEMANTICS_CLOSURE.md`.

Starting state:

- branch `codex/coregraph-iclr-buildout-2026`;
- local and remote tip `712732e4fc7a852135a9288eb90138976a98e0d7`;
- draft PR `#2`, open and draft;
- clean worktree;
- frozen boundary `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.

## Stop boundary

This pass changes deterministic pilot-semantics code, regression tests,
schemas, specifications, documentation, paper placeholders, CI gates and
history-free release artifacts. It will not install official baselines,
download data, create or connect real manifests, execute the pilot, launch
Kaggle, run multi-seed experiments, merge PR #2, or modify any frozen
FraudShiftBench/TKDE asset. It will not claim pilot execution readiness.

## Regression-first repair waves

1. Preserve frozen abstention decisions; define zero-coverage behavior;
   separate contract and instance oracle semantics.
2. Apply review budgets and abstention capacities within each source contract;
   exclude unavailable predictions from ranking; distinguish fallback from
   abstention.
3. Rename the partial GraphSafe component; source-train the Mowst-inspired
   threshold; separate expert and router seeds.
4. Enforce exact two-dataset, ten-seed, target-contract, method, ablation and
   metric coverage; compute meaningful corrected ablation effects.
5. Make corrected inference and effect thresholds decisive; preserve
   matched-contract worst-case and within-seed regret aggregation.
6. Align method, pilot, statistical, theory, claim and paper documentation;
   freeze the V3 gate specification and rebuild the anonymous release.
7. Run every deterministic gate, verify the public tree and frozen boundary,
   create focused commits, fast-forward push, update the still-draft PR body,
   and wait for clean CI.

Each ledger item advances from `TEST_PENDING` to `REPAIRED_GATE_PASS` only
after its regression and all mandated deterministic gates pass.
