# CoReGraph independent-audit repair plan

Governing specification:
`COREGRAPH_INDEPENDENT_AUDIT_METHOD_AND_PILOT_REPAIR.md`.

Starting state:

- branch `codex/coregraph-iclr-buildout-2026`;
- local and remote tip `67a3a3e13ad0fd0221de7ef19a1aa58b1106dd6f`;
- draft PR `#2`;
- clean worktree;
- frozen boundary `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.

## Stop boundary

This pass changes deterministic scientific-core code, tests, schemas,
documentation, and generated gate/audit artifacts only. It will not install
official baselines, acquire datasets, attach real saved predictions, execute
the saved-output pilot, launch Kaggle, or run any multi-seed experiment. It
will not claim execution readiness.

## Test-first repair waves

1. Contract schema, migration, hashes, access-consistent splits, construction
   composition, and DGraphFin unobserved-node semantics.
2. Structured expert availability, expert-aware routing tokens, score types,
   safe all-unavailable rows, and functional abstention.
3. Differentiable group objective, seed-bound prediction manifests, honest
   baseline adapters, source-only pilot training, and seed-block statistics.
4. Theory/code identity, adversarial numerical cases, SupportEngine proof and
   scope semantics, and deterministic qualitative synthetic scenarios.
5. Compile, Ruff, mypy, full tests, anonymous-package tests, critical-module
   coverage, theory/synthetic/smoke/notebook/paper/public/anonymous/frozen
   gates. Repair failures without weakening gates.

Regression tests are written before each corresponding implementation repair
where practical. The findings ledger is updated from `TEST_PENDING` through
`REPAIRED_GATE_PASS` only after the relevant deterministic gate passes.

## Commit and review strategy

Use focused normal commits on the existing branch, inspect each staged diff,
push only fast-forwards, retain PR `#2` as draft, and finish with a
second-review handoff. The preferred final verdict is
`COREGRAPH_INDEPENDENT_AUDIT_REPAIRS_COMPLETE_READY_FOR_SECOND_REVIEW`, subject
to honest downgrade if any mandated deterministic gate remains blocked.
