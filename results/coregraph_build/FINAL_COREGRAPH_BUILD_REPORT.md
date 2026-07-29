# Final CoReGraph / ContractShift pre-run build report

Verdict:
`COREGRAPH_BUILD_COMPLETE_BASELINES_PENDING_EXTERNAL_INTEGRATION`

The implementation, pre-run artifacts, deterministic validation, theory
package, anonymous release, and execution handoff are complete. The stronger
`COREGRAPH_ICLR_BUILD_COMPLETE_EXECUTION_READY` verdict is not issued because
headline official-baseline parity, two reuse licences, provider manifests, and
saved-output pilot evidence remain external prerequisites.

## Completed

- Isolated branch/worktree based exactly on curated commit
  `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`; the original dirty checkout was
  untouched.
- Pre-repair audit, 25-item risk register, legacy compatibility map, and frozen
  boundary manifest.
- Immutable six-axis `DeploymentContract`, canonical serialization/schema,
  compatibility and one-way legacy adapter.
- `EvidenceUnitV2`, `TypedClaim`, conservative `SupportEngine`, 22-claim
  mutation fixture, failure localization, and predictive-ordering gates.
- Node, edge, and transaction tasks; corrected Elliptic, DGraphFin,
  T-Finance, Elliptic++, IBM AML, and GOOD V2 boundaries.
- Separate causal train/validation/target graph views, central leakage audit,
  identifier guards, target-access records, and 15-threat leakage matrix.
- Feature/tree experts, deterministic sampled GCN/SAGE execution, graph memory
  guards, event batching, official subprocess/parity adapters, and robust
  source objectives.
- CoReGraph/CoReRouter factorised encoding, unknown axes, interactions,
  label-free diagnostics, resource masks, fallback, abstention, explanations,
  compute, and stability.
- Classification/ranking/budget/regret/CVaR/compute/calibration/composite
  objectives and correct evaluation/statistical implementation.
- Eight-regime deterministic synthetic generator and held-out contract split
  families.
- Multi-contract, source-only saved-output router pilot and preregistered gate.
- Frozen future matrices: 1,400 five-seed screening rows, 1,050 ten-seed final
  rows, 105 ablations, 160 theory/synthetic rows, 75 GOOD rows, 1,520 fraud
  rows, and 135 resource rows.
- Nine Kaggle T4x2 scheduler notebooks, three local notebooks, exact runbooks,
  result import/statistics/paper/claim tooling, and CI with no heavy execution.
- Three proved, scoped theory results with numerical and wording gates.
- Anonymous seven-page ICLR skeleton with three non-empirical figures,
  generated result placeholders, claim ledger, bibliography, appendix,
  successful LaTeX build, and seven-page visual QA.
- Identity-scanned, history-free anonymous release package.

## Validation

- 94 root tests pass; frozen and new suites run together.
- 77% statement coverage over `coregraph/`.
- Ruff and typed-core mypy gates pass; compileall passes.
- One-epoch CPU feature/router/sampled-GCN smoke passes.
- 67 tests pass from inside the anonymous package.
- All 12 notebooks pass static syntax/metadata validation.
- Three theory checks and theorem-status gate pass.
- Claim and paper skeleton audits pass without empirical evidence.
- Anonymous audit reports no identity/path/checksum failure.
- `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)`.

## Honest blockers

1. No provider manifests or files were staged, as required by the session
   constraint. Real-data scheduling remains blocked.
2. No saved prediction manifests were supplied; the pilot correctly reports
   `BLOCKED_INCOMPLETE_METHOD_ROWS`.
3. Mowst, CIGA, official TGN, and GOOD are pinned but need isolated checkout,
   task bridge, smoke, and parity output.
4. GraphMETRO and EERM have no verified reusable licence at the pinned
   revisions. There is no legitimate shell command that resolves a missing
   licence; written permission or a licence-bearing upstream revision is
   required.
5. CIGA is graph-classification-only at the pinned integration boundary and
   cannot support fraud node/event ordering without a task-valid bridge.
6. IBM AML Large remains resource-blocked; T-Finance temporal claims remain
   blocked without genuine timestamps.
7. The official target-year ICLR template/policy is not yet frozen and must be
   rechecked.

## Exact next commands

Stage and validate provider manifests after manual lawful acquisition:

```bash
mkdir -p data/manifests/coregraph
cp configs/coregraph/datasets/MANIFEST_TEMPLATE.yaml data/manifests/coregraph/elliptic.yaml
# Fill provider, licence, temporal semantics, exact paths, bytes and SHA-256.
.venv/bin/python scripts/coregraph/validate_dataset_manifests.py --all
```

Review pinned baseline acquisition without executing it:

```bash
bash scripts/coregraph/install_official_baselines.sh --dry-run
```

The first licensed upstream integration command is:

```bash
git clone --filter=blob:none https://github.com/facebookresearch/mowst-gnn external/mowst-gnn
git -C external/mowst-gnn checkout 2e3569962d2388bfda4535cdd1fc0b6eaec88a28
```

After its task bridge and parity fixture are implemented, rerun:

```bash
.venv/bin/python scripts/coregraph/validate_baseline_registry.py
```

Run the saved-output pilot only after typed manifests are available:

```bash
export COREGRAPH_SAVED_PREDICTIONS_ROOT=/absolute/path/to/manifests
.venv/bin/python scripts/coregraph/run_saved_output_pilot.py --execute
.venv/bin/python scripts/coregraph/evaluate_pilot_gate.py
```

Then rerun all local gates:

```bash
make coregraph-local-gates PY=.venv/bin/python
```

The branch may be pushed only after data, pilot, baseline parity, and licence
gates applicable to the intended headline matrix pass:

```bash
git push -u origin codex/coregraph-iclr-buildout-2026
```
