# Third-Review Command Log

All commands are run from `$COREGRAPH_REPO` unless noted. Historical paths
under `$HISTORICAL_GNN_FRAUD_REPO` are read-only. The aliases denote the two
roots supplied in the governing specification.

| Stage | Command or action | Purpose | Status |
|---|---|---|---|
| Specification | `wc -l` and complete paged `sed` read of the attached 457-line governing document | Read the governing specification in full before editing | PASS |
| Baseline | `git status --short --branch`; `git rev-parse HEAD`; `git remote -v`; repository file/instruction discovery | Confirm clean existing branch at required tip and locate local instructions/surfaces | PASS |
| Discovery | Targeted `rg` over runner, manifest, gate, statistics, risk, comparator, tests, and build artifacts | Reproduce and map TR-01 through TR-08 before implementation | PASS |
| Regressions | Targeted `pytest` over `test_third_review_v4_integration.py` and affected prior-review suites | Exercise hashed contract IDs, aliases, row scope, unknown labels, leakage, stratification, converter evidence, runner CLI and gate | PASS |
| Historical audit | `convert_prediction_manifests_v4.py --root $COREGRAPH_REPO --root $HISTORICAL_GNN_FRAUD_REPO ...` | Read-only discovery, checksum, row/split/label-known, registry, completeness and leakage-readiness audit | PASS_BLOCKED_METADATA |
| Compile/lint/types | `make coregraph-compile coregraph-lint coregraph-typecheck` | Deterministic static validation | PASS |
| Tests | `python -m pytest -q` | Full repository suite | PASS_195 |
| Coverage | `make coregraph-coverage` | Explicit gates for manifest conversion, registry, scope/labels, leakage, runner/gate and stratified statistics | PASS |
| Theory/synthetic/smoke | Numerical theory gate; theorem-status gate; synthetic method checks; one-epoch CPU smoke | Dataset-free scientific and execution checks | PASS |
| Notebook/paper | Notebook validation; ICLR placeholder PDF build; paper skeleton and claim audit | Keep empirical placeholders blocked and ensure renderability | PASS_8_PAGES |
| Anonymous package | Anonymous release build and audit with package tests | Validate identity/path hygiene and package execution | PASS_250_FILES |
| Public tree | Export committed `HEAD` with `git archive`; run `validate_public_tree.py` | Audit the exact committed public tree without local environments or caches | PASS_ZERO_FINDINGS |
| Frozen boundary | `hash_frozen_assets.py --verify` | Verify the 249 frozen FraudShiftBench/TKDE files byte-for-byte | ZERO_TKDE_SCIENTIFIC_DELTAS |

This log is append-only for the duration of the third-review pass.
