# Saved-output pilot execution plan

The pilot reads only aligned prediction manifests. It does not train provider
models or download data.

Required surfaces are Elliptic and DGraphFin, experts MLP/GCN/GraphSAGE,
strict/isolated/transductive contracts, and seeds 1–10. IBM Small/Medium is
included only when common transaction IDs and matching prediction manifests
exist.

The harness validates task unit, dataset, contract, IDs, labels, split, config
hash, and checksum before alignment. Source validation fits the router,
Mowst-inspired confidence router, and all thresholds using per-contract
budgets and capacities. Target capacity may constrain only the frozen
label-free target decision. Held-out target labels are read only for final
offline scoring. Blocked cells remain unordered. Expert-prediction seed is
distinct from the deterministic router-training seed.

Dry discovery:

```bash
COREGRAPH_SAVED_PREDICTIONS_ROOT=/absolute/prediction/root \
  .venv/bin/python scripts/coregraph/run_saved_output_pilot.py \
  --config configs/coregraph/pilot/saved_output.yaml
```

Measured saved-output execution:

**Blocked in this pass.** Do not use the following command until manifest
conversion, dry-run completeness validation, and third independent review
explicitly authorize it.

```bash
COREGRAPH_SAVED_PREDICTIONS_ROOT=/absolute/prediction/root \
  .venv/bin/python scripts/coregraph/run_saved_output_pilot.py \
  --config configs/coregraph/pilot/saved_output.yaml --execute
```

Then:

```bash
.venv/bin/python scripts/coregraph/evaluate_pilot_gate.py \
  --pilot-result results/coregraph_pilot/saved_output_pilot.json
```

No result is currently claimed. The go/no-go criteria are frozen in
`PILOT_GATE_FROZEN_SPEC.json`.
