# Saved-output pilot execution plan

The pilot reads only aligned prediction manifests. It does not train provider
models or download data.

Required surfaces are Elliptic and DGraphFin, experts MLP/GCN/GraphSAGE,
strict/isolated/transductive contracts, and seeds 1–10. IBM Small/Medium is
included only when common transaction IDs and matching prediction manifests
exist.

The harness validates task unit, dataset, contract, IDs, labels, split, config
hash, and checksum before alignment. Source validation fits the router and all
thresholds. Held-out target labels are read only for final offline scoring.
Blocked cells remain unordered.

Dry discovery:

```bash
COREGRAPH_SAVED_PREDICTIONS_ROOT=/absolute/prediction/root \
  .venv/bin/python scripts/coregraph/run_saved_output_pilot.py \
  --config configs/coregraph/pilot/saved_output.yaml
```

Measured saved-output execution:

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
`PILOT_GO_NO_GO_SCHEMA.json`.
