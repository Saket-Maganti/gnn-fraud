# CPU pilot runbook

```bash
.venv/bin/python scripts/coregraph/check_theory_numerically.py
.venv/bin/python scripts/coregraph/run_coregraph_smoke.py
.venv/bin/python scripts/coregraph/run_saved_output_pilot.py
```

The smoke run is synthetic and one epoch. It validates gradients, masks,
fallback, abstention, prediction export, telemetry, and resume; it is not
scientific evidence.
