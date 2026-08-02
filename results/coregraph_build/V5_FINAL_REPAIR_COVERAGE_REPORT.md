# V5 final-repair coverage report

Status: `PASS`; no gate was weakened.

Coverage was measured with the complete 328-test repository suite using the pinned `.venv` runtime and Python tracing.

| Required module | Required | Observed |
|---|---:|---:|
| `coregraph/experiments/v5_pilot_executor.py` | 95% | 95% |
| `coregraph/experiments/v5_pilot_outputs.py` | 95% | 97% |
| `coregraph/experiments/v5_scenario_loader.py` | 90% | 90% |
| `coregraph/experiments/v5_package_validator.py` | 95% | 100% |
| `coregraph/evaluation/regret.py` | 95% | 100% |

All pre-existing declared CoreGraph coverage gates also passed: contracts 95%, routing 94%, objectives 91%, legacy pilot 87%, evidence 88%, theory 94%, and the remaining Makefile-scoped modules at or above their frozen thresholds.
