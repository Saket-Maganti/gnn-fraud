# V5 final-repair test report

Status: `PASS`; 328 repository tests passed.

The full `.venv` pytest collection passed after the V5.1 implementation. Focused repair coverage includes:

- matched-action oracle selection, all-unavailable abstention, separate best-fixed diagnostics, numeric tolerance, and malformed regret inputs;
- old/mixed metric schemas and corrected gate-field enforcement;
- effective hash changes for chunk rows, execution mode, workers, output schema, and metric schema;
- resume rejection on changed effective identity;
- exact planned/observed coordinate equality and 36 adversarial package/identity/checksum mutations;
- pre-ZIP, CRC, and post-extraction package validation;
- all four methods, target-label firewall, policy freeze, deterministic chunking, failure records, COMPLETE reuse, and corruption handling.

Quality gates passed:

| Gate | Result |
|---|---|
| `python -m compileall` | PASS |
| Ruff | PASS |
| mypy | PASS, 104 source files |
| Full pytest | PASS, 328 tests |
| Theory numerical/status | PASS |
| Deterministic synthetic mechanisms | PASS |
| One-epoch CPU smoke | PASS; no provider data |
| Notebook validation | PASS, 12 static and 0 executed |
| Anonymous release tests/audit | PASS |
| Level-4 checksum and clean-room validation | PASS |
| Frozen TKDE checker | `ZERO_TKDE_SCIENTIFIC_DELTAS (249 files)` |

The deterministic V5 synthetic campaign completed 240/240 coordinates with zero failures. Packaging, extraction validation, and unchanged-identity resume passed. No real pilot path was exercised.
