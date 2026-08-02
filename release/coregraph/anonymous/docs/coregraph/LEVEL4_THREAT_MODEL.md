# Level-4 leakage and validity threat model

| Threat | Attack path | Prevention | Required audit |
|---|---|---|---|
| Target-label leakage | labels enter gate, diagnostics, calibration, threshold, early stopping | typed access regime and source-only fit APIs | call/path audit plus scenario validation |
| Row-scope leakage | target test rows appear in source train/validation | scenario-local permitted splits and unit-ID disjointness | byte-backed row audit |
| Chronology leakage | future nodes/edges/features enter past graph view | cutoff-aware graph construction | timestamp and edge endpoint audit |
| Role leakage | one base artifact is source and target in one scenario | immutable role-neutral base plus atomic bindings | 540-binding structural audit |
| Cross-dataset seed pooling | seed 1 from two datasets treated as paired | compound experimental unit | statistical block-key audit |
| Score/label misalignment | predictions and labels use different row order | deterministic unit IDs, schema, and coordinate checks | streaming member validation |
| Unknown-label scoring | provider unknowns treated as negative | `label_known` and label mapping | member row audit |
| Resource leakage | unavailable expert selected then masked after routing | constraint composition before softmax | exact zero-mass tests |
| Budget leakage | target labels choose threshold or `K` | operational budget known; threshold source-frozen | provenance audit |
| Oracle leakage | target oracle reported as baseline | diagnostic-only registry status | claim-language audit |
| Private path/identity leak | absolute paths or authors enter release/PDF | environment path layer and public-tree scan | text, PDF, archive audit |
| Baseline misrepresentation | internal approximation uses official method name | exact registry status and parity requirement | baseline registry audit |
| Cost substitution | training runtime used as inference latency | resource schema requires inference measurement | profiler record audit |
| Archive substitution | same filename or coordinate accepted without checksum | canonical archive and member hashes | fail-closed cache validation |

The exact archives are now present locally. Structural, member-hash, row-order, chronology, label-known, cross-expert alignment, and cross-protocol scope audits pass. Target metrics, target oracles, resource measurements, and performance claims remain blocked until their separately authorised workflows run.
