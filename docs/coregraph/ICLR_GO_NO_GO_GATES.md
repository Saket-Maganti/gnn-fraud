# ICLR go/no-go gates

1. **Frozen boundary:** verify `FROZEN_TKDE_INPUT_HASHES.csv` and zero tracked
   TKDE scientific deltas.
2. **Local correctness:** compile, lint, type-check the typed core, run all unit
   and mutation tests, theory numerical checks, notebook validation, anonymous
   release audit, and one-epoch CPU smoke.
3. **Data:** provider manifests/checksums exist; temporal and task semantics
   pass leakage fixtures; no synthetic fallback.
4. **Baselines/licences:** every headline baseline is official or a validated
   reimplementation with parity evidence and reusable licence.
5. **Analysis freeze:** matrix and analysis-family hashes are recorded before
   final runs.
6. **Hardware:** memory/latency feasibility is profiled for every resource
   class.
7. **Evidence:** predictions, complete seeds, integrity, pairing, and claim
   scopes pass SupportEngine.
8. **Paper:** no result placeholder is converted into a factual claim without
   an eligible evidence unit; target-year policy/template is confirmed.

Failure of gates 1–2 stops local commit. Failure of gates 3–8 blocks heavy
execution, headline claims, release, or push as applicable; blockers must be
reported with the exact prerequisite or command.
