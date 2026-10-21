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
9. **Pilot semantics:** V5.1 requires `coregraph_v5_metric_schema_v2`. Primary
   regret is measured against a row-wise feasible hindsight oracle that includes
   abstention; the best fixed non-abstaining expert is diagnostic only. Global
   target AUPRC, selective risk, coverage, and the exact 1% review fraction are
   distinct fields. Old or mixed metric schemas force `INCONCLUSIVE`.
10. **Pilot authorization:** require 6/6 archives, 180 role-neutral artifacts,
    60 scenarios, 540 bindings, 240 exact coordinates, a clean tree, the V5.1
    preregistration hash, one shared effective-execution hash, exact package-set
    validation before and after ZIP extraction, and the explicit later token.
    Readiness does not authorize `--execute`; the real pilot remains unexecuted.

Failure of gates 1–2 stops local commit. Failure of gates 3–8 blocks heavy
execution, headline claims, release, or push as applicable; blockers must be
reported with the exact prerequisite or command.
