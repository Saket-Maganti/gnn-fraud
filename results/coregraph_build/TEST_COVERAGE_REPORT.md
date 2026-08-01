# Test and coverage report

Status: `PASS_LOCAL_DETERMINISTIC_GATES`

- Full repository suite: 237 tests passed after standalone theory entry-point,
  final-handoff generation, and source-snapshot public-audit regressions were
  added.
- All `coregraph/` statement coverage: 87%.
- Required Level-4 critical coverage:
  - contracts: 95%;
  - routing: 93%;
  - objectives: 91%;
  - evidence: 88%;
  - diagnostics: 90%;
  - benchmarks: 99%;
  - baselines: 93%;
  - resources: 100%;
  - theory: 94%;
  - selective/resource/counterfactual evaluation: 93%;
  - saved-output pilot core: 87%;
  - statistical analysis: 93%;
  - pilot gate evaluator: 85%.
- Expanded mypy gate: 97 source files, zero issues.
- Ruff and compileall: pass.

The legacy scenario-manifest module remains at 81% under its pre-existing 80%
gate; missed paths are defensive legacy-conversion/error branches. The
separately prohibited empirical branch in `run_saved_output_pilot.py` remains
at 58% under its documented 50% plan/validate-only gate. Neither exception is
used to lower the 85% gate for the new Level-4 critical packages listed above.

High-risk regression coverage includes path portability, archive/member
integrity, streaming without extraction, scenario-local role semantics,
row-scope/label/chronology leakage, unseen/noisy/missing contract axes,
diagnostics, three routing levels, all resource-mask branches, robust
objectives, abstention/selective risk, counterfactuals, all 15 synthetic
mechanisms, baseline truthfulness, statistical pairing, theorem
counterexamples, notebook syntax, paper claim blocking, overlap, release
hygiene, and the frozen inherited boundary.
