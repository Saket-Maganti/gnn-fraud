# CoReGraph pre-build repository audit

Date: 2026-07-29
Audited base: `2dec25eac1d7a8951f9d4639f49e889c4c9ca486`
Build branch: `codex/coregraph-iclr-buildout-2026`
Authoritative frozen branch: `origin/codex/curated-fraudshiftbench-2026`

## Repository and isolation

The authoritative checkout was on an unborn, heavily modified branch. It is
not safe for new development and was not changed. This audit runs in the
requested dedicated sibling worktree, created directly from the curated commit.
Raw data is absent from the curated tree and will remain external.

## Frozen scientific boundary

The following are immutable for this build:

- `results/tkde_rebuild/`
- `results/tkde_visual_rebuild/`
- `results/v24_imported/`
- `results/v26_imported/`
- `results/v27_imported/`
- `results/v28_imported/`
- `results/runs_rb09v3/`
- `results/runs_rb17_review_budget_worst_block/`
- `kaggle_workspace/manifests/V22_FINAL_GPU_EVIDENCE_LOCK.json`
- `manuscript_assets/`
- `paper_tkde/`
- `paper/pdf/FraudShiftBench_TKDE_main.pdf`
- `paper/pdf/FraudShiftBench_TKDE_supplement.pdf`

The build will record hashes for the complete boundary, not only files that
happen to be referenced by one table. Historical loaders and utilities remain
available for reproduction. Corrected behaviour is implemented under
`coregraph/` or behind an explicit compatibility adapter.

## Existing protocol and evidence system

`fraudshiftbench.protocols.ProtocolContract` is a frozen, string-valued record
with four defaults. It is useful as historical metadata but does not validate
axis values, compose contracts, represent budgets/resources, express unknown
coordinates, or create held-out compositional splits.

`fraudshiftbench.evidence.EvidenceUnit` and
`fraudshiftbench.claims.ClaimGate` provide a bounded legacy gate. They do not
encode task/prediction units, model contracts, statistical blocks, integrity or
construct state, resource semantics, contradiction predicates, pairing
diagnostics, or the full support-status lattice required by CoReGraph. The
current 22 claim-ledger rows are suitable compatibility fixtures, not a general
truth engine.

## Data and task audit

The unified `FraudDataset` is node-centric. Unknown labels use the repository
convention `0`, fraud is `1`, and normal is `2`; masks are disjoint. It cannot
cleanly model an edge or transaction as the prediction unit.

The legacy Elliptic adapter preserves official time steps and train-only
scaling. New evidence still needs explicit train, validation, and target graph
views for each visibility contract.

The DGraphFin loader derives a node time from the median of all incident event
timestamps, including future lifecycle events, using a Python list for every
node. It then converts the graph to undirected and drops edge types and
timestamps from the returned object. This is not admissible for new temporal
claims and is not scalable enough for the intended dataset.

The Elliptic++ loader has a useful correlated-domain warning and fail-loud data
policy. It requires stricter schema/join validation, chunk-aware loading, and a
small fixture.

The T-Finance loader silently substitutes edge order for missing timestamps in
both NPZ and DGL paths, then uses the DGraphFin lifecycle median. A temporal
claim cannot use that fallback. Source/licence and timestamp quality require
explicit audit fields.

There is no IBM AML transaction-edge adapter or GOOD adapter in the curated
tree.

## Graph visibility and leakage audit

`experiments._multi_harness.build_inductive_view` builds only a training
subgraph. `_train` evaluates checkpoints on `eval_data`; for inductive runs this
is the full graph, so validation logits can aggregate test-period structure.
Labels stay masked, but structure leakage changes the learned checkpoint. The
same harness evaluates the target on a full graph regardless of a typed
visibility contract.

Training and scaling masks are generally explicit, but the reusable system has
no central sentinels for future edges, target nodes in validation, target-label
priors, scaler provenance, threshold provenance, router target leakage, ID
memorisation, or duplicate contract-fold examples.

## Models and experts

The legacy registry is convenient but its status metadata is insufficient.
`GraphTransformer`, hand-rolled `GPS`, heuristic `PCGNN`, `SnapshotTGN`, and
light variants are useful diagnostics; they are not verified official
implementations. The alias `tgn` maps to a discrete snapshot approximation,
which can be mistaken for faithful streaming TGN. Full-graph GPS attention is
quadratic and infeasible even on Elliptic, while DGraphFin requires sampling.

GraphSafe-V2 is a saved-prediction selective wrapper. The graph-feature gate is
a validation-fit two-expert gate. The graph-harm detector mixes label-free and
label-dependent diagnostics. Rolling validation constructs ordered time
windows but does not construct fold-specific graph views. These are bounded
legacy case studies and do not supply CoReGraph novelty.

There is no task-general expert API, availability/resource mask, official
baseline registry, pinned integration status, edge mini-batching interface, or
memory guard.

## Metrics, statistics, and selection

The known critical defects are confirmed:

- `models.protocol_mitigation.select_threshold_on_validation` computes F1 for
  the `auprc` branch.
- `fraudshiftbench.result_analysis.calibration_slope_intercept` performs
  ordinary least squares of binary labels on logits, not logistic calibration.
- `fraudshiftbench.result_analysis.rank_average_scores` assigns distinct ranks
  to tied scores.
- `fraudshiftbench.result_analysis.correction_rows` compares each Holm
  threshold independently and does not enforce step-down stopping.
- `utils.metrics.holm_bonferroni` tolerates missing values, uses strict `<` for
  rejection, and is not the strict new API required by this build.
- Small-sample inference helpers elsewhere can fall back to normal
  approximations; the new default must be exact or resampling-based.

## Training, resume, and configuration

`experiments._multi_harness.run_grid` skips any existing result filename without
validating its configuration, code, data manifest, output schema, completion
state, or prediction checksum. Early stopping counts evaluation checks, while
callers express patience as if it were epochs; the default therefore permits up
to 400 epochs without stopping. The fallback train-loss comparison is also
encoded indirectly as a negated “best validation” score.

The repository has no complete validated run schema, canonical run hash, atomic
status transition model, stale-output classification, crash-recovery record, or
prediction manifest for the new project.

## Paper, release, CI, and licence

The TKDE manuscript and PDFs are complete frozen assets. CoReGraph needs a
distinct paper surface; no result numbers can be transferred or invented.

The current GitHub repository is named and therefore unsuitable as the final
anonymous submission package. Existing anonymisation utilities are tied to the
legacy paper. The root `LICENSE_REVIEW_REQUIRED.md` confirms that a permissive
project-wide licence has not been resolved.

The curated CI validates the legacy public tree without datasets or GPUs. It
does not run CoReGraph type checks, leakage tests, notebook validation, theory
checks, anonymous-release smoke tests, or frozen-boundary verification against
the new hash manifest.

## Audit conclusion

The curated tree is a sound frozen scientific input but not an execution-ready
CoReGraph system. New work must remain versioned and isolated. The detailed
issue register is in `PREBUILD_CODE_RISK_REGISTER.csv`; legacy compatibility
decisions are in `LEGACY_COMPATIBILITY_MAP.csv`.
