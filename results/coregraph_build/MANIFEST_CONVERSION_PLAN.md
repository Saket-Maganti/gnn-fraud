# V4 historical manifest conversion plan

Status: `EXECUTED_NO_TRAINING`

1. Search the two authorized roots read-only for exact
   `{elliptic,dgraphfin}__{protocol}__{mlp,gcn,sage}__seedN.csv` prediction
   names.
2. Trace candidates to historical prediction-validation reports and recompute
   SHA-256 checksums.
3. Stream rows to audit required columns, declared dataset/protocol/model/seed,
   split tokens, `label_known`, provider labels, score domain, duplicate IDs,
   and timestamps. Do not compute metrics.
4. Normalize only evidenced aliases: historical `inductive_isolated`,
   `transductive`, `mlp`, `sage`, and split token `val` to their frozen V4
   names.
5. Require an explicit evidence-map entry for role, full deployment contract,
   original checksum, code/config hashes, compute cost, and cost provenance.
   Never infer missing scientific metadata.
6. Write a V4 manifest only when every field and audit passes. Otherwise emit
   `BLOCKED_METADATA_UNRESOLVED` or the exact audit blocker.
7. Build the complete 360-cell matrix (two datasets, three protocols, ten
   expert seeds, three experts, two roles).
8. Run registry binding, typed cross-role leakage, no-training runner, and
   no-training gate-completeness checks only when loadable manifest coverage
   permits them.

Historical files are never overwritten. Converter outputs are isolated under
`results/coregraph_manifest_conversion_v4/`; review summaries are written
under `results/coregraph_build/`.
