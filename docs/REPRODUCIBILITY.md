# Reproducibility

## Tier 0 — inspect

Read the final PDFs under `paper/pdf/`, the claim ledger, evidence inventory,
resource boundaries, scalar provenance, and final readiness reports.

## Tier 1 — lightweight public verification

Install `requirements-publication.txt`, then run `make compile`, `make test`,
`make unittest`, `make claims`, `make figures`, `make tables`, and `make paper`
when TeX is available. These commands use frozen aggregates and do not train.

## Tier 2 — deterministic CPU analysis

Run `scripts/tkde_rebuild/validate_support_relation.py --frozen-only`. The
validator rechecks 14 support mutations and verifies the saved deterministic
regeneration audit against the included canonical inputs and generated
aggregates.

## Tier 3 — dataset/GPU experiments

Acquire datasets from their providers, install `requirements.txt`, declare the
resource envelope, and use the documented runners. Full
`scripts/tkde_rebuild/compute_analysis.py` regeneration also belongs here
because it reconstructs dataset statistics and seed rows from excluded raw and
imported files. These operations are not part of hosted CI.

## Tier 4 — resource-blocked cells

IBM Large, Medium GINE, the fixed DGraphFin GAT cell, and the DGraphFin
GraphSAGE max-pool rerun require a separately declared resource plan. A
successful run would be new evidence and must not silently alter the frozen
manuscript.

Exact commands and outcomes for the curated checkout are recorded in
`docs/CLEAN_REPOSITORY_VALIDATION.md`.
