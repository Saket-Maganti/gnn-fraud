# FraudShiftBench

FraudShiftBench is a research benchmark and evidence framework for evaluating
graph-based fraud models under changing deployment contracts. Its central
question is not simply which architecture wins, but whether the answer changes
when graph visibility, temporal ordering, feature access, model selection,
review budget, and resource limits match a different deployment setting.

The completed TKDE manuscript reports that evaluation contracts can change
model rankings and practical decisions across the evaluated evidence. It also
uses GraphSafe-TTA as a bounded saved-score case study. GraphSafe is not
presented as a universal improvement: Elliptic favors simple averaging in the
reported comparison, and no GraphSafe contrast survives correction across the
declared 48-test family.

Current paper status:
`TKDE_VISUAL_REBUILD_COMPLETE_PROFESSOR_REVIEW_READY`. This is not a
submission-ready or externally validated artifact claim. The scientific curation
gate is `ZERO_SCIENTIFIC_DELTAS`.

## What is in this repository

- executable deployment-contract, evidence-unit, metric, and claim-gate code in
  `fraudshiftbench/`;
- legacy and current dataset, model, runner, and evaluation code under `data/`,
  `models/`, `experiments/`, and `utils/`;
- the authoritative TKDE LaTeX sources under `paper_tkde/`;
- final main and supplement PDFs under `paper/pdf/`;
- frozen aggregate evidence, 22 typed claims, the 247-record evidence
  inventory, scalar provenance, resource boundaries, and visual/readiness
  reports under `results/`;
- deterministic figure, table, bibliography, support-validation, and
  publication-audit scripts under `scripts/`;
- a lightweight public test suite that requires neither datasets nor a GPU.

The final empirical surface covers Elliptic, DGraphFin, and IBM AML synthetic
regimes under explicitly recorded contracts. A T-Finance loader and other
extension scaffolding exist, but the completed manuscript does not promote them
to measured evidence.

## What is deliberately absent

Raw datasets, raw prediction exports, checkpoints, Kaggle downloads and upload
bundles, local environments, caches, backups, page renders, historical release
ZIPs, and machine-specific paths are excluded. Evidence locks and aggregate
source tables record provenance without republishing those payloads.

Data must be obtained from its original provider. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## Start here

1. [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
2. [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md)
3. [`docs/GITHUB_ANALYSIS_ENTRYPOINT.md`](docs/GITHUB_ANALYSIS_ENTRYPOINT.md)
4. [`docs/EVIDENCE_AND_CLAIM_MAP.md`](docs/EVIDENCE_AND_CLAIM_MAP.md)
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
6. [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)

## Installation

For inspection, publication regeneration, and the curated tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
```

For model training and the original PyTorch Geometric runners, use the pinned
research environment:

```bash
python -m pip install -r requirements.txt
```

PyTorch/PyG compatibility is intentionally pinned. GPU experiments also require
provider data and may need substantially more than 16 GB of accelerator memory.

## Quick verification

These commands do not train models or download data:

```bash
make compile
make test
make unittest
make support
make claims
```

The same lightweight gates run in GitHub Actions. High-cost tests are skipped
because the corresponding raw data and prediction payloads are intentionally
not distributed.

## Paper and asset regeneration

Regenerate reviewer-facing figures and tables from the included frozen
aggregates:

```bash
make figures
make tables
```

Build the main paper and supplement with an installed TeX distribution:

```bash
make paper
```

The build runs strict BibTeX cycles for both documents. Full commands and input
boundaries are in [`README_BUILD.md`](README_BUILD.md) and
[`paper_tkde/README_BUILD.md`](paper_tkde/README_BUILD.md).

## Reproduction tiers

- Tier 0: inspect final PDFs, claims, and source tables.
- Tier 1: run the curated tests and regenerate paper assets.
- Tier 2: run deterministic CPU analysis from included aggregate evidence.
- Tier 3: obtain datasets and execute declared GPU experiments.
- Tier 4: attempt cells recorded as resource-blocked under a larger,
  predeclared resource envelope.

Details are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository history

This branch is curated from the existing public `Saket-Maganti/gnn-fraud`
history. Earlier Elliptic figures and reproduction scripts from `origin/main`
are preserved as legacy surfaces. They are not the authoritative TKDE
manuscript or evidence inventory; the repository map labels the distinction.

## Citation and licence

Review-anonymous citation metadata is provided in
[`CITATION.cff`](CITATION.cff). Project-wide licensing and third-party
redistribution terms are not yet resolved. Until that review is completed,
[`LICENSE_REVIEW_REQUIRED.md`](LICENSE_REVIEW_REQUIRED.md) grants no permission
to copy, modify, or redistribute the repository.

