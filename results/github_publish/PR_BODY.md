## Summary

This draft PR curates the completed FraudShiftBench research project into the
existing public repository while preserving its history and legacy Elliptic
reproduction surface. It adds the canonical benchmark and GraphSafe code,
active TKDE sources, final PDFs, frozen aggregate evidence, typed claim maps,
reader documentation, publication audits, and lightweight CI.

## Deliberately excluded

Raw datasets, raw prediction payloads, checkpoints, virtual environments,
caches, provider workspaces, local logs, backups, obsolete manuscript trees,
duplicate release ZIPs, private paths, credentials, and files above GitHub's
ordinary 100 MiB limit remain local.

## Validation

- Curated pytest: 27 passed.
- Corrected unittest discovery: 7 passed.
- Frozen support relation: 14 cases and 36 hash checks passed.
- Eight figures, 51 tables, and 50 verified bibliography entries regenerated.
- Main and supplement built locally as 14- and 30-page PDFs.
- Strict manuscript, visual-object, table-readability, and PDF-layout audits
  passed.
- Authoritative baseline ZIP hashes passed before exclusion; the public-package
  gate reports `ZERO_SCIENTIFIC_DELTAS`.
- Final staged safety audit: no secrets, identity-bearing contact data, private
  paths, raw data, raw predictions, archives, caches, or oversized objects.

## Known limitations

- No training, dataset download, or GPU experiment was run for this curation.
- Resource-blocked and unmeasured cells remain explicitly non-predictive.
- Full analysis reconstruction needs excluded provider data/imports.
- Hosted CI intentionally omits the full TeX build.
- Project-wide licensing remains unresolved; the branch adds
  `LICENSE_REVIEW_REQUIRED.md` and grants no new permission.

## Reviewer checklist

- [ ] Confirm the README reframing and legacy/canonical boundary.
- [ ] Review the complete file diff and removal of `.DS_Store`.
- [ ] Confirm final PDFs and active LaTeX sources are the intended release.
- [ ] Check evidence/claim and resource-boundary documentation.
- [ ] Confirm public-safety and large-file audit reports.
- [ ] Resolve licensing before granting reuse permission.
- [ ] Observe the new lightweight Actions workflow.
- [ ] Do not merge until the repository owner completes review.

