# Anonymous CoReGraph release

Build the anonymous package with:

```bash
python scripts/coregraph/build_anonymous_release.py
python scripts/coregraph/audit_anonymous_release.py
```

The generated `release/coregraph/anonymous/` directory contains source,
configuration, tests, paper source, notebooks, and reproducibility metadata.
It excludes git history, frozen manuscript assets, raw data, model weights,
predictions, results, caches, local environments, personal paths, and author
identity. The audit fails on forbidden paths, oversized files, symlinks,
identity markers, absolute user paths, and checksum drift.

This pre-run package intentionally contains no empirical result. Provider
datasets and official baseline repositories must be obtained under their own
licences and verified against the included registry.
