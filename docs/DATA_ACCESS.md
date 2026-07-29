# Data access

No raw dataset is distributed in this repository.

Set a local data root, or use the repository-relative default:

```bash
export FRAUDSHIFTBENCH_DATA_ROOT="$PWD/data/raw"
```

Expected provider-acquired files include:

- Elliptic transaction feature, class, and edge CSVs;
- DGraphFin in its provider-supplied NPZ form;
- T-Finance only for code paths that explicitly use the unmeasured extension;
- IBM AML synthetic transaction files for the declared variant/scale studies.

The loaders fail loudly when required files are missing. There is no silent
synthetic substitute outside the dedicated synthetic smoke/theory tests.

Provider terms, access procedures, and any Kaggle credentials remain external.
Never commit `kaggle.json`, `.env`, raw CSV/NPZ files, generated predictions, or
download/upload bundles.

