# Kaggle GPU runs (no GitHub)

Start here in every fresh Kaggle attempt:

1. Rebuild the code bundle locally:

```bash
python kaggle/generate_notebooks.py
bash scripts/bundle_kaggle_code.sh
```

2. Upload `kaggle/datasets/gnn-fraud-runs-code.zip` as a **new Kaggle dataset version**. The dataset title can be anything.
3. Start a fresh Kaggle notebook/session and attach the code bundle plus the Elliptic dataset.
4. Run `00_kaggle_input_debug.ipynb`.
5. If that passes, run `00_setup_smoke_test.ipynb`.
6. Then run `00_01_02_max_results.ipynb`.
7. Do not patch old notebooks manually unless you are debugging a stale upload.

The first cell of every generated notebook must print:

- `IF THIS CELL DOES NOT PRINT NAME-AGNOSTIC RECURSIVE DISCOVERY, YOU ARE USING AN OLD NOTEBOOK.`
- notebook name and generated timestamp
- recursive `/kaggle/input` listing
- candidate zips and extracted roots
- selected code source
- bootstrap path and `INSTALL_DEPS = False`

If those lines are missing, the notebook is stale. Re-upload the regenerated notebook or start from a fresh one.

## Quick Path

| Session | Notebook | Time | Unlocks |
| ------- | -------- | ---- | ------- |
| debug | `00_kaggle_input_debug.ipynb` | minutes | input discovery, staging, file-path bootstrap import |
| smoke | `00_setup_smoke_test.ipynb` | 5-15 min | import check + dry-run only |
| 1 | `00_01_02_max_results.ipynb` | ~3-5 hr | P03 all-model + P04 matched GNN 10-seed + prediction CSVs |
| 2a | `03_p06_elliptic_protocol_gap.ipynb` | ~2-3 hr | Elliptic protocol gap |
| 2b-pilot | `03_p06_npz_pilot.ipynb` | ~30-90 min | DGraphFin/T-Finance load/export sanity |
| 2b split/max | `03_p06_dgraphfin_gnn.ipynb`, `03_p06_tfinance_gnn.ipynb`, or `03_p06_multidataset_max.ipynb` | ~8-14 hr total | 3-dataset GNN, if NPZ inputs are ready |

## Inputs

Attach to every notebook:

- Code bundle zip from `kaggle/datasets/gnn-fraud-runs-code.zip`
- Elliptic dataset containing the three `elliptic_txs_*.csv` files

Optional:

- CPU results containing `validation_clean/runs.csv`
- `dgraphfin.npz`
- `tfinance.npz`

Dataset names are irrelevant. The bootstrap recursively discovers inputs by markers:

- code: `kaggle/kaggle_bootstrap.py`, `scripts/runs_harness_common.py`
- full bundle: also `scripts/run_validation_clean_gnn.py`, `runs_expansion/README.md`
- Elliptic: CSV filenames
- CPU results: `validation_clean/runs.csv`
- optional NPZ: `dgraphfin.npz`, `tfinance.npz`

## Dependency Mode

Generated notebooks call `bootstrap(install_deps=False)` through `INSTALL_DEPS = False`.
That is the safe default for Kaggle because the live kernel may already contain
compiled packages built against its current NumPy and Torch stack.

Only opt into dependency installation in a fresh throwaway kernel after reading
the printed Python, NumPy, torch, CUDA, and PyG versions. Do not downgrade NumPy
in a live Kaggle kernel.

If NumPy is broken, repair it manually and restart the kernel:

```bash
pip install --force-reinstall --no-cache-dir \
  "numpy>=2.0,<2.4"
```

## Troubleshooting

**Dataset mounted as `/kaggle/input/datasets`:** this is supported. The debug notebook recursively searches below `/kaggle/input`.

**Zip extracted by Kaggle:** this is supported. The bootstrap finds nested marker files and copies the inferred repo root.

**`ModuleNotFoundError: No module named kaggle.kaggle_bootstrap`:** you are using an old notebook. Fresh notebooks import the staged file with `importlib.util.spec_from_file_location`.

**`ValueError: numpy.dtype size changed`:** do not downgrade NumPy. Use the repair command above, restart the kernel, then rerun the debug notebook.

**Old notebook still has old bootstrap cell:** regenerate notebooks locally, upload the fresh notebook, and verify the first cell prints the old-notebook warning plus recursive discovery.

**When to use dependency installs:** leave `INSTALL_DEPS = False` by default. Only switch the boolean in a new kernel if PyG is missing and you accept a conservative `--no-deps` wheel install.

Regenerate notebooks with `python kaggle/generate_notebooks.py`.

## Download and import `runs_outputs`

Generated notebooks now write a consistent output package:

```text
/kaggle/working/runs_outputs/
  validation_clean/
  matched_gnn_protocol/
  multi_dataset_protocol/
  mitigation/
  predictions/
  logs/
  manifests/
  validation/
  tables/
  provenance/
  README.md
  KAGGLE_OUTPUT_MANIFEST.json
```

After a notebook finishes, download the whole `runs_outputs` folder and import it
locally:

```bash
python scripts/import_kaggle_outputs.py \
  --input "$KAGGLE_OUTPUT_DIR/runs_outputs" \
  --dest results/runs \
  --dedupe \
  --validate \
  --execute
```

Then run the local refresh:

```bash
bash scripts/post_kaggle_refresh.sh
```

Build reviewer/reproducibility outputs and score readiness:

```bash
python scripts/build_runs_reviewer_package.py --anonymize-check-only
python scripts/build_reproducibility_capsule.py
python scripts/score_runs_readiness.py
```

Conflict reports live under `results/runs/import_reports/`. Prediction CSV
validation lives under `results/runs/prediction_validation/`. Keep unsupported
claims Pending / Not claimed when datasets, predictions, matched seeds, or
multi-model rankings are missing.
