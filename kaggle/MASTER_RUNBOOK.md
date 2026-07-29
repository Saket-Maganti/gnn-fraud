# Master runbook — maximum efficiency + maximum results

Single plan across **local CPU** (done) and **Kaggle GPU** (P03/P04/P06).
No GitHub on Kaggle — code via uploaded zip dataset.

---

## Fresh Kaggle procedure

Use this sequence whenever the code bundle or notebooks change:

1. Rebuild locally:

```bash
python kaggle/generate_notebooks.py
bash scripts/bundle_kaggle_code.sh
```

2. Upload `kaggle/datasets/gnn-fraud-runs-code.zip` as a **new Kaggle dataset version**. The dataset name can be anything.
3. Start a fresh Kaggle notebook/session.
4. Run `00_kaggle_input_debug.ipynb`.
5. If debug passes, run `00_setup_smoke_test.ipynb`.
6. Then run `00_01_02_max_results.ipynb`.
7. Do not patch old notebooks manually unless you are debugging stale state.

The first cell must print the old-notebook warning plus name-agnostic recursive
discovery, selected code source, bootstrap path, and `INSTALL_DEPS = False`.
If it does not, stop and upload the regenerated notebook.

---

## Already verified (do not re-run)

| Prompt | Artifact | Time spent |
| ------ | -------- | ---------- |
| P01 | Foundation freeze | ~15 min |
| P02 | GraphSAGE 10-seed validation-clean (CPU) | ~90 min |
| P05 | Elliptic feature-only multi-protocol (30 rows) | ~5 min |
| CPU prediction validation | 30 feature-only CSVs, 1,396,920 rows | ~15 sec |
| CPU feature-only mitigation sweep | 240 policy/budget rows | ~15 sec |
| P09 | Go/no-go audit | ~45 min |

**Never re-train GraphSAGE 10-seed on GPU** — attach `gnn-fraud-cpu-results`.

---

## Design principles

1. **No duplicate training** — strict_inductive lives in `validation_clean/` (P03); P06 uses only `chronological,transductive` for extra protocol coverage.
2. **Fewest Kaggle sessions** — blend smoke + P03 + P04; separate session only for P06 (long).
3. **Checkpoint after each phase** — partial downloads if session dies.
4. **Max seeds where claims need it** — 10 seeds validation-clean; 10 seeds matched GNN (not 5).
5. **Export predictions during GPU runs** — enables mitigation without re-training.
6. **Three-dataset last** — stage NPZ once, pilot first, then split or max P06.

---

## Kaggle dataset attachments

| Dataset | Sessions | Required? |
| ------- | -------- | ----------- |
| Code bundle zip | all | **Yes** |
| Elliptic CSV dataset | all | **Yes** |
| CPU results with `validation_clean/runs.csv` | Session 1 | **Strongly yes** (~40 min saved) |
| Dataset containing `dgraphfin.npz` | Session 2b | For 3-dataset claims |
| Dataset containing `tfinance.npz` | Session 2b | For 3-dataset claims |

Dataset names do not matter. The bootstrap locates every input by contents:
code markers (`kaggle/kaggle_bootstrap.py`, `scripts/runs_harness_common.py`),
Elliptic CSV filenames, `validation_clean/runs.csv`, and optional NPZ names.
It searches recursively below `/kaggle/input`, including `/kaggle/input/datasets`
and Kaggle-extracted nested zips.

Settings for training sessions: **GPU on**. Internet is only needed if you
intentionally opt into dependency installation in a fresh kernel.

Kaggle planning constraints: budget around a ~30 hr weekly GPU quota and a
12 hr CPU/GPU notebook execution cap. Split anything estimated above 8-10 hr.

---

## Session 1 — Core Elliptic GNN evidence (GPU)

**Notebook:** `notebooks/00_01_02_max_results.ipynb`

| Step | Prompt | Runs | Est. time |
| ---- | ------ | ---- | --------- |
| Bootstrap + smoke | 00 | 0 | ~8–20 min first session |
| All-model validation-clean | P03 | 30 new (skip 10 GraphSAGE) | ~1.5–2 hr |
| Matched GNN 10×2 | P04 | 20 | ~1–1.5 hr |
| Metric + prediction validation + checkpoint | — | — | ~10–20 min |
| **Session total** | | | **~3–5 hr** |

**Unlocks:**
- Validation-clean all-model benchmark (**Verified**)
- Matched GNN GraphSAGE 10-seed Elliptic (**Verified**)
- GNN prediction CSVs for mitigation/gap-decomposition follow-up
- RUNS workshop-tier Elliptic evidence

**After download:** merge `runs_outputs/` → local `results/runs/`, then:
```bash
python scripts/build_runs_results_provenance.py
python scripts/export_runs_paper_tables.py --output-dir runs_paper/tables
```

---

## Session 2a — Elliptic protocol gap (GPU, no NPZ needed)

**Notebook:** `notebooks/03_p06_elliptic_protocol_gap.ipynb`

Runs GNNs on **chronological + transductive only** (skips strict_inductive — already in P03).

| Models | Protocols | Seeds | Runs | Est. time |
| ------ | --------- | ----- | ---- | --------- |
| MLP, GCN, SAGE, GAT | chronological, transductive | 1–5 | 40 | ~2–3 hr |

**Unlocks:** rank-reversal / protocol-gap analysis on Elliptic without redundant training.

---

## Session 2b — Three-dataset GNN (GPU, requires NPZ)

**Preferred notebooks:** pilot first, then split or max.

| Notebook | Use |
| -------- | --- |
| `notebooks/03_p06_npz_pilot.ipynb` | seed-1 DGraphFin/T-Finance load/train/export check |
| `notebooks/03_p06_dgraphfin_gnn.ipynb` | DGraphFin-only split sweep |
| `notebooks/03_p06_tfinance_gnn.ipynb` | T-Finance-only split sweep |
| `notebooks/03_p06_multidataset_max.ipynb` | all-in run only with enough quota/headroom |

Same as 2a but adds DGraphFin + T-Finance when NPZ datasets attached.

| Scope | Runs | Est. time |
| ----- | ---- | --------- |
| Elliptic only (2a) | 40 | ~2–3 hr |
| NPZ pilot | up to 16 | ~30–90 min |
| DGraphFin split | 40 | ~3–7 hr, OOM risk |
| T-Finance split | 40 | ~2–5 hr |
| + DGraphFin + T-Finance max | 120 | **~8–14 hr**; split if near cap |

Run **2a first** while staging NPZ; run **2b** when files ready. Do not wait for NPZ to finish Elliptic protocol-gap work.

**Unlocks:** multi-dataset leaderboard, cross-dataset rank-reversal (RUNS main borderline).

---

## Local CPU — after GPU downloads

| Step | Prompt | Est. time | Depends on |
| ---- | ------ | --------- | ---------- |
| Merge + validate all GPU artifacts | — | ~15 min | Session 1–2 |
| Merge + validate prediction CSVs | — | ~5–15 min | Session 1–2 |
| Mitigation / gap decomposition | P07 | ~5–30 min CPU | prediction CSVs |
| Paper rewrite | P08 | ~2–4 hr | P03+P04 min; richer after P06 |
| Final go/no-go re-audit | P09 | ~30 min | all above |

P08 can **start after Session 1** using Verified Elliptic claims; update again after Session 2b.

---

## Total wall-clock budget

| Track | Sessions | Est. total |
| ----- | -------- | ---------- |
| **Minimum viable (workshop)** | Session 1 only | **~3–5 hr GPU** + ~3 hr local P08 |
| **Strong Elliptic** | Session 1 + 2a | **~5–8 hr GPU** |
| **RUNS main attempt** | Session 1 + 2b | **~12–18 hr GPU** + local |

---

## Notebook map (use these, ignore others unless debugging)

| Priority | Notebook | Use when |
| -------- | -------- | -------- |
| **A** | `00_01_02_max_results.ipynb` | Default — Session 1 |
| debug | `00_kaggle_input_debug.ipynb` | First notebook in every fresh Kaggle session |
| B | `03_p06_elliptic_protocol_gap.ipynb` | Session 2a |
| B0 | `03_p06_npz_pilot.ipynb` | NPZ staging sanity before heavy non-Elliptic |
| B1 | `03_p06_dgraphfin_gnn.ipynb` | DGraphFin split |
| B2 | `03_p06_tfinance_gnn.ipynb` | T-Finance split |
| C | `03_p06_multidataset_max.ipynb` | Session 2b (NPZ ready) |
| smoke | `00_setup_smoke_test.ipynb` | Dry-run after input debug |

Legacy split notebooks (`01_`, `02_`) kept for retries if one phase fails.

---

## Claim coverage matrix

| Claim | Session |
| ----- | ------- |
| Validation-clean all-model 10-seed | 1 |
| Matched GNN random vs chronological | 1 |
| Feature-only multi-protocol | done (P05) |
| Elliptic protocol gap (chrono/trans) | 2a |
| Three-dataset GNN benchmark | 2b |
| Mitigation / gap decomposition | local P07 |
| RUNS main generality | 2b + P07 |

---

## One-page checklist

```
[ ] python kaggle/generate_notebooks.py
[ ] bash scripts/bundle_kaggle_code.sh
[ ] Upload kaggle/datasets/gnn-fraud-runs-code.zip as a new Kaggle dataset version
[ ] zip results/runs/validation_clean → upload gnn-fraud-cpu-results
[ ] Fresh Kaggle session: run 00_kaggle_input_debug
[ ] If debug passes: run 00_setup_smoke_test
[ ] Kaggle Session 1: 00_01_02_max_results (~3-5 hr)
[ ] Download runs_outputs → merge locally → provenance + tables
[ ] Kaggle Session 2a: 03_p06_elliptic_protocol_gap (~2-3 hr)
[ ] Stage dgraphfin.npz + tfinance.npz as Kaggle datasets
[ ] Kaggle NPZ pilot: 03_p06_npz_pilot (~30-90 min)
[ ] Kaggle Session 2b: split DGraphFin/T-Finance or max notebook (~8-14 hr total)
[ ] Local: validate prediction CSVs → P07 → P08 → P09
```

## Troubleshooting

| Symptom | Action |
| ------- | ------ |
| Dataset appears only as `/kaggle/input/datasets` | Supported; run the debug notebook and check recursive marker hits. |
| Kaggle extracted the uploaded zip | Supported; the bootstrap infers repo root from marker paths. |
| `ModuleNotFoundError: No module named kaggle.kaggle_bootstrap` | Old notebook. Use regenerated notebooks with file-path import. |
| `ValueError: numpy.dtype size changed` | Do not downgrade NumPy. Repair manually, restart the kernel, and rerun debug. |
| First cell lacks the old-notebook warning | Stale notebook. Regenerate locally and upload a fresh notebook. |
| Unsure about dependency installs | Keep `INSTALL_DEPS = False`. Only switch the boolean in a fresh kernel if PyG is missing. |

NumPy repair command, followed by kernel restart:

```bash
pip install --force-reinstall --no-cache-dir \
  "numpy>=2.0,<2.4"
```

---

## Post-Kaggle local refresh

At the end of each generated notebook, outputs are packaged under:

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

Download the entire `runs_outputs` folder. Import it locally with:

```bash
python scripts/import_kaggle_outputs.py \
  --input "$KAGGLE_OUTPUT_DIR/runs_outputs" \
  --dest results/runs \
  --dedupe \
  --validate \
  --execute
```

Then refresh the local result lake, validation reports, provenance, tables,
statistics, rank-reversal analysis, calibration/review-budget analysis, claim
gates, and paper compile:

```bash
bash scripts/post_kaggle_refresh.sh
```

Reviewer package, reproducibility capsule, and final readiness score:

```bash
python scripts/build_runs_reviewer_package.py --anonymize-check-only
python scripts/build_reproducibility_capsule.py
python scripts/score_runs_readiness.py
```

The importer defaults to dry-run unless `--execute` is passed. Duplicate
`runs.csv` rows with identical metric values are ignored. Metric conflicts are
reported under `results/runs/import_reports/` and must be resolved before any
claim upgrade.

Do not claim from partial runs: one model cannot prove rank reversal,
Elliptic-only rows cannot prove multi-dataset reversal, feature-only baselines
are not GNN evidence, missing prediction CSVs block mitigation/calibration
claims, and aggregate-only summaries do not support p-values.
