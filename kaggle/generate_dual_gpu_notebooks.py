#!/usr/bin/env python3
"""Generate T4 x2 (dual-GPU) Kaggle notebooks for the RUNS sweep.

These reuse the existing bootstrap + save-output scaffolding from
``generate_notebooks.py`` but drive every heavy sweep through
``scripts/run_gpu_sweep.py``, which shards seeds across both T4 GPUs and runs
two runner processes in parallel (one pinned per GPU via CUDA_VISIBLE_DEVICES).

Run locally:  python kaggle/generate_dual_gpu_notebooks.py
Output:       kaggle/notebooks/t4x2_*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

from generate_notebooks import (  # noqa: E402
    NB_DIR,
    SAVE_CELL,
    SETUP_HEADER,
    bootstrap_cell,
    code,
    md,
    nb,
)

# Model rosters. Elliptic is small enough for the full "four modern families"
# zoo; DGraphFin is ~3.7M nodes so full-graph modern models OOM on a 16 GB T4 —
# keep non-Elliptic to the classic set (see CLAUDE.md OOM note).
ZOO = "mlp,gcn,graphsage,gat,graph_transformer,gps,pcgnn,snapshot_tgn"
CLASSIC = "mlp,gcn,graphsage,gat"

SWEEP_HELPER = '''# Dual-GPU sweep helper: shards seeds across both T4s via run_gpu_sweep.py.
def sweep(runner, seeds, out, passthrough, n_gpus=2, predictions=True):
    args = ["scripts/run_gpu_sweep.py", "--runner", runner,
            "--seeds", seeds, "--n-gpus", str(n_gpus), "--output-dir", out]
    if predictions:
        args += ["--prediction-dir", "results/runs/predictions", "--export-predictions"]
    return args + ["--"] + passthrough
'''


def write(name: str, title: str, extra_md: str, run_cell: str, save: bool = True) -> None:
    cells = [code(bootstrap_cell(name, title)), md(SETUP_HEADER + "\n" + extra_md), code(run_cell)]
    if save:
        cells.append(code(SAVE_CELL))
    path = NB_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells, title), indent=1), encoding="utf-8")
    print(f"[dual-gpu notebook] {path}")


def main() -> None:
    # ---- 0. setup + GPU check (no training) ----------------------------------
    write(
        "t4x2_00_setup_check.ipynb",
        "T4x2 00 — setup + dual-GPU check",
        """## Run first. Confirms T4 x2 is visible and previews the sharding.
No training. Enable the **GPU T4 x2** accelerator in notebook settings.
""",
        '''import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  -", i, torch.cuda.get_device_name(i))
if torch.cuda.device_count() < 2:
    print("WARNING: fewer than 2 GPUs visible — set accelerator to 'GPU T4 x2'.")

run_cmd(["-c",
    "import torch; from data.datasets import load_dataset; "
    "d=load_dataset('elliptic'); print('elliptic nodes', d.num_nodes, 'edges', d.num_edges)"])

# Preview how seeds shard across the two GPUs (no processes launched).
run_cmd(["scripts/run_gpu_sweep.py",
    "--runner", "scripts/run_validation_clean_gnn.py",
    "--seeds", "1-10", "--n-gpus", "2",
    "--output-dir", "results/runs/validation_clean", "--plan-only",
    "--", "--dataset", "elliptic", "--models", "mlp,gcn,graphsage,gat"])
print("Setup check OK")
''',
        save=False,
    )

    # ---- 1. Elliptic validation-clean, full zoo, 10 seeds --------------------
    write(
        "t4x2_01_elliptic_validation_clean.ipynb",
        "T4x2 01 — Elliptic validation-clean (full model zoo, 10 seeds)",
        f"""## Strict-inductive, validation-clean leaderboard on Elliptic (~2.5–4 hr)
Full zoo (`{ZOO}`) × 10 seeds, sharded across both T4s. Early stopping on
validation only; train-only scaler; predictions exported for the TPC/mitigation
studies.
""",
        SWEEP_HELPER + f'''
run_cmd(sweep(
    "scripts/run_validation_clean_gnn.py", "1-10",
    "results/runs/validation_clean",
    ["--dataset", "elliptic", "--models", "{ZOO}",
     "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
))
run_cmd(["scripts/validate_runs_results.py",
    "--result-file", "results/runs/validation_clean/runs.csv",
    "--output-dir", "results/runs/validation", "--strict"])
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 01 complete")
''',
    )

    # ---- 2. Elliptic protocol gap (random/chronological/transductive) --------
    write(
        "t4x2_02_elliptic_protocol_gap.ipynb",
        "T4x2 02 — Elliptic protocol gap (full zoo)",
        f"""## Protocol-gap arms on Elliptic (~3–5 hr)
`{ZOO}` × {{random, chronological, transductive}} × 5 seeds, dual-GPU.
`strict_inductive` is skipped here (covered by notebook 01); the rank-reversal
analysis combines both result dirs.
""",
        SWEEP_HELPER + f'''
run_cmd(sweep(
    "scripts/run_multi_dataset_protocol_benchmark.py", "1-5",
    "results/runs/multi_dataset_protocol",
    ["--datasets", "elliptic", "--models", "{ZOO}",
     "--protocols", "random,chronological,transductive",
     "--gnn-only", "--skip-missing-datasets",
     "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
))
run_cmd(["scripts/summarize_multi_dataset_protocol.py",
    "--input-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/multi_dataset_protocol"])
run_cmd(["scripts/validate_runs_results.py",
    "--result-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/validation", "--strict"])
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 02 complete")
''',
    )

    # ---- 3. Matched GNN random-vs-chronological -----------------------------
    write(
        "t4x2_03_matched_gnn.ipynb",
        "T4x2 03 — matched GNN random vs chronological",
        """## Paired same-split protocol test (~1–1.5 hr)
GraphSAGE and GCN, chronological vs random, 10 seeds each, dual-GPU. This is the
clean paired design (matched random split per seed) backing the headline
protocol p-values; predictions exported for TPC.
""",
        SWEEP_HELPER + '''
for model in ["graphsage", "gcn"]:
    run_cmd(sweep(
        "scripts/run_matched_gnn_protocol_comparison.py", "1-10",
        "results/runs/matched_gnn_protocol",
        ["--dataset", "elliptic", "--model", model,
         "--protocols", "chronological,random", "--random-split", "stratified",
         "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
    ))
run_cmd(["scripts/summarize_matched_gnn_protocol.py",
    "--input-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/matched_gnn_protocol"])
run_cmd(["scripts/validate_runs_results.py",
    "--result-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/validation", "--strict"])
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 03 complete")
''',
    )

    # ---- 4. NPZ pilot --------------------------------------------------------
    write(
        "t4x2_04_npz_pilot.ipynb",
        "T4x2 04 — DGraphFin/T-Finance NPZ pilot",
        f"""## Pilot before the full non-Elliptic sweeps (~30–90 min)
**Attach** `dgraphfin.npz` and/or `tfinance.npz` (any dataset name). Runs the
classic set (`{CLASSIC}`) for 2 seeds × all protocols to confirm the NPZs load,
train, and export predictions, and to surface OOM **before** a full session.
Missing datasets are skipped, not failed.
""",
        SWEEP_HELPER + f'''
run_cmd(sweep(
    "scripts/run_multi_dataset_protocol_benchmark.py", "1-2",
    "results/runs/multi_dataset_protocol",
    ["--datasets", "dgraphfin,tfinance", "--models", "{CLASSIC}",
     "--protocols", "random,chronological,strict_inductive,transductive",
     "--gnn-only", "--skip-missing-datasets",
     "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
))
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 04 (pilot) complete — inspect for OOM before full sweeps")
''',
    )

    # ---- 5. DGraphFin full sweep --------------------------------------------
    write(
        "t4x2_05_dgraphfin.ipynb",
        "T4x2 05 — DGraphFin full sweep (classic models)",
        f"""## DGraphFin GNN sweep (~3–7 hr, OOM risk)
**Attach** `dgraphfin.npz`. Classic set (`{CLASSIC}`) × all protocols × 5 seeds,
dual-GPU. **Stop rule:** if GAT/SAGE OOM, drop the offending model and keep the
DGraphFin gate Pending until a sampling harness exists — do not claim partial.
""",
        SWEEP_HELPER + f'''
run_cmd(sweep(
    "scripts/run_multi_dataset_protocol_benchmark.py", "1-5",
    "results/runs/multi_dataset_protocol",
    ["--datasets", "dgraphfin", "--models", "{CLASSIC}",
     "--protocols", "random,chronological,strict_inductive,transductive",
     "--gnn-only", "--skip-missing-datasets",
     "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
))
run_cmd(["scripts/summarize_multi_dataset_protocol.py",
    "--input-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/multi_dataset_protocol"])
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 05 complete")
''',
    )

    # ---- 6. T-Finance full sweep --------------------------------------------
    write(
        "t4x2_06_tfinance.ipynb",
        "T4x2 06 — T-Finance full sweep",
        f"""## T-Finance GNN sweep (~2–5 hr)
**Attach** `tfinance.npz`. T-Finance is small enough for the full zoo; this uses
the classic set by default (`{CLASSIC}`) for speed — edit `MODELS` to `{ZOO}` to
add the modern families.
""",
        SWEEP_HELPER + f'''
MODELS = "{CLASSIC}"   # set to "{ZOO}" to include modern + temporal families
run_cmd(sweep(
    "scripts/run_multi_dataset_protocol_benchmark.py", "1-5",
    "results/runs/multi_dataset_protocol",
    ["--datasets", "tfinance", "--models", MODELS,
     "--protocols", "random,chronological,strict_inductive,transductive",
     "--gnn-only", "--skip-missing-datasets",
     "--early-stopping-metric", "val_f1", "--scaler-mode", "train_only"],
))
run_cmd(["scripts/summarize_multi_dataset_protocol.py",
    "--input-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/multi_dataset_protocol"])
run_cmd(["scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation", "--strict"])
print("Notebook 06 complete")
''',
    )

    # ---- 7. Post-processing (CPU): TPC, mitigation, stats, tables ------------
    write(
        "t4x2_07_postprocess.ipynb",
        "T4x2 07 — post-processing (CPU: TPC, mitigation, stats, tables)",
        """## CPU analysis after the GPU sweeps (~10–25 min, no GPU needed)
Runs on whatever prediction CSVs and result rows exist: TPC+TTA mechanism eval,
mitigation sweep, statistics (paired tests + effect sizes), rank-reversal,
calibration/budget, paper-table export, provenance, claim gates, readiness.
Can also be run locally after downloading `runs_outputs`.
""",
        '''# Rebuild every runs.csv from per-cell JSONs (race-free after parallel sweeps).
for d in ["validation_clean", "matched_gnn_protocol", "multi_dataset_protocol"]:
    run_cmd(["scripts/aggregate_runs_csv.py", "--result-dir", f"results/runs/{d}"])

# TPC+TTA mechanism eval over all exported prediction CSVs (raw/temp/prior/thresh/full).
run_cmd(["scripts/run_tpc_tta_eval.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/tpc_tta", "--execute"])

# Training-free mitigation sweep over predictions.
run_cmd(["scripts/run_feature_mitigation_sweep.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/mitigation", "--execute"])

# Statistics, rank-reversal, calibration/budget.
run_cmd(["scripts/analyze_runs_statistics.py", "--output-dir", "results/runs/statistics"])
run_cmd(["scripts/analyze_runs_rank_reversal.py", "--output-dir", "results/runs/rank_reversal"])
run_cmd(["scripts/analyze_runs_calibration_and_budget.py", "--prediction-dir", "results/runs/predictions"])

# Provenance, registries, paper tables, claim gates, readiness score.
run_cmd(["scripts/build_runs_results_provenance.py"])
run_cmd(["scripts/build_runs_run_database.py"])
run_cmd(["scripts/build_runs_artifact_registry.py"])
run_cmd(["scripts/export_runs_paper_tables.py", "--output-dir", "runs_paper/tables"])
run_cmd(["scripts/check_claim_gates.py"])
run_cmd(["scripts/score_runs_readiness.py"])

# Stage the analysis dirs into runs_outputs so they download with everything else.
import shutil
from pathlib import Path
out = Path("/kaggle/working/runs_outputs")
for d in ["tpc_tta", "statistics", "rank_reversal", "calibration", "review_budget"]:
    src = Path("results/runs") / d
    if src.is_dir():
        shutil.copytree(src, out / d, dirs_exist_ok=True)
print("Notebook 07 complete")
''',
    )


if __name__ == "__main__":
    main()
