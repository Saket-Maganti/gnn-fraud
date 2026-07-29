#!/usr/bin/env python3
"""Generate deterministic, non-executing Kaggle and local orchestration notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KAGGLE = (
    ("01_bootstrap_validate", "Environment bootstrap and repository validation", "bootstrap"),
    ("02_elliptic_screening", "Elliptic five-seed screening wave", "elliptic"),
    ("03_dgraphfin_screening", "DGraphFin five-seed screening wave", "dgraphfin"),
    ("04_good_screening", "GOOD external-validity screening wave", "good"),
    ("05_router_screening", "CoReRouter held-out-combination screening", "router"),
    ("06_ablation_wave", "CoReGraph ablation wave", "ablation"),
    ("07_final_ten_seed", "Confirmatory ten-seed wave", "final"),
    ("08_synthetic_theory", "Synthetic mechanism and theory checks", "synthetic"),
    ("09_package_outputs", "Checksum and package completed outputs", "package"),
)
LOCAL = (
    ("01_saved_output_pilot", "Saved-output pilot", "pilot"),
    ("02_synthetic_smoke", "Synthetic deterministic smoke", "synthetic"),
    ("03_import_analysis", "Manifest import and analysis", "analysis"),
)


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def notebook(title: str, wave: str, *, kaggle: bool) -> dict[str, object]:
    matrix = {
        "elliptic": "SCREENING_5SEED_GRID.csv",
        "dgraphfin": "SCREENING_5SEED_GRID.csv",
        "good": "GOOD_GRID.csv",
        "router": "SCREENING_5SEED_GRID.csv",
        "ablation": "ABLATION_GRID.csv",
        "final": "FINAL_10SEED_GRID.csv",
        "synthetic": "THEORY_SYNTHETIC_GRID.csv",
    }.get(wave, "MASTER_EXPERIMENT_MATRIX.csv")
    execution = (
        "Execution is disabled by default. Set EXECUTE=True only after all "
        "prerequisites and upstream provider/licence gates pass."
    )
    cells = [
        markdown_cell(
            f"# {title}\n\n{execution}\n\n"
            "This notebook never downloads provider data and never treats two GPUs "
            "as data-parallel unless a runner explicitly declares that support."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import csv, hashlib, json, os, subprocess, sys\n"
            "REPO = Path('/kaggle/working/gnn-fraud') if Path('/kaggle').exists() "
            "else Path.cwd()\n"
            "EXECUTE = False\n"
            f"WAVE = {wave!r}\n"
            f"MATRIX = REPO / 'configs/coregraph/run_matrices/{matrix}'\n"
            "print({'repo': str(REPO), 'wave': WAVE, 'execute': EXECUTE})"
        ),
        code_cell(
            "required = [REPO / 'coregraph', MATRIX, "
            "REPO / 'results/coregraph_build/ANALYSIS_PLAN_FREEZE.json']\n"
            "missing = [str(path) for path in required if not path.exists()]\n"
            "assert not missing, f'Missing prerequisites: {missing}'\n"
            "rows = list(csv.DictReader(MATRIX.open()))\n"
            "print({'matrix_rows': len(rows), 'matrix_sha256': "
            "hashlib.sha256(MATRIX.read_bytes()).hexdigest()})"
        ),
        code_cell(
            "selected = [row for row in rows if row['execution_status'] == 'PLANNED']\n"
            "print({'planned_rows': len(selected), 'blocked_rows': len(rows)-len(selected)})\n"
            "assert all(row['runtime_status'] == 'TBD_PROFILE' for row in selected)\n"
            "# Deterministic sharding across two T4 scheduling lanes; this is not DDP.\n"
            "lanes = {0: selected[0::2], 1: selected[1::2]}\n"
            "print({f't4_lane_{lane}': len(items) for lane, items in lanes.items()})"
        ),
        code_cell(
            "if EXECUTE:\n"
            "    raise RuntimeError('Heavy execution entry point intentionally requires "
            "the runbook command and staged manifests; do not toggle ad hoc.')\n"
            "print('DRY_RUN_VALIDATED')"
        ),
    ]
    metadata: dict[str, object] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "coregraph": {
            "schema": "coregraph_notebook_v1",
            "wave": wave,
            "accelerator_envelope": "T4x2" if kaggle else "local_cpu",
            "heavy_execution_default": False,
        },
    }
    if kaggle:
        metadata["kaggle"] = {
            "accelerator": "gpu",
            "dataSources": [],
            "dockerImageVersionId": None,
            "isInternetEnabled": False,
            "language": "python",
        }
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    for name, title, wave in KAGGLE:
        path = ROOT / f"kaggle/coregraph/{name}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(notebook(title, wave, kaggle=True), indent=1) + "\n",
            encoding="utf-8",
        )
    for name, title, wave in LOCAL:
        path = ROOT / f"notebooks/coregraph/{name}.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(notebook(title, wave, kaggle=False), indent=1) + "\n",
            encoding="utf-8",
        )
    print(f"generated {len(KAGGLE)} Kaggle and {len(LOCAL)} local notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
