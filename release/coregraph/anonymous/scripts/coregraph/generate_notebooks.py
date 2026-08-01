#!/usr/bin/env python3
"""Generate deterministic, non-executing Kaggle and local orchestration notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KAGGLE = (
    ("01_saved_output_pilot", "Saved-output pilot", "saved_output_pilot"),
    ("02_fraud_full_training", "Fraud full training", "fraud_full_training"),
    ("03_fraud_baseline_waves", "Fraud baseline waves", "fraud_baseline_waves"),
    ("04_synthetic_suite", "Controlled synthetic suite", "synthetic_suite"),
    ("05_graph_ood_benchmark", "Non-fraud graph-OOD benchmark", "graph_ood_benchmark"),
    ("06_resource_profiling", "Resource profiling", "resource_profiling"),
    ("07_ablation_waves", "Ablation waves", "ablation_waves"),
    ("08_prediction_regeneration", "Prediction regeneration", "prediction_regeneration"),
    (
        "09_output_validation_packaging",
        "Output validation and packaging",
        "output_validation_packaging",
    ),
)
LOCAL = (
    ("01_saved_output_pilot", "Saved-output pilot local companion", "saved_output_pilot"),
    ("02_synthetic_smoke", "Synthetic deterministic smoke", "synthetic_suite"),
    (
        "03_import_analysis",
        "Manifest import and output validation",
        "output_validation_packaging",
    ),
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
    execution = (
        "Execution is disabled by default. Set EXECUTE=True only after all "
        "prerequisites and upstream provider/licence gates pass."
    )
    cells = [
        markdown_cell(
            f"# {title}\n\n{execution}\n\n"
            "This Level-4 T4x2 runbook verifies code and data identities before creating "
            "a run manifest. It shards independent coordinates across two scheduling "
            "lanes; it does not silently turn two GPUs into unsupported data parallelism.\n\n"
            "Safety boundary: never use target labels for fitting or threshold selection; "
            "never compute an oracle as a deployable input; never silently skip a failure."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import csv, hashlib, json, os, shutil, subprocess, sys, time, zipfile\n"
            "REPO = Path(os.environ.get('COREGRAPH_REPO_ROOT', '/kaggle/working/gnn-fraud')) "
            "if Path('/kaggle').exists() else Path.cwd()\n"
            "CACHE = Path(os.environ.get('COREGRAPH_EVIDENCE_CACHE', "
            "'/kaggle/input/coregraph-evidence-cache'))\n"
            "OUTPUT_ROOT = Path(os.environ.get('COREGRAPH_OUTPUT_ROOT', "
            "'/kaggle/working/coregraph-level4-output'))\n"
            "EXECUTE = False\n"
            f"WAVE = {wave!r}\n"
            "SEEDS = tuple(range(1, 11))\n"
            "GPU_LANES = (0, 1)\n"
            "MATRIX = REPO / 'results/coregraph_build/LEVEL4_FULL_RUN_MATRIX.csv'\n"
            "RUN_ROOT = OUTPUT_ROOT / WAVE\n"
            "RUN_MANIFEST = RUN_ROOT / 'run_manifest.json'\n"
            "COMPLETION_REPORT = RUN_ROOT / 'completion_report.json'\n"
            "FINAL_ZIP = OUTPUT_ROOT / f'{WAVE}_validated_outputs.zip'\n"
            "print({'repo': str(REPO), 'wave': WAVE, 'seeds': SEEDS, "
            "'gpu_lanes': GPU_LANES, 'execute': EXECUTE})"
        ),
        code_cell(
            "required = [REPO / 'coregraph', MATRIX, REPO / 'PROJECT_STATE_AND_AUTHORITY.md', "
            "REPO / 'results/coregraph_build/LEVEL4_PREREGISTRATION_HASH.txt']\n"
            "missing = [str(path) for path in required if not path.exists()]\n"
            "assert not missing, f'Missing prerequisites: {missing}'\n"
            "commit_sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', "
            "'HEAD'], text=True).strip()\n"
            "branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', "
            "'--show-current'], text=True).strip()\n"
            "assert branch == 'codex/coregraph-iclr-buildout-2026', branch\n"
            "print({'commit_sha': commit_sha, 'branch': branch})"
        ),
        code_cell(
            "evidence_manifest = REPO / 'results/coregraph_build/EVIDENCE_CACHE_MANIFEST.csv'\n"
            "evidence_rows = list(csv.DictReader(evidence_manifest.open()))\n"
            "archives = [row for row in evidence_rows if row['record_type'] == 'archive']\n"
            "assert len(archives) == 6\n"
            "dataset_hashes = {}\n"
            "for row in archives:\n"
            "    archive = CACHE / 'archives' / row['archive']\n"
            "    assert archive.is_file(), f'Missing canonical archive: {archive.name}'\n"
            "    digest = hashlib.sha256(archive.read_bytes()).hexdigest()\n"
            "    assert digest == row['expected_sha256'], f'Hash mismatch: {archive.name}'\n"
            "    dataset_hashes[archive.name] = digest\n"
            "print({'verified_dataset_hashes': dataset_hashes})"
        ),
        code_cell(
            "rows = list(csv.DictReader(MATRIX.open()))\n"
            "wave_filters = {\n"
            " 'saved_output_pilot': lambda r: r['priority'] == 'PILOT_MUST_RUN',\n"
            " 'fraud_full_training': lambda r: r['dataset'] in {'elliptic','dgraphfin'} "
            "and r['priority'] == 'FULL_MUST_RUN',\n"
            " 'fraud_baseline_waves': lambda r: r['dataset'] in {'elliptic','dgraphfin'} "
            "and r['baseline'] != 'coregraph',\n"
            " 'synthetic_suite': lambda r: r['dataset'] == 'controlled_synthetic',\n"
            " 'graph_ood_benchmark': lambda r: r['dataset'] in {'GOOD','OGB_molecular_fallback'},\n"
            " 'resource_profiling': lambda r: r['held_out_composition'] in "
            "{'tight_memory','tight_latency','tight_review_budget','combined_graph_resource_shift',"
            "'dynamic_availability_change','one_graph_expert_unavailable','all_graph_experts_unavailable'},\n"
            " 'ablation_waves': lambda r: any(r[k] != 'none' for k in "
            "('objective_ablation','encoder_ablation','diagnostic_ablation')),\n"
            " 'prediction_regeneration': lambda r: r['dataset'] in {'elliptic','dgraphfin'},\n"
            " 'output_validation_packaging': lambda r: True,\n"
            "}\n"
            "selected = [row for row in rows if wave_filters[WAVE](row)]\n"
            "assert selected, f'No matrix rows for {WAVE}'\n"
            "assert set(int(row['seed']) for row in selected).issubset(SEEDS)\n"
            "matrix_sha256 = hashlib.sha256(MATRIX.read_bytes()).hexdigest()\n"
            "manifest = {'schema':'coregraph_level4_run_manifest_v1','wave':WAVE,"
            "'code_sha':commit_sha,'matrix_sha256':matrix_sha256,'dataset_hashes':dataset_hashes,"
            "'seeds':list(SEEDS),'gpu_lanes':list(GPU_LANES),'created_before_execution':True,"
            "'target_labels_used_for_fitting':False,'target_oracle_input':False,"
            "'coordinates':[row['run_id'] for row in selected]}\n"
            "if EXECUTE:\n"
            "    RUN_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "    temporary = RUN_MANIFEST.with_suffix('.json.tmp')\n"
            "    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\\n')\n"
            "    temporary.replace(RUN_MANIFEST)\n"
            "    assert RUN_MANIFEST.exists(), 'Run manifest must exist before execution'\n"
            "print({'selected_coordinates':len(selected),'manifest_preview':manifest})"
        ),
        code_cell(
            "def checkpoint_path(run_id):\n"
            "    return RUN_ROOT / 'checkpoints' / f'{run_id}.json'\n"
            "def validated_complete(run_id):\n"
            "    path = checkpoint_path(run_id)\n"
            "    if not path.exists(): return False\n"
            "    payload = json.loads(path.read_text())\n"
            "    return payload.get('status') == 'COMPLETE_VALIDATED' and bool(payload.get('output_sha256'))\n"
            "resume_completed = {row['run_id'] for row in selected if validated_complete(row['run_id'])}\n"
            "pending = [row for row in selected if row['run_id'] not in resume_completed]\n"
            "lanes = {lane: pending[lane::len(GPU_LANES)] for lane in GPU_LANES}\n"
            "assert set(x['run_id'] for items in lanes.values() for x in items) == "
            "{x['run_id'] for x in pending}\n"
            "print({'resume_completed':len(resume_completed), **{f't4_lane_{k}':len(v) "
            "for k,v in lanes.items()}})"
        ),
        code_cell(
            "def classify_failure(exc):\n"
            "    message = str(exc)\n"
            "    return 'RESOURCE_BLOCKED_OOM' if 'out of memory' in message.lower() "
            "else 'FAILED_EXPLICIT'\n"
            "def execute_coordinate(row, lane):\n"
            "    command_template = os.environ.get('COREGRAPH_LEVEL4_COORDINATE_COMMAND', '')\n"
            "    if not command_template:\n"
            "        raise RuntimeError('Execution command is unset; obtain wave-specific authority first')\n"
            "    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(lane), "
            "COREGRAPH_RUN_ID=row['run_id'])\n"
            "    completed = subprocess.run(command_template.split(), cwd=REPO, env=environment, "
            "capture_output=True, text=True, check=False)\n"
            "    if completed.returncode:\n"
            "        raise RuntimeError(completed.stderr[-2000:] or f'exit {completed.returncode}')\n"
            "    return completed\n"
            "failures = []\n"
            "if EXECUTE:\n"
            "    assert RUN_MANIFEST.exists(), 'Refuse execution without pre-created manifest'\n"
            "    (RUN_ROOT/'checkpoints').mkdir(parents=True, exist_ok=True)\n"
            "    for lane, coordinates in lanes.items():\n"
            "        for row in coordinates:\n"
            "            checkpoint = {'run_id':row['run_id'],'lane':lane,'status':'STARTED'}\n"
            "            try:\n"
            "                execute_coordinate(row, lane)\n"
            "                output = RUN_ROOT/'coordinates'/row['run_id']/'validated_output.json'\n"
            "                assert output.is_file(), f'Missing validated prediction output: {output}'\n"
            "                checkpoint.update(status='COMPLETE_VALIDATED', "
            "output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())\n"
            "            except Exception as exc:\n"
            "                checkpoint.update(status=classify_failure(exc), error=str(exc))\n"
            "                failures.append(checkpoint.copy())\n"
            "            finally:\n"
            "                path = checkpoint_path(row['run_id']); path.parent.mkdir(parents=True,exist_ok=True)\n"
            "                temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(checkpoint,sort_keys=True)+'\\n')\n"
            "                temp.replace(path)\n"
            "    assert not failures, f'Failed coordinates may not be silently skipped: {failures[:3]}'\n"
            "print({'execution_status':'ENABLED' if EXECUTE else 'DRY_RUN_VALIDATED', "
            "'failure_policy':'EXPLICIT_WITH_OOM_CLASSIFICATION'})"
        ),
        code_cell(
            "completion = {'schema':'coregraph_level4_completion_v1','wave':WAVE,"
            "'code_sha':commit_sha,'planned':len(selected),'validated_complete':len(resume_completed),"
            "'failures':failures,'final_zip':FINAL_ZIP.name}\n"
            "if EXECUTE:\n"
            "    checkpoints = [json.loads(checkpoint_path(row['run_id']).read_text()) for row in selected]\n"
            "    invalid = [x for x in checkpoints if x.get('status') != 'COMPLETE_VALIDATED']\n"
            "    assert not invalid, f'Packaging blocked by incomplete coordinates: {invalid[:3]}'\n"
            "    completion['validated_complete'] = len(checkpoints)\n"
            "    COMPLETION_REPORT.write_text(json.dumps(completion,indent=2,sort_keys=True)+'\\n')\n"
            "    if FINAL_ZIP.exists(): FINAL_ZIP.unlink()\n"
            "    with zipfile.ZipFile(FINAL_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as archive:\n"
            "        for path in sorted(RUN_ROOT.rglob('*')):\n"
            "            if path.is_file(): archive.write(path,path.relative_to(OUTPUT_ROOT))\n"
            "    zip_sha256 = hashlib.sha256(FINAL_ZIP.read_bytes()).hexdigest()\n"
            "    completion['zip_sha256'] = zip_sha256\n"
            "    COMPLETION_REPORT.write_text(json.dumps(completion,indent=2,sort_keys=True)+'\\n')\n"
            "    assert zipfile.is_zipfile(FINAL_ZIP)\n"
            "print({'completion_report':completion,'one_final_downloadable_zip':FINAL_ZIP.name})"
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
            "seeds": list(range(1, 11)),
            "failure_semantics": "explicit_with_resource_blocked_oom",
            "idempotent_resume": True,
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
