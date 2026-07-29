"""
Heavy-sweep planning and execution utilities.

This module turns the multi-dataset revision into a resumable job plan. It can
inspect expected result artifacts, select missing/failed jobs, write Markdown
and CSV manifests, emit a runnable shell script, and optionally execute jobs.

By default it only plans. Execution is guarded by the CLI layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from experiments.result_audit import (
    DEFAULT_DATASETS,
    SWEEP_SPECS,
    ArtifactAuditRow,
    audit_sweep,
    summarize_rows,
)


DEFAULT_ACTION_STATUSES = (
    "MISSING",
    "ERROR_RESULT",
    "INVALID_JSON",
    "INVALID_SCHEMA",
)


RUNNERS = {
    "multi": "experiments/run_multi_dataset.py",
    "shuffle": "experiments/run_shuffle_ablation_multi.py",
    "tpc_tta": "experiments/run_tpc_tta.py",
}


@dataclass(frozen=True)
class SweepCommandOptions:
    python_bin: str = "python"
    epochs: int = 200
    hidden: int = 256
    layers: int = 3
    dropout: float = 0.5
    lr: float = 1e-3
    patience: int = 40
    device: str = "auto"
    scaler_mode: str = "train_only"
    window: int = 3


@dataclass
class SweepJob:
    job_id: str
    sweep: str
    dataset: str
    model: str
    seed: int
    status: str
    result_path: str
    detail: str
    command: str
    command_sha256: str = ""


def _quote_parts(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def job_id_for(row: ArtifactAuditRow) -> str:
    return f"{row.sweep}:{row.dataset}:{row.model}:seed{row.seed}"


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_for_job(
    row: ArtifactAuditRow,
    options: SweepCommandOptions,
) -> str:
    runner = RUNNERS[row.sweep]
    parts: List[object] = [
        options.python_bin,
        runner,
        "--datasets",
        row.dataset,
        "--models",
        row.model,
        "--seeds",
        row.seed,
        "--epochs",
        options.epochs,
        "--hidden",
        options.hidden,
        "--layers",
        options.layers,
        "--dropout",
        options.dropout,
        "--lr",
        options.lr,
        "--patience",
        options.patience,
        "--device",
        options.device,
        "--scaler-mode",
        options.scaler_mode,
    ]
    if row.sweep == "tpc_tta":
        parts.extend(["--window", options.window])
    result_dir = os.path.dirname(row.path) or "."
    parts.extend(["--out", result_dir])
    return _quote_parts(parts)


def selected_sweeps(name: str) -> List[str]:
    if name == "all":
        return sorted(SWEEP_SPECS.keys())
    if name not in SWEEP_SPECS:
        raise ValueError(f"Unknown sweep '{name}'. Available: all, {sorted(SWEEP_SPECS)}")
    return [name]


def _stable_key(row: ArtifactAuditRow) -> tuple:
    return (row.sweep, row.dataset, row.model, row.seed)


def select_rows(
    rows: Sequence[ArtifactAuditRow],
    statuses: Sequence[str] = DEFAULT_ACTION_STATUSES,
    limit: Optional[int] = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> List[ArtifactAuditRow]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    status_set = set(statuses)
    ordered = sorted((row for row in rows if row.status in status_set), key=_stable_key)
    sharded = [
        row for idx, row in enumerate(ordered)
        if idx % num_shards == shard_index
    ]
    if limit is not None:
        sharded = sharded[:limit]
    return sharded


def build_jobs(
    sweep_name: str = "all",
    datasets: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    result_dirs: Optional[Dict[str, str]] = None,
    statuses: Sequence[str] = DEFAULT_ACTION_STATUSES,
    limit: Optional[int] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    command_options: Optional[SweepCommandOptions] = None,
) -> List[SweepJob]:
    command_options = command_options or SweepCommandOptions()
    datasets = tuple(datasets or DEFAULT_DATASETS)
    jobs: List[SweepJob] = []

    for sweep in selected_sweeps(sweep_name):
        rows = audit_sweep(
            sweep,
            result_dir=(result_dirs or {}).get(sweep),
            datasets=datasets,
            models=models,
            seeds=seeds,
        )
        for row in select_rows(
            rows,
            statuses=statuses,
            limit=None,
            shard_index=0,
            num_shards=1,
        ):
            command = command_for_job(row, command_options)
            jobs.append(SweepJob(
                job_id=job_id_for(row),
                sweep=row.sweep,
                dataset=row.dataset,
                model=row.model,
                seed=row.seed,
                status=row.status,
                result_path=row.path,
                detail=row.detail,
                command=command,
                command_sha256=_text_sha256(command),
            ))

    jobs.sort(key=lambda job: (job.sweep, job.dataset, job.model, job.seed))
    if num_shards != 1 or shard_index != 0:
        jobs = [
            job for idx, job in enumerate(jobs)
            if idx % num_shards == shard_index
        ]
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def summarize_jobs(jobs: Sequence[SweepJob]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for job in jobs:
        key = f"{job.sweep}:{job.status}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _escape_md(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_job_outputs(
    jobs: Sequence[SweepJob],
    out_dir: str,
    repo_root: Optional[str] = None,
    prefix: str = "heavy_sweep_jobs",
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths = {
        "json": os.path.join(out_dir, f"{prefix}.json"),
        "csv": os.path.join(out_dir, f"{prefix}.csv"),
        "md": os.path.join(out_dir, f"{prefix}.md"),
        "sh": os.path.join(out_dir, f"{prefix}.sh"),
    }

    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump([asdict(job) for job in jobs], fh, indent=2)
        fh.write("\n")

    with open(paths["csv"], "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "job_id",
                "sweep",
                "dataset",
                "model",
                "seed",
                "status",
                "result_path",
                "detail",
                "command",
                "command_sha256",
            ),
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))

    counts = summarize_jobs(jobs)
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write("# Heavy Sweep Job Plan\n\n")
        fh.write(
            "This file is a generated plan for missing or invalid heavy-sweep "
            "artifacts. It does not mean these jobs have been run.\n\n"
        )
        fh.write(f"- Selected jobs: {len(jobs)}\n")
        for key in sorted(counts):
            fh.write(f"- {key}: {counts[key]}\n")
        fh.write("\n")
        fh.write("| job | sweep | dataset | model | seed | status | command hash | result | command |\n")
        fh.write("| --- | --- | --- | --- | ---: | --- | --- | --- | --- |\n")
        for job in jobs:
            fh.write(
                f"| `{_escape_md(job.job_id)}` | {_escape_md(job.sweep)} | {_escape_md(job.dataset)} | "
                f"{_escape_md(job.model)} | {job.seed} | {_escape_md(job.status)} | "
                f"`{job.command_sha256[:12]}` | "
                f"`{_escape_md(job.result_path)}` | `{_escape_md(job.command)}` |\n"
            )

    with open(paths["sh"], "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("set -euo pipefail\n")
        fh.write(f"cd {shlex.quote(repo_root)}\n\n")
        fh.write("# Generated heavy-sweep command file.\n")
        fh.write("# Running this script can train models and take a long time.\n\n")
        for job in jobs:
            fh.write(f"# job_id: {job.job_id}\n")
            fh.write(f"# command_sha256: {job.command_sha256}\n")
            fh.write(
                f"echo '[{job.sweep}] {job.dataset}/{job.model}/seed{job.seed}'\n"
            )
            fh.write(job.command + "\n\n")
    os.chmod(paths["sh"], 0o755)
    return paths


def execute_jobs(
    jobs: Sequence[SweepJob],
    repo_root: Optional[str] = None,
    stop_on_error: bool = True,
) -> List[Dict[str, object]]:
    repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outcomes: List[Dict[str, object]] = []
    for job in jobs:
        proc = subprocess.run(  # noqa: S603 - command is generated from repo arguments
            job.command,
            cwd=repo_root,
            shell=True,  # noqa: S602 - commands are quoted by command_for_job
            text=True,
        )
        outcome = {
            "sweep": job.sweep,
            "dataset": job.dataset,
            "model": job.model,
            "seed": job.seed,
            "returncode": proc.returncode,
        }
        outcomes.append(outcome)
        if proc.returncode != 0 and stop_on_error:
            break
    return outcomes
