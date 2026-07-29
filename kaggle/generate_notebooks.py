#!/usr/bin/env python3
"""Generate Kaggle GPU notebooks (no GitHub import)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "kaggle" / "notebooks"


def nb(cells: list[dict], title: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "title": title,
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


SETUP_HEADER = """# Kaggle GPU setup (no GitHub)

**Required input datasets** (Add Data in the notebook):
1. **Your code bundle:** create from `kaggle/datasets/gnn-fraud-runs-code.zip` (run `bash scripts/bundle_kaggle_code.sh` locally). The Kaggle dataset name can be anything.
2. **Public:** [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

**Optional inputs:**
- `gnn-fraud-cpu-results` — folder containing `validation_clean/` from local CPU runs (skips re-training GraphSAGE).
- `dgraphfin-npz` / `tfinance-npz` — for Prompt 06 multi-dataset.

Run `00_kaggle_input_debug.ipynb` first in a fresh Kaggle session. Enable **GPU**
for training notebooks. Internet is only needed if you deliberately opt into
dependency installation.
"""

def bootstrap_cell(notebook_name: str, title: str) -> str:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return f'''# IF THIS CELL DOES NOT PRINT NAME-AGNOSTIC RECURSIVE DISCOVERY, YOU ARE USING AN OLD NOTEBOOK.
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

NOTEBOOK_NAME = "{notebook_name}"
NOTEBOOK_TITLE = "{title}"
GENERATED_AT_UTC = "{generated_at}"
INSTALL_DEPS = False

WORK = Path("/kaggle/working/gnn-fraud")
INPUT_ROOT = Path("/kaggle/input")
CORE_CODE_MARKERS = (
    "kaggle/kaggle_bootstrap.py",
    "scripts/runs_harness_common.py",
)
EXPECTED_CODE_MARKERS = (
    "kaggle/kaggle_bootstrap.py",
    "scripts/runs_harness_common.py",
    "scripts/run_validation_clean_gnn.py",
    "runs_expansion/README.md",
)

print("Notebook:", NOTEBOOK_NAME)
print("Title:", NOTEBOOK_TITLE)
print("Generated at UTC:", GENERATED_AT_UTC)
print("Install deps mode:", INSTALL_DEPS)
print("Expected code markers:", list(EXPECTED_CODE_MARKERS))

def _normalize_zip_name(name):
    path = PurePosixPath(name.replace("\\\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = [part for part in path.parts if part not in ("", ".")]
    return "/".join(parts) if parts else None

def _zip_marker_presence(zpath):
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [
                normalized
                for normalized in (_normalize_zip_name(info.filename) for info in zf.infolist())
                if normalized
            ]
    except zipfile.BadZipFile as exc:
        return {{}}, f"bad zip: {{exc}}"
    presence = {{
        marker: any(name == marker or name.endswith("/" + marker) for name in names)
        for marker in EXPECTED_CODE_MARKERS
    }}
    return presence, None

def _path_marker_presence(root):
    return {{marker: (root / marker).is_file() for marker in EXPECTED_CODE_MARKERS}}

def _missing_core(presence):
    return [marker for marker in CORE_CODE_MARKERS if not presence.get(marker, False)]

def _present(presence):
    return [marker for marker, ok in presence.items() if ok]

def _candidate_roots(input_root):
    roots = {{}}
    for marker in CORE_CODE_MARKERS:
        for hit in sorted(input_root.rglob(marker)):
            depth = len(PurePosixPath(marker).parts)
            root = hit.parents[depth - 1]
            roots.setdefault(str(root.resolve(strict=False)), root)
    return sorted(roots.values(), key=lambda p: str(p))

def _input_listing(input_root):
    if not input_root.is_dir():
        return f"{{input_root}} is not mounted"
    lines = ["top-level inputs:"]
    entries = sorted(input_root.iterdir(), key=lambda p: p.name)
    if not entries:
        return f"{{input_root}} is empty"
    for entry in entries[:80]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"- {{entry.relative_to(input_root)}}{{suffix}}")
    zips = sorted(input_root.rglob("*.zip"))
    lines.append("recursive zip files:")
    lines.extend([f"- {{z.relative_to(input_root)}}" for z in zips[:80]] or ["- none"])
    marker_hits = []
    for marker in CORE_CODE_MARKERS:
        marker_hits.extend(sorted(input_root.rglob(marker)))
    lines.append("recursive code marker hits:")
    lines.extend([f"- {{h.relative_to(input_root)}}" for h in marker_hits[:80]] or ["- none"])
    return "\\n".join(lines)

def find_code_payload(input_root=INPUT_ROOT):
    print("Name-agnostic recursive discovery under:", input_root)
    print(_input_listing(input_root))
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing Kaggle input root: {{input_root}}")

    rejected = []
    valid_zips = []
    print("Candidate zips:")
    zips = sorted(input_root.rglob("*.zip"))
    if not zips:
        print("  - none")
    for zpath in zips:
        presence, error = _zip_marker_presence(zpath)
        if error is not None:
            rejected.append(f"{{zpath}}: {{error}}")
            print("  - rejected", zpath, error)
            continue
        missing = _missing_core(presence)
        if missing:
            rejected.append(f"{{zpath}}: missing core markers {{missing}}")
            print("  - rejected", zpath, "missing", missing, "present", _present(presence))
            continue
        print("  - accepted", zpath, "present", _present(presence))
        valid_zips.append(zpath)

    valid_roots = []
    print("Candidate extracted roots:")
    roots = _candidate_roots(input_root)
    if not roots:
        print("  - none")
    for root in roots:
        presence = _path_marker_presence(root)
        missing = _missing_core(presence)
        if missing:
            rejected.append(f"{{root}}: missing core markers {{missing}}")
            print("  - rejected", root, "missing", missing, "present", _present(presence))
            continue
        print("  - accepted", root, "present", _present(presence))
        valid_roots.append(root)

    if valid_zips:
        print("Selected code source:", valid_zips[0])
        return ("zip", valid_zips[0])
    if valid_roots:
        print("Selected code source:", valid_roots[0])
        return ("extracted", valid_roots[0])
    raise FileNotFoundError(
        "No valid gnn-fraud code payload found. Expected core markers: "
        + ", ".join(CORE_CODE_MARKERS)
        + "\\nFull expected markers: "
        + ", ".join(EXPECTED_CODE_MARKERS)
        + "\\nRejected candidates:\\n- "
        + "\\n- ".join(rejected or ["none"])
    )

def _safe_extract_zip(zpath, dst):
    dst.mkdir(parents=True, exist_ok=True)
    root = dst.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            normalized = _normalize_zip_name(info.filename)
            if normalized is None:
                raise ValueError(f"Unsafe zip member path: {{info.filename!r}}")
            if not normalized:
                continue
            target = (dst / normalized).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe zip member path: {{info.filename!r}}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)

def _repo_root_under(root):
    if all((root / marker).is_file() for marker in CORE_CODE_MARKERS):
        return root
    for candidate in _candidate_roots(root):
        if all((candidate / marker).is_file() for marker in CORE_CODE_MARKERS):
            return candidate
    raise FileNotFoundError(f"Extracted zip lacks core markers under {{root}}")

def stage_code_payload(input_root=INPUT_ROOT, work_dir=WORK):
    kind, source = find_code_payload(input_root)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        tmp = work_dir.parent / ".gnn_fraud_code_extract"
        if tmp.exists():
            shutil.rmtree(tmp)
        _safe_extract_zip(source, tmp)
        repo_root = _repo_root_under(tmp)
        print("Extracted repo root:", repo_root)
        shutil.copytree(repo_root, work_dir)
        shutil.rmtree(tmp)
    else:
        shutil.copytree(source, work_dir)
    print("Staged code at:", work_dir)
    print("Selected code source:", source)
    return work_dir

stage_code_payload(INPUT_ROOT, WORK)

for marker_path in (
    WORK / "kaggle" / "BUNDLE_BUILD.txt",
    WORK / ".kaggle_bundle_build.txt",
):
    if marker_path.is_file():
        print("Repo bundle marker:", marker_path)
        print(marker_path.read_text(encoding="utf-8", errors="replace").strip())
        break
else:
    print("Repo bundle marker: not found")

bootstrap_path = WORK / "kaggle" / "kaggle_bootstrap.py"
print("Bootstrap path:", bootstrap_path)
print("Bootstrap exists:", bootstrap_path.is_file())

if not bootstrap_path.is_file():
    raise FileNotFoundError(f"Missing bootstrap file: {{bootstrap_path}}")

spec = importlib.util.spec_from_file_location("gnn_fraud_kaggle_bootstrap", bootstrap_path)
bootstrap_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
# Register before exec: Python 3.12 dataclasses (with `from __future__ import
# annotations`) resolve field types via sys.modules[cls.__module__]; if the
# module is not registered this raises AttributeError on the first @dataclass.
sys.modules[spec.name] = bootstrap_module
spec.loader.exec_module(bootstrap_module)

bootstrap = bootstrap_module.bootstrap
run_cmd = bootstrap_module.run_cmd
run_cmd_logged = bootstrap_module.run_cmd_logged

bootstrap(install_deps=INSTALL_DEPS)
'''

SAVE_CELL = """# Package results as Kaggle dataset output
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess

out = Path("/kaggle/working/runs_outputs")
out.mkdir(parents=True, exist_ok=True)
for dirname in [
    "validation_clean",
    "matched_gnn_protocol",
    "multi_dataset_protocol",
    "mitigation",
    "predictions",
    "logs",
    "manifests",
    "validation",
    "tables",
    "provenance",
]:
    (out / dirname).mkdir(parents=True, exist_ok=True)

repo = Path("/kaggle/working/gnn-fraud")
src = repo / "results" / "runs"
for dirname in [
    "validation_clean",
    "matched_gnn_protocol",
    "multi_dataset_protocol",
    "mitigation",
    "predictions",
    "validation",
]:
    source = src / dirname
    dest = out / dirname
    if source.is_dir():
        if dest.exists() and dirname != "logs":
            shutil.rmtree(dest)
        shutil.copytree(source, dest, dirs_exist_ok=True)

table_src = repo / "runs_paper" / "tables"
if table_src.is_dir():
    shutil.copytree(table_src, out / "tables", dirs_exist_ok=True)

for provenance_src in [
    repo / "runs_expansion" / "RUNS_RESULTS_PROVENANCE.md",
    repo / "gnnpaper" / "RESULTS_PROVENANCE.md",
    src / "RUNS_RESULTS_INDEX.json",
    src / "ARTIFACT_REGISTRY.json",
    src / "RUN_DATABASE.json",
]:
    if provenance_src.is_file():
        shutil.copy2(provenance_src, out / "provenance" / provenance_src.name)

bundle_source = ""
for marker_path in [repo / "kaggle" / "BUNDLE_BUILD.txt", repo / ".kaggle_bundle_build.txt"]:
    if marker_path.is_file():
        bundle_source = marker_path.read_text(encoding="utf-8", errors="replace")
        break

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    git_commit = ""

def _files_under(path):
    if not path.is_dir():
        return []
    return sorted(str(p.relative_to(out)) for p in path.rglob("*") if p.is_file())

command_log = list(getattr(bootstrap_module, "COMMAND_LOG", []))
warnings = []
errors = []
for item in command_log:
    if item.get("return_code", 0) != 0:
        errors.append(f"command failed: {item.get('cmd')}")

manifest = {
    "notebook_name": NOTEBOOK_NAME,
    "generated_notebook_marker": GENERATED_AT_UTC,
    "started_at_utc": GENERATED_AT_UTC,
    "finished_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "repo_bundle_source": bundle_source,
    "git_commit_if_available": git_commit,
    "commands_run": command_log,
    "result_files": _files_under(out / "validation_clean")
        + _files_under(out / "matched_gnn_protocol")
        + _files_under(out / "multi_dataset_protocol")
        + _files_under(out / "mitigation"),
    "prediction_files": _files_under(out / "predictions"),
    "validation_files": _files_under(out / "validation"),
    "logs": _files_under(out / "logs"),
    "warnings": warnings,
    "errors": errors,
    "claim_status": "Local import and validation required before any claim upgrade.",
    "expected_local_import_command": (
        "python scripts/import_kaggle_outputs.py "
        '--input "$KAGGLE_OUTPUT_DIR/runs_outputs" --dest results/runs '
        "--dedupe --validate --execute"
    ),
}
(out / "KAGGLE_OUTPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
(out / "manifests" / "KAGGLE_OUTPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")

(out / "README.md").write_text(
    "# RUNS Kaggle outputs\\n\\n"
    "Download the whole `runs_outputs` folder, not individual files.\\n\\n"
    "Import locally with:\\n\\n"
    "```bash\\n"
    "python scripts/import_kaggle_outputs.py \\\\\\n"
    '  --input "$KAGGLE_OUTPUT_DIR/runs_outputs" \\\\\\n'
    "  --dest results/runs \\\\\\n"
    "  --dedupe \\\\\\n"
    "  --validate \\\\\\n"
    "  --execute\\n"
    "```\\n\\n"
    "Then refresh local evidence, tables, and paper outputs with:\\n\\n"
    "```bash\\n"
    "bash scripts/post_kaggle_refresh.sh\\n"
    "```\\n\\n"
    "Do not mark results Verified unless local validation and claim gates pass.\\n",
    encoding="utf-8",
)

print("Saved to /kaggle/working/runs_outputs — download or publish as dataset")
print("Manifest:", out / "KAGGLE_OUTPUT_MANIFEST.json")
"""


def _combined_run_cell(*, p04_seeds: str) -> str:
    return f"""# --- smoke: import check ---
run_cmd([
    "-c",
    "import torch; from data.datasets import load_dataset; "
    "print('cuda', torch.cuda.is_available()); "
    "d=load_dataset('elliptic'); print('nodes', d.num_nodes, 'edges', d.num_edges)",
])

# --- P03: validation-clean all models (GraphSAGE skipped if cpu-results attached) ---
run_cmd([
    "scripts/run_validation_clean_gnn.py",
    "--dataset", "elliptic",
    "--models", "mlp,gcn,graphsage,gat",
    "--seeds", "1,2,3,4,5,6,7,8,9,10",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation_clean",
])
run_cmd([
    "scripts/validate_runs_results.py",
    "--result-file", "results/runs/validation_clean/runs.csv",
    "--output-dir", "results/runs/validation",
    "--strict",
])

from pathlib import Path
import shutil
ckpt = Path("/kaggle/working/runs_outputs_p03")
if ckpt.exists():
    shutil.rmtree(ckpt)
shutil.copytree("results/runs/validation_clean", ckpt / "validation_clean")
print("P03 checkpoint: /kaggle/working/runs_outputs_p03")

# --- P04: matched GNN ({p04_seeds.count(',') + 1} seeds x 2 protocols) ---
run_cmd([
    "scripts/run_matched_gnn_protocol_comparison.py",
    "--dataset", "elliptic",
    "--model", "graphsage",
    "--seeds", "{p04_seeds}",
    "--protocols", "chronological,random",
    "--random-split", "stratified",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/matched_gnn_protocol",
])
run_cmd([
    "scripts/summarize_matched_gnn_protocol.py",
    "--input-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/matched_gnn_protocol",
])
run_cmd([
    "scripts/validate_runs_results.py",
    "--result-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/validation",
    "--strict",
])
run_cmd([
    "scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation",
    "--strict",
])
print("Session 1 complete")
"""


def _p06_run_cell(*, datasets: str, seeds: str, title_note: str) -> str:
    return f"""# P06 — {{title_note}}
# Skips strict_inductive (already in validation_clean from Session 1).
run_cmd([
    "scripts/run_multi_dataset_protocol_benchmark.py",
    "--datasets", "{datasets}",
    "--models", "mlp,gcn,graphsage,gat",
    "--protocols", "chronological,transductive",
    "--seeds", "{seeds}",
    "--gnn-only",
    "--skip-missing-datasets",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/multi_dataset_protocol",
])
run_cmd([
    "scripts/summarize_multi_dataset_protocol.py",
    "--input-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/multi_dataset_protocol",
])
run_cmd([
    "scripts/validate_runs_results.py",
    "--result-dir", "results/runs/multi_dataset_protocol",
    "--output-dir", "results/runs/validation",
    "--strict",
])
run_cmd([
    "scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation",
    "--strict",
])
print("P06 complete")
"""


def write(name: str, title: str, extra_md: str, run_cell: str) -> None:
    cells = [
        code(bootstrap_cell(name, title)),
        md(SETUP_HEADER + "\n" + extra_md),
        code(run_cell),
        code(SAVE_CELL),
    ]
    path = NB_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells, title), indent=1), encoding="utf-8")
    print(f"[notebook] {path}")


def main() -> None:
    write(
        "00_kaggle_input_debug.ipynb",
        "00 - Kaggle input debug",
        """## First notebook to run in a fresh Kaggle session

No training. This notebook recursively lists `/kaggle/input`, stages the code
payload by content markers, imports the bootstrap by file path, runs
`bootstrap(install_deps=False)`, and checks that expected scripts exist.
""",
        """print("Debug notebook: no training commands are executed.")
if hasattr(bootstrap_module, "print_environment_versions"):
    bootstrap_module.print_environment_versions()

expected_scripts = [
    "kaggle/kaggle_bootstrap.py",
    "scripts/runs_harness_common.py",
    "scripts/run_validation_clean_gnn.py",
    "scripts/run_matched_gnn_protocol_comparison.py",
    "scripts/run_multi_dataset_protocol_benchmark.py",
    "runs_expansion/README.md",
]
for rel in expected_scripts:
    path = WORK / rel
    print(f"{rel}: {path.is_file()} -> {path}")

print("Debug smoke complete.")
""",
    )

    write(
        "00_setup_smoke_test.ipynb",
        "00 — Kaggle setup smoke test",
        "## Smoke test\nRuns import check + validation-clean dry-run (no training).\n",
        """run_cmd([
    "-c",
    "import torch; from data.datasets import load_dataset; "
    "print('cuda', torch.cuda.is_available()); "
    "d=load_dataset('elliptic'); print('nodes', d.num_nodes, 'edges', d.num_edges)",
])
run_cmd([
    "scripts/run_validation_clean_gnn.py",
    "--dataset", "elliptic",
    "--models", "graphsage",
    "--seeds", "1",
    "--dry-run",
    "--device", "cuda",
    "--output-dir", "/kaggle/working/gnn-fraud/results/runs/smoke_dryrun",
])
print("Smoke test OK")
""",
    )

    write(
        "00_01_02_combined_gpu.ipynb",
        "00+01+02 — smoke + P03 + P04 (efficiency, 5-seed P04)",
        """## Efficiency runbook (minimum time)

Same as max-results but P04 uses **5 seeds** only (~40 min saved vs 10 seeds).

Use **`00_01_02_max_results.ipynb`** unless you are time-constrained.
""",
        _combined_run_cell(p04_seeds="1,2,3,4,5"),
    )

    write(
        "00_01_02_max_results.ipynb",
        "00+01+02 — smoke + P03 + P04 MAX (recommended Session 1)",
        """## Session 1 — maximum Elliptic GNN results (~3–5 hr)

**Attach:** code bundle dataset (any name), Elliptic CSV dataset, and CPU results dataset if available.

1. Smoke check  
2. P03 — MLP/GCN/GAT/SAGE × 10 seeds (GraphSAGE skipped from CPU)  
3. P04 — matched GNN **10 seeds** × 2 protocols (max statistical power)  
4. Export + validate prediction CSVs  
5. Checkpoints + final save  

See **`kaggle/MASTER_RUNBOOK.md`** for full plan.
""",
        _combined_run_cell(p04_seeds="1,2,3,4,5,6,7,8,9,10"),
    )

    write(
        "01_p03_validation_clean_all_models.ipynb",
        "01 — P03 validation-clean all models (GPU)",
        """## Prompt 03 — validation-clean Elliptic all-model benchmark

Trains **MLP, GCN, GraphSAGE, GAT** (10 seeds). Skips existing GraphSAGE rows if you attached CPU results.

Est. **~1.5–2 hr** on Kaggle GPU.
""",
        """run_cmd([
    "scripts/run_validation_clean_gnn.py",
    "--dataset", "elliptic",
    "--models", "mlp,gcn,graphsage,gat",
    "--seeds", "1,2,3,4,5,6,7,8,9,10",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation_clean",
])
run_cmd([
    "scripts/validate_runs_results.py",
    "--result-file", "results/runs/validation_clean/runs.csv",
    "--output-dir", "results/runs/validation",
    "--strict",
])
run_cmd([
    "scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation",
    "--strict",
])
print("P03 complete")
""",
    )

    write(
        "02_p04_matched_gnn_protocol.ipynb",
        "02 — P04 matched GNN random vs chronological (GPU)",
        """## Prompt 04 — matched GNN protocol (GraphSAGE)

5 seeds × chronological + random. Est. **~40–50 min** on GPU.
""",
        """run_cmd([
    "scripts/run_matched_gnn_protocol_comparison.py",
    "--dataset", "elliptic",
    "--model", "graphsage",
    "--seeds", "1,2,3,4,5",
    "--protocols", "chronological,random",
    "--random-split", "stratified",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/matched_gnn_protocol",
])
run_cmd([
    "scripts/summarize_matched_gnn_protocol.py",
    "--input-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/matched_gnn_protocol",
])
run_cmd([
    "scripts/validate_runs_results.py",
    "--result-dir", "results/runs/matched_gnn_protocol",
    "--output-dir", "results/runs/validation",
    "--strict",
])
run_cmd([
    "scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation",
    "--strict",
])
print("P04 complete")
""",
    )

    write(
        "03_p06_elliptic_protocol_gap.ipynb",
        "03 — P06 Elliptic protocol gap only (Session 2a)",
        """## Session 2a — Elliptic chronological + transductive (~2–3 hr)

**No NPZ required.** Skips `strict_inductive` (covered by Session 1 P03).

4 models × 2 protocols × 5 seeds = 40 runs.
""",
        _p06_run_cell(
            datasets="elliptic",
            seeds="1,2,3,4,5",
            title_note="Elliptic protocol gap",
        ),
    )

    write(
        "03_p06_multidataset_max.ipynb",
        "03 — P06 three-dataset MAX (Session 2b)",
        """## Session 2b — three-dataset GNN (~8–14 hr)

**Attach:** `dgraphfin-npz`, `tfinance-npz` when ready.

Same as 2a but all datasets; skips strict_inductive to avoid duplicating P03.
5 seeds × 4 models × 2 protocols × up to 3 datasets.
""",
        _p06_run_cell(
            datasets="elliptic,dgraphfin,tfinance",
            seeds="1,2,3,4,5",
            title_note="Multi-dataset max",
        ),
    )

    write(
        "03_p06_npz_pilot.ipynb",
        "03 — P06 NPZ pilot (DGraphFin + T-Finance seed 1)",
        """## NPZ pilot — run before the max three-dataset sweep

Checks that manually staged `dgraphfin-npz` and `tfinance-npz` load, train, and
export predictions on Kaggle before spending a full session.

2 datasets × 4 models × 2 protocols × 1 seed = up to 16 runs.
""",
        _p06_run_cell(
            datasets="dgraphfin,tfinance",
            seeds="1",
            title_note="NPZ pilot",
        ),
    )

    write(
        "03_p06_dgraphfin_gnn.ipynb",
        "03 — P06 DGraphFin-only GNN sweep",
        """## DGraphFin-only split sweep

Use this instead of the max notebook if you want checkpointable progress or if
the three-dataset notebook risks exceeding the Kaggle session limit.
""",
        _p06_run_cell(
            datasets="dgraphfin",
            seeds="1,2,3,4,5",
            title_note="DGraphFin-only split",
        ),
    )

    write(
        "03_p06_tfinance_gnn.ipynb",
        "03 — P06 T-Finance-only GNN sweep",
        """## T-Finance-only split sweep

Use this instead of the max notebook if you want checkpointable progress or if
the three-dataset notebook risks exceeding the Kaggle session limit.
""",
        _p06_run_cell(
            datasets="tfinance",
            seeds="1,2,3,4,5",
            title_note="T-Finance-only split",
        ),
    )

    write(
        "03_p06_multidataset_gnn.ipynb",
        "03 — P06 legacy (all protocols, seeds 1-3)",
        """## Legacy P06 notebook

Prefer **`03_p06_elliptic_protocol_gap`** or **`03_p06_multidataset_max`** —
they skip duplicate strict_inductive training.
""",
        """run_cmd([
    "scripts/run_multi_dataset_protocol_benchmark.py",
    "--datasets", "elliptic,dgraphfin,tfinance",
    "--models", "mlp,gcn,graphsage,gat",
    "--protocols", "chronological,strict_inductive,transductive",
    "--seeds", "1,2,3",
    "--gnn-only",
    "--skip-missing-datasets",
    "--early-stopping-metric", "val_f1",
    "--scaler-mode", "train_only",
    "--device", "cuda",
    "--execute",
    "--export-predictions",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/multi_dataset_protocol",
])
run_cmd([
    "scripts/validate_prediction_artifacts.py",
    "--prediction-dir", "results/runs/predictions",
    "--output-dir", "results/runs/validation",
    "--strict",
])
print("legacy P06 complete")
""",
    )


if __name__ == "__main__":
    main()
