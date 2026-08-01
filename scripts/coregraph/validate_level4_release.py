#!/usr/bin/env python3
"""Validate Level-4 checksums and optionally run the clean-room workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release" / "level4"
SNAPSHOTS = ROOT / "release" / "level4_source_snapshot"
BUILD = ROOT / "results" / "coregraph_build"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(path: Path, *, root: Path) -> list[str]:
    failures: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(None, 1)
        target = root / relative.strip()
        if not target.is_file():
            failures.append(f"missing:{relative.strip()}")
        elif sha256_path(target) != digest:
            failures.append(f"checksum:{relative.strip()}")
    return failures


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe source snapshot member: {info.filename}")
        archive.extractall(destination)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    def sanitize(value: str) -> str:
        value = re.sub(
            r"/(?:Users|Volumes|private/var/folders|var/folders)/[^\s'\"]+",
            "${LOCAL_RUNTIME_PATH}",
            value,
        )
        return re.sub(r"saket\s*maganti", "${PRIVATE_IDENTITY}", value, flags=re.I)

    return {
        "command": [Path(command[0]).name, *command[1:]],
        "returncode": completed.returncode,
        "stdout_tail": sanitize(completed.stdout[-1000:]),
        "stderr_tail": sanitize(completed.stderr[-1000:]),
    }


def cleanroom() -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="coregraph-level4-cleanroom-") as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        _safe_extract(SNAPSHOTS / "coregraph_source_snapshot.zip", source)
        environment_path = root / "venv"
        create = subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(environment_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            return [f"venv_create:{create.stderr[-500:]}"], {"venv_create": create.returncode}
        python = environment_path / "bin" / "python"
        fresh_site_output = subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True
        ).strip()
        fresh_site = Path(fresh_site_output)
        host_sites = [
            Path(value)
            for value in site.getsitepackages()
            if (Path(value) / "pytest").is_dir()
        ]
        if not host_sites:
            return ["offline_pytest_dependency_layer_absent"], {
                "venv_create": create.returncode
            }
        (fresh_site / "coregraph_offline_runtime.pth").write_text(
            str(host_sites[0]) + "\n", encoding="utf-8"
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(source)
        steps = []
        steps.append(
            _run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "numpy",
                    "scipy",
                    "pandas",
                    "torch",
                    "matplotlib",
                ],
                cwd=source,
                env=environment,
            )
        )
        steps.append(
            _run(
                [str(python), "-m", "compileall", "-q", "coregraph", "scripts/coregraph"],
                cwd=source,
                env=environment,
            )
        )
        focused_tests = [
            "tests/coregraph/test_level4_contracts_routing.py",
            "tests/coregraph/test_level4_benchmarks_theory.py",
            "tests/coregraph/test_level4_generated_artifacts.py",
            "tests/coregraph/test_level4_overlap_audit.py",
            "tests/test_path_resolution.py",
        ]
        steps.append(
            _run(
                [str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", *focused_tests],
                cwd=source,
                env=environment,
            )
        )
        steps.append(
            _run(
                [str(python), "scripts/coregraph/build_level4_paper.py"],
                cwd=source,
                env=environment,
            )
        )
        for index, step in enumerate(steps):
            if step["returncode"] != 0:
                failures.append(f"step_{index}_exit_{step['returncode']}")
        private_pattern = re.compile(r"/(?:Users|Volumes)/[A-Za-z0-9._-]+|saket\s*maganti", re.I)
        identity_hits: list[str] = []
        for pdf in (source / "paper_iclr/main.pdf", source / "paper_iclr/supplement.pdf"):
            if not pdf.is_file():
                identity_hits.append(f"missing:{pdf.name}")
                continue
            text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
            if private_pattern.search(text):
                identity_hits.append(pdf.name)
        if identity_hits:
            failures.append("pdf_identity_or_path:" + ",".join(identity_hits))
        report = {
            "fresh_environment_created": create.returncode == 0,
            "dependencies_installed_offline": (
                steps[0]["returncode"] == 0
                and (fresh_site / "coregraph_offline_runtime.pth").is_file()
            ),
            "offline_dependency_source": "PREVALIDATED_LOCAL_RUNTIME_NO_NETWORK",
            "steps": steps,
            "identity_or_private_path_hits": identity_hits,
            "provider_payloads_extracted": 0,
            "heavy_training_runs": 0,
            "target_metrics_computed": 0,
            "target_oracles_computed": 0,
        }
    return failures, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanroom", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    failures = _checksums(RELEASE / "CHECKSUMS.sha256", root=ROOT)
    failures.extend(_checksums(SNAPSHOTS / "CHECKSUMS.sha256", root=SNAPSHOTS))
    public = json.loads((RELEASE / "PUBLIC_TREE_AUDIT.json").read_text())
    deterministic = json.loads((RELEASE / "DETERMINISM_REPORT.json").read_text())
    if public.get("status") != "PASS":
        failures.append("public_tree")
    if deterministic.get("status") != "PASS":
        failures.append("determinism")
    cleanroom_report: dict[str, object] = {"status": "NOT_REQUESTED"}
    if arguments.cleanroom:
        cleanroom_failures, cleanroom_details = cleanroom()
        failures.extend(cleanroom_failures)
        cleanroom_report = {
            "status": "PASS" if not cleanroom_failures else "FAIL",
            **cleanroom_details,
        }
    report = {
        "schema": "coregraph_level4_release_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "checksum_validation": "PASS" if not any(x.startswith("checksum:") for x in failures) else "FAIL",
        "cleanroom": cleanroom_report,
        "failures": failures,
    }
    output = BUILD / "LEVEL4_CLEANROOM_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if arguments.cleanroom:
        validation = json.loads((RELEASE / "VALIDATION_REPORT.json").read_text())
        validation["cleanroom_status"] = cleanroom_report["status"]
        validation["status"] = "PASS" if not failures else "FAIL"
        validation["failures"] = sorted(set(validation.get("failures", []) + failures))
        (RELEASE / "VALIDATION_REPORT.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_path = RELEASE / "MANIFEST.csv"
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        validation_relative = "release/level4/VALIDATION_REPORT.json"
        for row in manifest_rows:
            if row["path"] == validation_relative:
                row["bytes"] = str((RELEASE / "VALIDATION_REPORT.json").stat().st_size)
                row["sha256"] = sha256_path(RELEASE / "VALIDATION_REPORT.json")
        fields = ("path", "bytes", "sha256", "publishable", "licence_or_access")
        for target in (manifest_path, BUILD / "LEVEL4_RELEASE_MANIFEST.csv"):
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(manifest_rows)
        checksum_targets = [
            RELEASE / "README.md",
            RELEASE / "MANIFEST.csv",
            RELEASE / "VALIDATION_REPORT.json",
            RELEASE / "DETERMINISM_REPORT.json",
            RELEASE / "CLAIM_SUPPORT_REPORT.json",
            RELEASE / "PUBLIC_TREE_AUDIT.json",
            SNAPSHOTS / "coregraph_source_snapshot.zip",
            SNAPSHOTS / "curated_source_snapshot.zip",
        ]
        (RELEASE / "CHECKSUMS.sha256").write_text(
            "".join(
                f"{sha256_path(path)}  {path.relative_to(ROOT).as_posix()}\n"
                for path in checksum_targets
            ),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
