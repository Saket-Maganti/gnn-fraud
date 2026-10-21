#!/usr/bin/env python3
"""Fail closed on identity, data, result, path, or checksum leaks."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "release/coregraph/anonymous"
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "results",
    "predictions",
    "checkpoints",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".tex",
    ".bib",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
    ".csv",
    ".sh",
}
PATTERNS = (
    re.compile(r"/" + r"Users/[A-Za-z0-9._-]+"),
    re.compile(r"\bSaket\s+Maganti\b", re.IGNORECASE),
    re.compile(r"saket" r"maganti", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.org)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)
MAX_FILE_BYTES = 10 * 1024 * 1024


def main() -> int:
    failures: list[str] = []
    manifest_path = PACKAGE / "ANONYMOUS_RELEASE_MANIFEST.json"
    if not manifest_path.exists():
        failures.append("manifest_missing")
        manifest = {"files": {}}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in sorted(PACKAGE.rglob("*")):
        relative = path.relative_to(PACKAGE)
        if path.is_symlink():
            failures.append(f"symlink:{relative}")
            continue
        if relative.parts and relative.parts[0].lower() == "data":
            failures.append(f"forbidden_path:{relative}")
        if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
            failures.append(f"forbidden_path:{relative}")
        if not path.is_file() or path == manifest_path:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"oversized:{relative}")
        expected = manifest.get("files", {}).get(str(relative), {}).get("sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            failures.append(f"checksum:{relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in PATTERNS:
                if pattern.search(text):
                    failures.append(f"identity_or_path:{relative}:{pattern.pattern}")
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            "import coregraph; import coregraph.experts.registry; "
            "print(coregraph.__version__)",
        ],
        cwd=PACKAGE,
        check=False,
        capture_output=True,
        text=True,
    )
    if smoke.returncode != 0:
        failures.append(f"package_import:{smoke.stderr[-500:]}")
    test_environment = dict(os.environ)
    test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    package_tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/coregraph",
        ],
        cwd=PACKAGE,
        check=False,
        capture_output=True,
        text=True,
        env=test_environment,
    )
    if package_tests.returncode != 0:
        failures.append(f"package_tests:{package_tests.stdout[-500:]}")
    report = {
        "schema": "coregraph_anonymous_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "package": "release/coregraph/anonymous",
        "package_tests": (
            "PASS"
            if package_tests.returncode == 0
            else f"FAIL_EXIT_{package_tests.returncode}"
        ),
    }
    output = ROOT / "results/coregraph_build/ANONYMOUS_RELEASE_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
