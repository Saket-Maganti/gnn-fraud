#!/usr/bin/env python3
"""Fail-closed public-tree validator for the curated GitHub checkout.

The report contains only categories, paths, line numbers, and redacted
fingerprints. Matching secret values are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


MIB = 1024 * 1024
MAX_TEXT_SCAN = 4 * MIB

FORBIDDEN_PARTS = {
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    "__MACOSX",
    "__pycache__",
    "gnn_env",
    "node_modules",
}

FORBIDDEN_NAMES = {
    ".DS_Store",
    ".env",
    ".netrc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "kaggle.json",
}

BINARY_SUFFIXES = {
    ".7z",
    ".avi",
    ".bin",
    ".bz2",
    ".dylib",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".npz",
    ".npy",
    ".parquet",
    ".pkl",
    ".pt",
    ".pth",
    ".so",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}

TEXT_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "private_absolute_path": re.compile(
        r"(?:/" + r"Users/[^/\s]+|/" + r"home/[^/\s]+|/" + r"Volumes/[^/\s]+|"
        + r"[A-Za-z]:\\\\" + r"Users\\\\[^\\\\\s]+)"
    ),
    "home_alias": re.compile(r"(?<![A-Za-z0-9_])" + r"~" + r"/(?:[^\s`\"']+)"),
    "email_address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    path: str
    line: int | None
    fingerprint: str
    note: str


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES or path.suffix.lower() == ".pdf":
        return False
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return False
    return b"\x00" not in chunk


def raw_prediction_path(rel: str) -> bool:
    lower = rel.lower()
    if "prediction" not in lower:
        return False
    payload_suffixes = {
        ".arrow",
        ".csv",
        ".feather",
        ".json",
        ".jsonl",
        ".npy",
        ".npz",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".tsv",
        ".txt",
    }
    if Path(lower).suffix not in payload_suffixes:
        return False
    safe_markers = (
        "manifest",
        "index",
        "contract",
        "schema",
        "audit",
        "readme",
        "summary",
        "provenance",
    )
    return not any(marker in lower for marker in safe_markers)


def scan(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        rel_current = current_path.relative_to(root)
        if rel_current.parts and rel_current.parts[0] == ".git":
            dirnames[:] = []
            continue
        dirnames[:] = sorted(dirnames)
        for dirname in list(dirnames):
            candidate = current_path / dirname
            rel = candidate.relative_to(root).as_posix()
            if candidate.is_symlink():
                findings.append(Finding("error", "symlink", rel, None, "", "Symlinks are forbidden."))
            if dirname in FORBIDDEN_PARTS or "backup_" in dirname.lower():
                findings.append(
                    Finding("error", "archive_hygiene", rel, None, "", "Cache, environment, or backup directory.")
                )
        for filename in sorted(filenames):
            path = current_path / filename
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                findings.append(Finding("error", "symlink", rel, None, "", "Symlinks are forbidden."))
                continue
            try:
                size = path.stat().st_size
            except OSError:
                findings.append(Finding("error", "unreadable", rel, None, "", "File could not be statted."))
                continue
            if filename in FORBIDDEN_NAMES or filename.endswith((".pem", ".key")):
                findings.append(
                    Finding("error", "credential_or_metadata_filename", rel, None, "", "Forbidden filename.")
                )
            if size > 100 * MIB:
                findings.append(
                    Finding("error", "oversized_git_file", rel, None, "", f"{size / MIB:.2f} MiB exceeds 100 MiB.")
                )
            if rel.lower().startswith("data/raw/") and filename not in {".gitkeep", "README.md"}:
                findings.append(Finding("error", "raw_data", rel, None, "", "Raw dataset payload."))
            if raw_prediction_path(rel):
                findings.append(Finding("error", "raw_prediction_path", rel, None, "", "Prediction-like payload path."))
            if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
                findings.append(Finding("error", "duplicate_archive", rel, None, "", "Archives are excluded from Git."))
            if size > MAX_TEXT_SCAN or not is_probably_text(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for category, pattern in TEXT_PATTERNS.items():
                    for match in pattern.finditer(line):
                        value = match.group(0)
                        severity = "error"
                        note = "Sensitive value pattern."
                        if category == "email_address":
                            note = "Email address requires citation/anonymity review."
                        findings.append(
                            Finding(severity, category, rel, line_no, fingerprint(value), note)
                        )
    if not any(root.glob("LICENSE*")):
        findings.append(
            Finding("error", "license_gate", ".", None, "", "No LICENSE or LICENSE_REVIEW_REQUIRED file.")
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan(root)
    payload = {
        "root_label": root.name,
        "finding_count": len(findings),
        "ok": not findings,
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
