#!/usr/bin/env python3
"""Validate provider manifests and optionally verify staged file checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("manifests", nargs="*")
    args = parser.parse_args()
    paths = [Path(value) for value in args.manifests]
    if args.all:
        paths = sorted((ROOT / "data/manifests/coregraph").glob("*.yaml"))
    failures: list[str] = []
    checked = 0
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {"schema", "dataset_id", "provider", "licence", "task", "raw_files"}
        missing = required - set(payload or {})
        if missing:
            failures.append(f"{path}:missing:{sorted(missing)}")
            continue
        if payload["schema"] != "coregraph_dataset_manifest_v1":
            failures.append(f"{path}:schema")
        for record in payload["raw_files"]:
            raw = ROOT / record["path"]
            if not raw.exists():
                failures.append(f"{path}:raw_missing:{record['path']}")
                continue
            actual = hashlib.sha256(raw.read_bytes()).hexdigest()
            if actual != record["sha256"]:
                failures.append(f"{path}:checksum:{record['path']}")
            if raw.stat().st_size != int(record["bytes"]):
                failures.append(f"{path}:bytes:{record['path']}")
        checked += 1
    if args.all and not paths:
        failures.append("no_provider_manifests_staged")
    report = {
        "schema": "coregraph_dataset_manifest_audit_v1",
        "checked": checked,
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
    }
    output = ROOT / "results/coregraph_build/DATASET_MANIFEST_AUDIT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
