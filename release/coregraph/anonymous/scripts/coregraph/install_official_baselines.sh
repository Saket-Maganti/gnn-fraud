#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--dry-run" ]]; then
  echo "Refusing network installation without --dry-run. Follow the reviewed registry commands manually." >&2
  exit 2
fi

python3 - <<'PY'
from pathlib import Path
import yaml

registry = yaml.safe_load(Path("external_baselines/BASELINE_REGISTRY.yaml").read_text())
for name, record in registry["baselines"].items():
    print(f"[{name}] {record['status']} {record['commit']}")
    for command in record.get("acquisition", []):
        print(f"  {command}")
PY
