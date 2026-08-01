#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
python_bin="${PYTHON:-python3}"
cd "$repo_root"
exec "$python_bin" scripts/coregraph/build_level4_paper.py
