#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
paper_dir="$repo_root/paper_iclr"
report="$repo_root/results/coregraph_build/PAPER_BUILD_STATUS.json"

if ! command -v pdflatex >/dev/null 2>&1; then
  printf '{"schema":"coregraph_paper_build_v1","status":"BLOCKED_LATEX_UNAVAILABLE"}\n' > "$report"
  echo "pdflatex unavailable" >&2
  exit 2
fi

(
  cd "$paper_dir"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  if command -v bibtex >/dev/null 2>&1; then
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
  fi
)

test -s "$paper_dir/main.pdf"
bytes="$(wc -c < "$paper_dir/main.pdf" | tr -d ' ')"
printf '{"schema":"coregraph_paper_build_v1","status":"PASS_PLACEHOLDER_PDF","bytes":%s}\n' "$bytes" > "$report"
echo "Built paper_iclr/main.pdf ($bytes bytes; empirical placeholders retained)"
