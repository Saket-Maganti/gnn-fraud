#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

compile_document() {
  local directory="$1"
  local stem="$2"
  (
    cd "$directory"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex"
    bibtex "$stem"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex"
    pdflatex -interaction=nonstopmode -halt-on-error "$stem.tex"
  )
}

compile_document "$ROOT/paper_tkde" main
compile_document "$ROOT/paper_tkde/supplement" supplement

if rg -n "Citation .* undefined|Reference .* undefined|There were undefined references|Rerun to get cross-references right" \
  "$ROOT/paper_tkde/main.log" "$ROOT/paper_tkde/supplement/supplement.log"; then
  echo "LaTeX audit failed: unresolved citation or cross-reference warning." >&2
  exit 1
fi

echo "Main and supplement compiled with clean citation/cross-reference logs."
