#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = [
    "all gnns fail",
    "graph structure is universally harmful",
    "we prove graph structure hurts",
    "large results passed",
    "public hosted leaderboard exists",
]
SAFE_NEGATION_MARKERS = [
    "do not",
    "don't",
    "not claim",
    "never claim",
    "forbidden",
    "unsafe",
    "invalid",
    "blocked",
    "forbidden",
    "rejected",
    "not all",
    "overclaim",
    "prohibited",
    "counterexample",
    "what we do not claim",
    "claim_gate",
]

def _actionable_occurrences(text: str, phrase: str) -> list[int]:
    lines = text.splitlines()
    hits = []
    for number, line in enumerate(lines, start=1):
        if phrase not in line:
            continue
        window = " ".join(lines[max(0, number - 8) : min(len(lines), number + 3)])
        if any(marker in window for marker in SAFE_NEGATION_MARKERS):
            continue
        hits.append(number)
    return hits

def main() -> int:
    paths = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts and "promptpacks" not in p.parts]
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in BANNED:
            for line in _actionable_occurrences(text, phrase):
                findings.append({"path": str(path.relative_to(ROOT)), "phrase": phrase, "line": line})
    out = ROOT / "results/benchmark_synthesis/v28_to_v39_claim_language_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "PASS" if not findings else "FAIL", "findings": findings}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0 if not findings else 1

if __name__ == "__main__":
    raise SystemExit(main())
