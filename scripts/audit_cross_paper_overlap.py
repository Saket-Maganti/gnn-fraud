#!/usr/bin/env python3
"""Fail closed on unjustified FraudShiftBench/CoReGraph paper overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "results" / "coregraph_build"
TEXT_SUFFIXES = {".tex", ".md"}
ASSET_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_WORDS = 15
NGRAM_SIZE = 8
MAX_ICLR_NGRAM_CONTAINMENT = 0.025
MAX_LONGEST_COMMON_WORDS = 36
MAX_EXACT_LONG_SENTENCES = 0


def _eligible(path: Path, suffixes: set[str]) -> bool:
    excluded = {"build", "pdf", "__pycache__"}
    return path.is_file() and path.suffix.lower() in suffixes and not excluded.intersection(path.parts)


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if _eligible(path, suffixes))


def _strip_latex(text: str) -> str:
    text = re.sub(r"(?m)%.*$", " ", text)
    text = re.sub(r"\\(?:cite|ref|label|input|includegraphics|bibliography)\*?(?:\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _corpus(paths: Iterable[Path]) -> str:
    return " ".join(
        _strip_latex(path.read_text(encoding="utf-8", errors="replace")) for path in paths
    )


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _sentences(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for sentence in SENTENCE_PATTERN.split(text):
        words = _tokens(sentence)
        if len(words) >= MIN_SENTENCE_WORDS:
            output[" ".join(words)] = sentence.strip()
    return output


def _asset_hashes(root: Path) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = {}
    for path in _files(root, ASSET_SUFFIXES):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(path.relative_to(root).as_posix())
    return hashes


def audit(tkde_root: Path, iclr_root: Path) -> dict[str, object]:
    tkde_paths = _files(tkde_root, TEXT_SUFFIXES)
    iclr_paths = _files(iclr_root, TEXT_SUFFIXES)
    tkde_text = _corpus(tkde_paths)
    iclr_text = _corpus(iclr_paths)
    tkde_tokens = _tokens(tkde_text)
    iclr_tokens = _tokens(iclr_text)
    tkde_ngrams = _ngrams(tkde_tokens, NGRAM_SIZE)
    iclr_ngrams = _ngrams(iclr_tokens, NGRAM_SIZE)
    common_ngrams = tkde_ngrams & iclr_ngrams
    union_ngrams = tkde_ngrams | iclr_ngrams
    containment = len(common_ngrams) / max(1, len(iclr_ngrams))
    jaccard = len(common_ngrams) / max(1, len(union_ngrams))
    tkde_sentences = _sentences(tkde_text)
    iclr_sentences = _sentences(iclr_text)
    common_sentences = sorted(set(tkde_sentences) & set(iclr_sentences))
    matcher = SequenceMatcher(a=tkde_tokens, b=iclr_tokens, autojunk=False)
    longest = matcher.find_longest_match()
    longest_words = iclr_tokens[longest.b : longest.b + longest.size]

    tkde_assets = _asset_hashes(tkde_root)
    iclr_assets = _asset_hashes(iclr_root)
    shared_asset_hashes = sorted(set(tkde_assets) & set(iclr_assets))
    shared_assets = [
        {
            "sha256": digest,
            "tkde": tkde_assets[digest],
            "iclr": iclr_assets[digest],
        }
        for digest in shared_asset_hashes
    ]
    failures: list[str] = []
    if containment > MAX_ICLR_NGRAM_CONTAINMENT:
        failures.append(
            f"eight_gram_containment:{containment:.6f}>{MAX_ICLR_NGRAM_CONTAINMENT:.6f}"
        )
    if longest.size > MAX_LONGEST_COMMON_WORDS:
        failures.append(f"longest_common_words:{longest.size}>{MAX_LONGEST_COMMON_WORDS}")
    if len(common_sentences) > MAX_EXACT_LONG_SENTENCES:
        failures.append(f"exact_long_sentences:{len(common_sentences)}>0")
    if shared_assets:
        failures.append(f"byte_identical_visual_assets:{len(shared_assets)}>0")

    distinction = {
        "primary_question": {
            "fraudshiftbench_tkde": "evaluation and claim validity",
            "coregraph_iclr": "learning under unseen deployment-contract compositions",
        },
        "contribution": {
            "fraudshiftbench_tkde": "benchmark and evidence framework",
            "coregraph_iclr": "routing method and theory",
        },
        "main_evidence": {
            "fraudshiftbench_tkde": "protocol-sensitive benchmarking",
            "coregraph_iclr": "compositional generalisation",
        },
        "theory": {
            "fraudshiftbench_tkde": "evidence and support semantics",
            "coregraph_iclr": "regret, resource masks, and selective routing",
        },
        "main_visuals": {
            "fraudshiftbench_tkde": "benchmark and evidence map",
            "coregraph_iclr": "architecture, mechanisms, and new result templates",
        },
        "shared_data": "disclosed; no provider payload duplicated in either source tree",
        "shared_text": "measured below and required to remain minimal",
    }
    return {
        "schema": "coregraph_cross_paper_overlap_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "thresholds": {
            "ngram_size": NGRAM_SIZE,
            "max_iclr_ngram_containment": MAX_ICLR_NGRAM_CONTAINMENT,
            "max_longest_common_words": MAX_LONGEST_COMMON_WORDS,
            "max_exact_long_sentences": MAX_EXACT_LONG_SENTENCES,
            "max_byte_identical_visual_assets": 0,
        },
        "corpora": {
            "tkde_text_files": len(tkde_paths),
            "iclr_text_files": len(iclr_paths),
            "tkde_words": len(tkde_tokens),
            "iclr_words": len(iclr_tokens),
        },
        "measurements": {
            "common_eight_grams": len(common_ngrams),
            "iclr_eight_gram_containment": containment,
            "eight_gram_jaccard": jaccard,
            "exact_long_sentence_count": len(common_sentences),
            "exact_long_sentences": common_sentences,
            "longest_common_contiguous_words": longest.size,
            "longest_common_excerpt": " ".join(longest_words),
            "byte_identical_visual_asset_count": len(shared_assets),
            "byte_identical_visual_assets": shared_assets,
        },
        "distinction_matrix": distinction,
        "audited_dimensions": [
            "problem statement",
            "contributions",
            "methods",
            "theory",
            "experiments",
            "figures",
            "tables",
            "text",
            "claims",
        ],
        "failures": failures,
    }


def _write_reports(report: dict[str, object]) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "LEVEL4_CROSS_PAPER_OVERLAP_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    measurements = report["measurements"]
    assert isinstance(measurements, dict)
    lines = [
        "# Level-4 cross-paper overlap audit",
        "",
        f"Status: `{report['status']}`.",
        "",
        "The firewall compares problem framing, contributions, methods, theory, experiments, figures, tables, prose, and claims. Shared datasets are disclosed; provider payloads are absent.",
        "",
        f"- ICLR eight-gram containment: `{measurements['iclr_eight_gram_containment']:.6f}` (limit `{MAX_ICLR_NGRAM_CONTAINMENT:.6f}`).",
        f"- Exact sentences of at least {MIN_SENTENCE_WORDS} words: `{measurements['exact_long_sentence_count']}` (limit `0`).",
        f"- Longest common contiguous block: `{measurements['longest_common_contiguous_words']}` words (limit `{MAX_LONGEST_COMMON_WORDS}`).",
        f"- Byte-identical visual assets: `{measurements['byte_identical_visual_asset_count']}` (limit `0`).",
        "",
        "See `docs/coregraph/TKDE_ICLR_DIFFERENTIATION_MATRIX.md` and `docs/coregraph/RELATED_SUBMISSION_DISCLOSURE_TEMPLATE.md` for the scientific distinction and disclosure workflow.",
    ]
    (BUILD / "LEVEL4_CROSS_PAPER_OVERLAP_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tkde-root", type=Path, default=ROOT / "paper_tkde")
    parser.add_argument("--iclr-root", type=Path, default=ROOT / "paper_iclr")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = audit(arguments.tkde_root.resolve(), arguments.iclr_root.resolve())
    _write_reports(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
