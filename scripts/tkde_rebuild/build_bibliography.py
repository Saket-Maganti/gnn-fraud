#!/usr/bin/env python3
"""Build the paper bibliography only from the verified literature matrix."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "results" / "tkde_rebuild" / "LITERATURE_MATRIX.csv"
OUTPUT = ROOT / "paper_tkde" / "references.bib"

ARTICLE_VENUES = {
    "Decision Support Systems",
    "Journal of Machine Learning Research",
    "ACM Transactions on Knowledge Discovery from Data",
    "Patterns",
    "Transactions of the Association for Computational Linguistics",
    "Communications of the ACM",
    "PLOS ONE",
    "Neural Computation",
    "Expert Systems with Applications",
    "The Annals of Statistics",
    "Scandinavian Journal of Statistics",
    "Journal of the Royal Statistical Society, Series B",
}

ACCENTS = {
    "Béni": r"B{\'e}ni",
    "Blanuša": r"Blanu{\v{s}}a",
    "Niederhäusern": r"Niederh{\"a}usern",
    "Veličković": r"Veli{\v{c}}kovi{\'c}",
    "Liò": r"Li{\`o}",
    "Daumé": r"Daum{\'e}",
    "Björn": r"Bj{\"o}rn",
}


def latex_text(text: str) -> str:
    text = text.replace("–", "--").replace("—", "---")
    for source, target in ACCENTS.items():
        text = text.replace(source, target)
    return text


def authors(text: str) -> str:
    return " and ".join(latex_text(part.strip()) for part in text.split(";") if part.strip())


def entry(row: pd.Series) -> str:
    venue = str(row.venue)
    status = str(row.publication_status)
    if venue in ARTICLE_VENUES:
        kind = "article"
        venue_field = f"  journal = {{{latex_text(venue)}}},"
    elif status in {"preprint", "workshop/preprint"} or venue in {"arXiv", "arXiv/OpenReview"}:
        kind = "misc"
        venue_field = f"  howpublished = {{{latex_text(venue)}}},"
    else:
        kind = "inproceedings"
        venue_field = f"  booktitle = {{{latex_text(venue)}}},"
    lines = [
        f"@{kind}{{{row.cite_key},",
        f"  author = {{{authors(str(row.authors))}}},",
        f"  title = {{{{{latex_text(str(row.title))}}}}},",
        venue_field,
        f"  year = {{{int(row.year)}}},",
    ]
    identifier = str(row.doi_or_arxiv)
    if re.fullmatch(r"10\.\S+", identifier):
        lines.append(f"  doi = {{{identifier}}},")
    elif identifier.startswith("arXiv:"):
        arxiv = identifier.split(":", 1)[1]
        lines.append(f"  note = {{arXiv:{arxiv}}},")
    else:
        # Retain an official locator when no DOI or arXiv identifier exists.
        # DOI-backed entries already have a durable identifier; repeating long
        # proceedings URLs adds no provenance and can create unbreakable boxes.
        lines.append(f"  url = {{{row.official_url}}}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    frame = pd.read_csv(MATRIX)
    if not frame.cite_key.is_unique:
        raise SystemExit("duplicate verified citation keys")
    if not frame.verification_status.eq("VERIFIED_PRIMARY_OR_OFFICIAL").all():
        raise SystemExit("unverified row in literature matrix")
    text = "% Generated from results/tkde_rebuild/LITERATURE_MATRIX.csv.\n"
    text += "% Do not add an entry without updating the verification matrix.\n\n"
    text += "\n\n".join(entry(row) for _, row in frame.sort_values("cite_key").iterrows()) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {len(frame)} verified BibTeX entries")


if __name__ == "__main__":
    main()
