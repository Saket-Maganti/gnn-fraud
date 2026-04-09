#!/usr/bin/env python3
"""
Synchronize paper figures into the repository-level figures/ directory.

The script scans every LaTeX file in paper/, extracts \\includegraphics
references, resolves them against known result directories, copies only the
required assets into figures/, rewrites the LaTeX references to
figures/<snake_case_name>, and emits a manifest for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


RE_INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
RE_USEPACKAGE_GRAPHICX = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{graphicx\}")
RE_SNAKE = re.compile(r"[^a-z0-9]+")


def snake_case(name: str) -> str:
    stem = RE_SNAKE.sub("_", Path(name).stem.lower()).strip("_")
    ext = Path(name).suffix.lower()
    return f"{stem}{ext}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_paths(source_roots: list[Path], reference: str) -> list[Path]:
    ref_path = Path(reference)
    ref_name = ref_path.name
    candidates: list[Path] = []

    for root in source_roots:
        direct = (root / reference).resolve()
        if direct.exists() and direct.is_file():
            candidates.append(direct)

        by_name = sorted(root.rglob(ref_name))
        candidates.extend(path.resolve() for path in by_name if path.is_file())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def choose_candidate(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No matching asset found.")

    preferred = sorted(
        paths,
        key=lambda path: (
            "figures" not in path.parts,
            "results2" not in path.parts,
            len(path.parts),
            str(path),
        ),
    )
    return preferred[0]


def ensure_graphics_preamble(tex_path: Path) -> None:
    text = tex_path.read_text()

    if not RE_USEPACKAGE_GRAPHICX.search(text):
        begin_doc = text.find("\\begin{document}")
        if begin_doc != -1:
            text = text[:begin_doc] + "\\usepackage{graphicx}\n" + text[begin_doc:]

    lines = text.splitlines()
    graphicspath_line = r"\graphicspath{{./}{../}}"
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(r"\graphicspath"):
            lines[index] = graphicspath_line
            replaced = True
            break

    if not replaced:
        begin_doc = next((i for i, line in enumerate(lines) if r"\begin{document}" in line), None)
        if begin_doc is not None:
            lines.insert(begin_doc, "")
            lines.insert(begin_doc, graphicspath_line)

    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"

    tex_path.write_text(text)


def sync_figures(repo_root: Path, rewrite_tex: bool = True, clean: bool = True) -> dict:
    paper_dir = repo_root / "paper"
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    tex_files = sorted(paper_dir.glob("*.tex"))
    source_roots = [
        repo_root / "figures",
        repo_root / "results",
        repo_root / "results2",
        repo_root / "gnnpaper" / "figures",
        repo_root / "gnnpaper" / "results2",
        repo_root / "paper",
    ]

    manifest: dict[str, dict[str, str]] = {}
    referenced_targets: set[Path] = set()

    for tex_path in tex_files:
        ensure_graphics_preamble(tex_path)
        original = tex_path.read_text()

        def replacer(match: re.Match[str]) -> str:
            reference = match.group(1).strip()
            candidates = candidate_paths(source_roots, reference)
            chosen = choose_candidate(candidates)
            normalized = snake_case(chosen.name)
            target = figures_dir / normalized
            referenced_targets.add(target)

            if target.exists():
                if sha256(target) != sha256(chosen):
                    raise RuntimeError(
                        f"Conflicting figure content for {normalized}: "
                        f"{target} vs {chosen}"
                    )
            else:
                shutil.copy2(chosen, target)

            manifest[reference] = {
                "source": str(chosen.relative_to(repo_root)),
                "target": str(target.relative_to(repo_root)),
                "sha256": sha256(target),
            }

            return match.group(0).replace(reference, f"figures/{normalized}")

        rewritten = RE_INCLUDE.sub(replacer, original)
        if rewrite_tex and rewritten != original:
            tex_path.write_text(rewritten)

    if clean:
        for existing in figures_dir.iterdir():
            if existing.is_file() and existing not in referenced_targets:
                existing.unlink()

    manifest_path = paper_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root containing paper/ and figures/.",
    )
    parser.add_argument(
        "--no-rewrite-tex",
        action="store_true",
        help="Copy figures without rewriting LaTeX references.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep extra files that already exist in figures/.",
    )
    args = parser.parse_args()

    manifest = sync_figures(
        repo_root=args.repo_root.resolve(),
        rewrite_tex=not args.no_rewrite_tex,
        clean=not args.no_clean,
    )
    print(f"Synchronized {len(manifest)} figure references.")


if __name__ == "__main__":
    main()
