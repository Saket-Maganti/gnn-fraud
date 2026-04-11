"""
experiments/compare_leakage_gap.py

Print a side-by-side comparison of the transductive and inductive GraphSAGE
runs produced by `run_transductive.py` and `run_inductive.py`.

Usage:
    python experiments/run_transductive.py
    python experiments/run_inductive.py
    python experiments/compare_leakage_gap.py
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments._leakage_gap_utils import load_results


def _fmt(val) -> str:
    return f"{val:.3f}" if isinstance(val, (int, float)) else "   — "


def _row(model: str, setting: str, metrics: dict) -> str:
    f1 = metrics.get("f1", float("nan")) if metrics else float("nan")
    return f"{model:<11s} {setting:<13s} {_fmt(f1)}"


def print_table(transductive: dict, inductive: dict) -> None:
    print()
    print("Transductive vs Inductive Results")
    print()
    print(f"{'Model':<11s} {'Setting':<13s} F1")
    print("-" * 32)
    if transductive:
        print(_row("GraphSAGE", "Transductive",
                   transductive.get("best_metrics", {})))
    else:
        print(f"{'GraphSAGE':<11s} {'Transductive':<13s}  (missing)")
    if inductive:
        print(_row("GraphSAGE", "Inductive",
                   inductive.get("best_metrics", {})))
    else:
        print(f"{'GraphSAGE':<11s} {'Inductive':<13s}  (missing)")
    print()

    if transductive and inductive:
        t_f1 = transductive["best_metrics"].get("f1", float("nan"))
        i_f1 = inductive["best_metrics"].get("f1", float("nan"))
        gap  = t_f1 - i_f1
        print(f"Leakage gap (transductive − inductive F1): {gap:+.3f}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trans", default="results/transductive_results.json")
    p.add_argument("--ind",   default="results/inductive_results.json")
    args = p.parse_args()

    transductive = load_results(args.trans)
    inductive    = load_results(args.ind)

    if transductive is None and inductive is None:
        print("No result files found. Run run_transductive.py and "
              "run_inductive.py first.")
        return

    print_table(transductive, inductive)


if __name__ == "__main__":
    main()
