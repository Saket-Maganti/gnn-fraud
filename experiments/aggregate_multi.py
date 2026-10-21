"""
experiments/aggregate_multi.py

Collapse the per-(dataset, model, seed) JSON files produced by
``run_multi_dataset.py`` and ``run_shuffle_ablation_multi.py`` into the
summary tables the paper needs:

    Table A : mean ± std F1 per (dataset, model) under transductive,
              inductive, and TPC+TTA.  Used to make the reviewer's central
              point visible at a glance:
                  transductive is wildly optimistic,
                  inductive is weak,
                  TPC+TTA recovers most of the gap.
    Table B : shuffled-edges results per (dataset, model) — real vs.
              shuffled vs. none.  Anywhere ``shuffled.f1 >= real.f1`` is a
              data point for the paper's secondary claim.

All tables are also dumped to CSV so the LaTeX tables in the paper can be
regenerated verbatim from the JSON artefacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.result_audit import (  # noqa: E402
    validate_multi_result,
    validate_shuffle_result,
)


def _iter_results(path: str, validator=None, strict_schema: bool = False):
    for fn in sorted(glob.glob(os.path.join(path, "*.json"))):
        if os.path.basename(fn).startswith("_"):
            continue
        with open(fn) as f:
            payload = json.load(f)
        if "error" in payload:
            yield payload
            continue
        if validator is not None:
            problems = validator(payload)
            if problems:
                msg = f"{fn}: " + "; ".join(problems[:8])
                if strict_schema:
                    raise ValueError(msg)
                print(f"[warn] skipping invalid result: {msg}")
                continue
        yield payload


def _agg(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (statistics.mean(xs), statistics.pstdev(xs))


def _fmt(mean: float, std: float) -> str:
    if mean != mean:      # NaN
        return "   --   "
    return f"{mean:.3f} ± {std:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Table A — transductive vs. inductive vs. TPC+TTA
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_multi(in_dir: str, strict_schema: bool = False) -> List[Dict]:
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"trans": [], "ind": [], "tpc": []}
    )
    for r in _iter_results(in_dir, validate_multi_result, strict_schema):
        if "error" in r:
            continue
        key = (r["dataset"], r["model"])
        buckets[key]["trans"].append(r["transductive"]["f1"])
        buckets[key]["ind"].append(r["inductive"]["f1"])
        tpc = r.get("tpc_tta", {}).get("tpc_tta", {})
        if "f1" in tpc:
            buckets[key]["tpc"].append(tpc["f1"])

    out = []
    for (ds, mdl), scores in sorted(buckets.items()):
        row = {
            "dataset":    ds,
            "model":      mdl,
            "n_seeds":    len(scores["trans"]),
            "trans_mean": _agg(scores["trans"])[0],
            "trans_std":  _agg(scores["trans"])[1],
            "ind_mean":   _agg(scores["ind"])[0],
            "ind_std":    _agg(scores["ind"])[1],
            "tpc_mean":   _agg(scores["tpc"])[0],
            "tpc_std":    _agg(scores["tpc"])[1],
            "leak_gap":   _agg(scores["trans"])[0] - _agg(scores["ind"])[0],
            "tpc_lift":   _agg(scores["tpc"])[0]   - _agg(scores["ind"])[0],
        }
        out.append(row)
    return out


def print_table_a(rows: List[Dict]) -> None:
    print("\n" + "=" * 92)
    print(" TABLE A — Evaluation protocol reverses rankings; TPC+TTA recovers the gap")
    print("=" * 92)
    header = (f"{'dataset':<12} {'model':<20} {'#seeds':>6} "
              f"{'transductive':>14} {'inductive':>14} {'TPC+TTA':>14} "
              f"{'leak':>7} {'lift':>7}")
    print(header)
    print("-" * 92)
    for r in rows:
        print(f"{r['dataset']:<12} {r['model']:<20} {r['n_seeds']:>6d} "
              f"{_fmt(r['trans_mean'], r['trans_std']):>14} "
              f"{_fmt(r['ind_mean'],   r['ind_std']):>14} "
              f"{_fmt(r['tpc_mean'],   r['tpc_std']):>14} "
              f"{r['leak_gap']:>+7.3f} {r['tpc_lift']:>+7.3f}")
    print("=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# Table B — shuffled-edges ablation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_shuffle(in_dir: str, strict_schema: bool = False) -> List[Dict]:
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"real": [], "shuffled": [], "none": []}
    )
    for r in _iter_results(in_dir, validate_shuffle_result, strict_schema):
        if "error" in r:
            continue
        key = (r["dataset"], r["model"])
        for v in ("real", "shuffled", "none"):
            if v in r and "f1" in r[v]:
                buckets[key][v].append(r[v]["f1"])

    out = []
    for (ds, mdl), scores in sorted(buckets.items()):
        real_m, real_s   = _agg(scores["real"])
        shuf_m, shuf_s   = _agg(scores["shuffled"])
        none_m, none_s   = _agg(scores["none"])
        out.append({
            "dataset":     ds,
            "model":       mdl,
            "n_seeds":     len(scores["real"]),
            "real_mean":   real_m, "real_std":   real_s,
            "shuf_mean":   shuf_m, "shuf_std":   shuf_s,
            "none_mean":   none_m, "none_std":   none_s,
            "graph_value": real_m - max(shuf_m, none_m),
        })
    return out


def print_table_b(rows: List[Dict]) -> None:
    print("\n" + "=" * 92)
    print(" TABLE B — Shuffled-edges ablation (inductive; negative 'graph_value' = "
          "real graph hurts)")
    print("=" * 92)
    header = (f"{'dataset':<12} {'model':<20} {'#seeds':>6} "
              f"{'real':>14} {'shuffled':>14} {'no-edges':>14} "
              f"{'graph_value':>12}")
    print(header)
    print("-" * 92)
    for r in rows:
        print(f"{r['dataset']:<12} {r['model']:<20} {r['n_seeds']:>6d} "
              f"{_fmt(r['real_mean'], r['real_std']):>14} "
              f"{_fmt(r['shuf_mean'], r['shuf_std']):>14} "
              f"{_fmt(r['none_mean'], r['none_std']):>14} "
              f"{r['graph_value']:>+12.3f}")
    print("=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / main
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[io] wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--multi",   default="results/multi")
    p.add_argument("--shuffle", default="results/shuffle_multi")
    p.add_argument("--out",     default="results/aggregated")
    p.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail instead of skipping malformed result JSON files.",
    )
    args = p.parse_args()

    if os.path.isdir(args.multi):
        rows_a = aggregate_multi(args.multi, strict_schema=args.strict_schema)
        print_table_a(rows_a)
        write_csv(rows_a, os.path.join(args.out, "table_a_protocol_gap.csv"))
    else:
        print(f"[skip] {args.multi} not found — run run_multi_dataset.py first")

    if os.path.isdir(args.shuffle):
        rows_b = aggregate_shuffle(args.shuffle, strict_schema=args.strict_schema)
        print_table_b(rows_b)
        write_csv(rows_b, os.path.join(args.out, "table_b_shuffle.csv"))
    else:
        print(f"[skip] {args.shuffle} not found — run run_shuffle_ablation_multi.py first")


if __name__ == "__main__":
    main()
