"""
experiments/run_phase_sensitivity.py

Generator-assumption sensitivity analysis for the protocol rank-reversal theory.

The phase-diagram validation (``run_protocol_phase_diagram.py``) confirms the
theory on *one* synthetic generator setting. A reviewer will reasonably ask
whether that result is cherry-picked: does the reversal — and the theory<->sim
agreement — survive when the generator's structural assumptions change?

This script re-runs the phase sweep across a grid of generator knobs:

  * ``k_neighbours``   — neighbourhood size the structural channel aggregates over;
  * ``homophily_train``— train-era P(neighbour shares label) (structure strength);
  * ``struct_sep``     — latent structural signal magnitude.

For each knob value it reports the theory<->sim reversal agreement (full grid and
held-out interior-h), the bootstrap CI, and the fraction of cells that reverse.
A robust result is: reversal happens (frac > 0) and agreement stays high across
*every* setting — i.e. the phenomenon is a property of protocol-induced structure
decay, not of one hand-tuned generator.

CPU-only, seeded, no dataset. This is a controlled mechanism testbed and never a
stand-in for a real loader.

CLI
---
    python experiments/run_phase_sensitivity.py            # default sweep
    python experiments/run_phase_sensitivity.py --quick    # tiny smoke sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.run_protocol_phase_diagram import (  # noqa: E402
    GenConfig,
    run_phase_sweep,
)

# Knob -> values to sweep. Each is applied as a single-field override on the base
# config; everything else is held fixed so the effect is attributable.
DEFAULT_KNOB_GRID: Dict[str, List[float]] = {
    "k_neighbours": [3, 5, 8],
    "homophily_train": [0.75, 0.85, 0.95],
    "struct_sep": [1.0, 1.5, 2.0],
}


def _row_from_result(knob: str, value: float, res) -> Dict[str, object]:
    ag = res.agreement
    frac_rev = ag["sim_reversal_cells"] / max(ag["n_cells"], 1)
    return {
        "knob": knob,
        "value": value,
        "reversal_agreement": round(ag["reversal_binary_agreement"], 4),
        "reversal_ci_lo": ag["reversal_agreement_ci95"][0],
        "reversal_ci_hi": ag["reversal_agreement_ci95"][1],
        "holdout_agreement": (None if ag["holdout_reversal_agreement"] is None
                              else round(ag["holdout_reversal_agreement"], 4)),
        "holdout_ci_lo": ag["holdout_reversal_agreement_ci95"][0],
        "holdout_ci_hi": ag["holdout_reversal_agreement_ci95"][1],
        "frac_cells_reversed": round(frac_rev, 4),
        "n_cells": ag["n_cells"],
        "auc_critical_decay": res.theory_params.get("auc_critical_decay"),
    }


def run_sensitivity(
    base: GenConfig,
    knob_grid: Dict[str, List[float]],
    h_points: int,
    delta_points: int,
    delta_max: float,
    seeds: List[int],
    verbose: bool = False,
) -> Dict[str, object]:
    """Sweep each knob value, re-running a phase sweep and recording agreement."""
    t0 = time.time()
    h_grid = np.linspace(0.0, 1.0, h_points)
    delta_grid = np.linspace(0.0, delta_max, delta_points)

    rows: List[Dict[str, object]] = []
    for knob, values in knob_grid.items():
        for value in values:
            cfg = replace(base, **{knob: type(getattr(base, knob))(value)})
            res = run_phase_sweep(h_grid, delta_grid, cfg, seeds, verbose=False)
            row = _row_from_result(knob, value, res)
            rows.append(row)
            if verbose:
                print(f"  [{knob}={value}] reversal_agree={row['reversal_agreement']:.3f} "
                      f"holdout={row['holdout_agreement']} "
                      f"frac_reversed={row['frac_cells_reversed']:.3f}")

    agrees = [r["reversal_agreement"] for r in rows]
    fracs = [r["frac_cells_reversed"] for r in rows]
    summary = {
        "min_reversal_agreement": float(min(agrees)) if agrees else None,
        "mean_reversal_agreement": round(float(np.mean(agrees)), 4) if agrees else None,
        "min_frac_cells_reversed": float(min(fracs)) if fracs else None,
        "all_settings_show_reversal": bool(all(f > 0 for f in fracs)) if fracs else False,
    }
    return {
        "schema": "protocol_phase_sensitivity_v1",
        "base_config": asdict(base),
        "knobs_swept": {k: list(v) for k, v in knob_grid.items()},
        "grid": {"h_points": h_points, "delta_points": delta_points,
                 "delta_max": delta_max, "seeds": list(seeds)},
        "rows": rows,
        "summary": summary,
        "wall_sec": round(time.time() - t0, 1),
    }


def write_outputs(result: Dict[str, object], json_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    cols = ["knob", "value", "reversal_agreement", "reversal_ci_lo", "reversal_ci_hi",
            "holdout_agreement", "holdout_ci_lo", "holdout_ci_hi",
            "frac_cells_reversed", "n_cells", "auc_critical_decay"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in result["rows"]:  # type: ignore[index]
            w.writerow({k: r.get(k, "") for k in cols})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/phase/sensitivity/phase_sensitivity.json")
    p.add_argument("--csv", default="results/phase/sensitivity/phase_sensitivity.csv")
    p.add_argument("--quick", action="store_true", help="tiny smoke sweep")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--h-points", type=int, default=7)
    p.add_argument("--delta-points", type=int, default=5)
    p.add_argument("--delta-max", type=float, default=0.30)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        base = GenConfig(n_timesteps=8, nodes_per_step=160)
        knob_grid = {"homophily_train": [0.8, 0.9], "struct_sep": [1.2, 1.8]}
        h_points, delta_points, seeds = 4, 3, [0]
    else:
        base = GenConfig(nodes_per_step=240, n_timesteps=10)
        knob_grid = DEFAULT_KNOB_GRID
        h_points, delta_points, seeds = args.h_points, args.delta_points, list(args.seeds)

    n_settings = sum(len(v) for v in knob_grid.values())
    print(f"[sensitivity] {n_settings} settings  grid {h_points}x{delta_points}  seeds={seeds}")
    result = run_sensitivity(base, knob_grid, h_points, delta_points,
                             args.delta_max, seeds, verbose=True)
    write_outputs(result, args.out, args.csv)

    s = result["summary"]  # type: ignore[index]
    print(f"\n[sensitivity] wrote {args.out} and {args.csv}")
    print(f"[sensitivity] min reversal agreement across settings: "
          f"{s['min_reversal_agreement']}  (mean {s['mean_reversal_agreement']})")
    print(f"[sensitivity] all settings show reversal: {s['all_settings_show_reversal']} "
          f"(min frac reversed {s['min_frac_cells_reversed']})")
    print(f"[sensitivity] wall={result['wall_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
