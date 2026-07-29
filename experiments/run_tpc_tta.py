"""
experiments/run_tpc_tta.py

Focused evaluation of the TPC+TTA solution against four ablations.

While ``run_multi_dataset.py`` reports the single "TPC+TTA" column for every
(dataset, model, seed) triple, this script drills into *why* the method
works by reporting each ingredient in isolation on top of the inductive
logits of a fixed reference backbone (GraphSAGE by default):

    raw         argmax(logits)                            — nothing applied
    temp        argmax(softmax(logits / T*))              — calibration only
    prior       log-linear prior correction, fixed T*     — priors only
    thresh      F1-optimal threshold on raw p_pos          — threshold only
    tpc_tta     temperature + prior + threshold           — proposed method

If the contribution of any one of prior / threshold vanishes, that is a
clean signal that the corresponding mechanism was not actually doing work.
The resulting five-column table is what supports the paper's Sec. 8
"mechanism" figure.

Output: ``results/tpc_tta/{dataset}__{model}__seed{s}.json`` and a summary
printed to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.datasets import load_dataset                                  # noqa: E402
from data.datasets.base import describe_dataset                        # noqa: E402
from experiments._multi_harness import (                                # noqa: E402
    RunConfig, build_inductive_view, _train,
)
from experiments.run_metadata import build_run_metadata                  # noqa: E402
from experiments.result_audit import (                                  # noqa: E402
    audit_sweep,
    summarize_rows,
    write_audit_reports,
    write_json_atomic,
)
from models.registry import build_model                                 # noqa: E402
from models.temporal_calibration import (                               # noqa: E402
    TPCCalibrator, _softmax, _fit_threshold, _binary_f1,
)


def _metrics(preds: np.ndarray, y: np.ndarray, mask: np.ndarray,
             pos: int = 1) -> Dict[str, float]:
    y_m, p_m = y[mask], preds[mask]
    tp = int(((p_m == pos) & (y_m == pos)).sum())
    fp = int(((p_m == pos) & (y_m != pos)).sum())
    fn = int(((p_m != pos) & (y_m == pos)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = _binary_f1(y_m, p_m, pos_label=pos)
    return {"f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)}


def ablate(logits: torch.Tensor, data, cfg) -> Dict[str, Dict[str, float]]:
    logits_np = logits.cpu().numpy()
    y         = data.y.cpu().numpy()
    t         = data.time_step.cpu().numpy()
    vm        = data.val_mask.cpu().numpy().astype(bool)
    tm        = data.test_mask.cpu().numpy().astype(bool)
    trm       = data.train_mask.cpu().numpy().astype(bool)
    pos       = int(data.pos_label)

    # ---- raw argmax ----
    preds_raw = logits_np.argmax(-1)

    # ---- temperature-only ----
    cal = TPCCalibrator.fit(
        val_logits = logits_np[vm],
        val_y      = y[vm],
        y_train    = y[trm],
        pos_label  = pos,
    )
    probs_T   = _softmax(logits_np, temperature=cal.temperature)
    preds_temp = probs_T.argmax(-1)

    # ---- prior-only (no F1-threshold) ----
    prior     = cal.rolling_prior(logits_np, t, window=cfg.tpc_window)
    preds_prior = cal.predict(
        logits_np, t, target_prior=prior, use_threshold=False,
    )

    # ---- threshold-only (no prior correction) ----
    tau       = _fit_threshold(probs_T[vm], y[vm], pos_label=pos)
    p_pos     = probs_T[:, pos]
    preds_thresh = np.where(p_pos >= tau, pos, probs_T.argmax(-1))

    # ---- full TPC+TTA ----
    preds_full = cal.predict(
        logits_np, t, target_prior=prior, use_threshold=True,
    )

    return {
        "raw":     _metrics(preds_raw,    y, tm, pos),
        "temp":    _metrics(preds_temp,   y, tm, pos),
        "prior":   _metrics(preds_prior,  y, tm, pos),
        "thresh":  _metrics(preds_thresh, y, tm, pos),
        "tpc_tta": _metrics(preds_full,   y, tm, pos),
        "calibration": {
            "temperature": round(cal.temperature, 4),
            "threshold":   round(cal.threshold,   4),
            "train_prior": [round(float(v), 6) for v in cal.train_prior],
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+",
                   default=["elliptic", "dgraphfin", "tfinance"])
    p.add_argument("--models",   nargs="+",
                   default=["sage", "graph_transformer", "gps", "pcgnn"])
    p.add_argument("--seeds",    nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--hidden",   type=int, default=256)
    p.add_argument("--layers",   type=int, default=3)
    p.add_argument("--dropout",  type=float, default=0.5)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--window",   type=int, default=3)
    p.add_argument("--device",   type=str, default="auto")
    p.add_argument("--out",      type=str, default="results/tpc_tta")
    p.add_argument(
        "--scaler-mode",
        choices=["train_only", "full_population", "none"],
        default="train_only",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Write an artifact manifest for the selected grid and exit without loading data or training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.plan_only:
        rows = audit_sweep(
            "tpc_tta",
            result_dir=args.out,
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
        )
        counts = summarize_rows(rows)
        print("TPC+TTA plan: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        write_audit_reports(
            rows,
            csv_path="results/reports/tpc_tta_plan.csv",
            md_path="results/reports/tpc_tta_plan.md",
            title="TPC+TTA sweep plan",
        )
        print("[plan] wrote results/reports/tpc_tta_plan.md")
        print("[plan] no datasets loaded; no training launched")
        return

    os.makedirs(args.out, exist_ok=True)

    for ds in args.datasets:
        data = load_dataset(ds, scaler_mode=args.scaler_mode)
        print(describe_dataset(data))
        train_max = int(data.time_step[data.train_mask].max().item())
        sub = build_inductive_view(data, train_max=train_max)

        for mdl in args.models:
            for sd in args.seeds:
                tag = f"{ds}__{mdl}__seed{sd}.json"
                path = os.path.join(args.out, tag)
                if os.path.exists(path):
                    print(f"[skip] {tag}"); continue
                print(f"\n[run] {tag}")
                cfg = RunConfig(
                    dataset=ds, model=mdl, seed=sd,
                    hidden_channels=args.hidden, num_layers=args.layers,
                    dropout=args.dropout, lr=args.lr,
                    epochs=args.epochs, patience=args.patience,
                    device=args.device, scaler_mode=args.scaler_mode,
                    tpc_window=args.window, results_dir=args.out,
                )
                built = build_model(
                    mdl, in_channels=data.num_node_features,
                    hidden_channels=cfg.hidden_channels,
                    num_layers=cfg.num_layers, dropout=cfg.dropout,
                    out_channels=3,
                )
                logits, info = _train(built, sub, data, cfg, setting="inductive")
                abl = ablate(logits, data, cfg)
                out = {
                    "result_schema_version": "tpc_tta_v2",
                    "dataset": ds, "model": mdl, "seed": sd,
                    "ablation": abl, "info": info,
                    "hyperparams": asdict(cfg),
                    "scaler_mode": data.meta.get("scaler_mode", cfg.scaler_mode),
                    "protocol": {
                        "setting": "inductive logits only",
                        "raw": "argmax(logits)",
                        "temp": "temperature-scaled argmax",
                        "prior": "temporal prior correction without threshold",
                        "thresh": "validation threshold without temporal prior",
                        "tpc_tta": "temperature, prior correction, and threshold",
                    },
                    "run_metadata": build_run_metadata("run_tpc_tta", asdict(cfg)),
                }
                write_json_atomic(path, out)
                print("  " + " | ".join(
                    f"{k}:{v['f1']:.3f}" for k, v in abl.items()
                    if k != "calibration"
                ))


if __name__ == "__main__":
    main()
