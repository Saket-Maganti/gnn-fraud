"""Protocol sensitivity and ranking volatility helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def protocol_sensitivity_index(rows: Iterable[dict], metrics: tuple[str, ...] = ("auprc", "auroc", "f1")) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset"), row.get("variant"), row.get("model"))].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        protocols = sorted({item.get("protocol") for item in items})
        if len(protocols) < 2:
            out.append({"dataset": key[0], "variant": key[1], "model": key[2], "status": "NOT_APPLICABLE"})
            continue
        diffs = []
        by_protocol = {item.get("protocol"): item for item in items}
        for metric in metrics:
            vals = []
            for protocol in protocols[:2]:
                try:
                    vals.append(float(by_protocol[protocol].get(metric)))
                except Exception:
                    vals = []
                    break
            if len(vals) == 2:
                diffs.append(abs(vals[1] - vals[0]))
        out.append({"dataset": key[0], "variant": key[1], "model": key[2], "status": "PASS" if diffs else "NOT_APPLICABLE", "psi": sum(diffs) / len(diffs) if diffs else None})
    return out


def top_model_disagreement(rows: Iterable[dict], metrics: tuple[str, ...] = ("auprc", "auroc", "f1")) -> dict:
    winners = {}
    items = list(rows)
    for metric in metrics:
        best = None
        best_val = None
        for row in items:
            try:
                val = float(row.get(metric))
            except Exception:
                continue
            if best_val is None or val > best_val:
                best_val = val
                best = row.get("model")
        winners[metric] = best
    unique = {winner for winner in winners.values() if winner}
    return {"winners": winners, "top_model_disagreement_rate": (len(unique) - 1) / max(len(metrics), 1) if unique else 0.0}
