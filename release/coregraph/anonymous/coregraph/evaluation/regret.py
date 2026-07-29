"""Loss-parametric regret summaries.

Callers must supply router and oracle risks measured with one identical,
explicitly declared loss. Pilot result schemas use loss-specific names rather
than exporting these generic helper labels.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def contract_regrets(
    router_risk: Mapping[str, float],
    expert_risk: Mapping[str, Mapping[str, float]],
    availability: Mapping[str, Mapping[str, bool]],
) -> dict[str, float]:
    regrets: dict[str, float] = {}
    for contract, risk in router_risk.items():
        feasible = [
            value
            for expert, value in expert_risk[contract].items()
            if availability[contract].get(expert, False)
        ]
        if not feasible:
            raise ValueError(f"contract {contract} has no feasible oracle expert")
        regrets[contract] = float(risk - min(feasible))
    return regrets


def regret_summary(regrets: Mapping[str, float], *, alpha: float = 0.8) -> dict[str, float]:
    values = np.asarray(list(regrets.values()), dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarise an empty regret set")
    tail_count = max(1, int(np.ceil((1 - alpha) * len(values))))
    tail = np.sort(values)[-tail_count:]
    return {
        "mean_contract_regret": float(values.mean()),
        "maximum_contract_regret": float(values.max()),
        "median_contract_regret": float(np.median(values)),
        "cvar_contract_regret": float(tail.mean()),
    }
