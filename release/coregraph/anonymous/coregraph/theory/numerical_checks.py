"""Deterministic numerical checks for every theory claim."""

from __future__ import annotations

import numpy as np

from coregraph.theory.compositional_bound import verify_additive_bound
from coregraph.theory.fixed_mixture import (
    fixed_mixture_lower_bound,
    fixed_mixture_regret_curve,
)
from coregraph.theory.resource_mask import resource_mask_monotonicity


def run_numerical_checks() -> dict[str, object]:
    fixed = fixed_mixture_lower_bound(0.3, 0.5)
    weights = np.linspace(0, 1, 10001)
    curve = fixed_mixture_regret_curve(0.3, 0.5, weights)
    fixed_pass = abs(float(curve.min()) - fixed.lower_bound) < 1e-4
    compositional_pass = verify_additive_bound(
        np.asarray([0.2, -0.1, 0.3]),
        np.asarray([0.18, -0.08, 0.25]),
        np.asarray([0.01, -0.02]),
        np.asarray([0.03]),
    )
    resource_pass = resource_mask_monotonicity(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([True, True, True]),
        np.asarray([False, True, True]),
    )
    return {
        "fixed_mixture": {
            "pass": fixed_pass,
            "analytic_lower_bound": fixed.lower_bound,
            "grid_minimum": float(curve.min()),
            "proof_status": fixed.proof_status,
        },
        "compositional_bound": {
            "pass": compositional_pass,
            "proof_status": "PROVED",
        },
        "resource_mask": {
            "pass": resource_pass,
            "proof_status": "PROVED",
        },
        "all_pass": fixed_pass and compositional_pass and resource_pass,
    }
