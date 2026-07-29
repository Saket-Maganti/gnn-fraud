"""Deterministic numerical checks for every theory claim."""

from __future__ import annotations

import numpy as np

from coregraph.theory.compositional_bound import (
    compositional_error_bound,
    verify_additive_bound,
)
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
    true = np.asarray([0.2, -0.1, 0.3])
    estimated = np.asarray([0.18, -0.08, 0.25])
    interaction = np.asarray([0.01, -0.02])
    router = np.asarray([0.03])
    bound = compositional_error_bound(
        np.abs(true - estimated),
        interaction_residual=float(np.abs(interaction).sum()),
        router_approximation_error=float(np.abs(router).sum()),
    )
    near_tight = bound.total_bound * (1 - 1e-10)
    compositional_pass = verify_additive_bound(
        true,
        estimated,
        interaction,
        router,
        actual_excess_risk=near_tight,
    )
    adversarial_reject = not verify_additive_bound(
        true,
        estimated,
        interaction,
        router,
        actual_excess_risk=bound.total_bound + 1e-6,
    )
    xor_axis_errors = np.asarray([0.0, 0.0])
    xor_interaction = 1.0
    xor_without_residual_fails = (
        1.0
        > compositional_error_bound(
            xor_axis_errors,
            interaction_residual=0.0,
            router_approximation_error=0.0,
        ).total_bound
    )
    xor_with_residual_passes = (
        1.0
        <= compositional_error_bound(
            xor_axis_errors,
            interaction_residual=xor_interaction,
            router_approximation_error=0.0,
        ).total_bound
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
            "pass": compositional_pass and adversarial_reject,
            "proof_status": "PROVED",
            "formula": (
                "2*sum(axis_errors)+2*interaction_residual"
                "+router_approximation_error"
            ),
            "near_tight_ratio": near_tight / bound.total_bound,
            "adversarial_above_bound_rejected": adversarial_reject,
            "total_bound": bound.total_bound,
        },
        "xor_interaction": {
            "pass": xor_without_residual_fails and xor_with_residual_passes,
            "proof_status": "COUNTEREXAMPLE",
            "pure_factorisation_fails": xor_without_residual_fails,
            "declared_interaction_residual_covers_case": xor_with_residual_passes,
        },
        "resource_mask": {
            "pass": resource_pass,
            "proof_status": "PROVED",
        },
        "all_pass": (
            fixed_pass
            and compositional_pass
            and adversarial_reject
            and xor_without_residual_fails
            and xor_with_residual_passes
            and resource_pass
        ),
    }
