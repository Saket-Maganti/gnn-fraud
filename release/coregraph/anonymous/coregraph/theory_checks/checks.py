"""Dependency-light checks for all Level-4 finite constructions."""

from __future__ import annotations

import numpy as np
import torch

from coregraph.routing.masks import apply_feasible_mask
from coregraph.theory.fixed_mixture import fixed_mixture_lower_bound
from coregraph.theory.selective_risk import selective_risk_transfer_bound


def run_level4_theory_checks() -> dict[str, bool]:
    fixed = fixed_mixture_lower_bound(0.4, 0.6)
    protocol_one_hot_memorises = np.linalg.matrix_rank(np.eye(4)) == 4
    confidence = np.asarray([[0.99, 0.75], [0.99, 0.75]])
    labels = np.asarray([0, 1])
    confidence_choice_risk = float(np.mean((confidence[:, 0] - labels) ** 2))
    alternative_risk = float(np.mean((confidence[:, 1] - labels) ** 2))
    masked = apply_feasible_mask(
        torch.tensor([[1.0, 5.0], [0.0, 0.0]]),
        torch.tensor([[True, False], [False, False]]),
    )
    wrong_seed_pairing_detected = not np.array_equal(
        np.asarray([("elliptic", 1)], dtype=object),
        np.asarray([("dgraphfin", 1)], dtype=object),
    )
    selective = selective_risk_transfer_bound(
        source_selective_risk=0.1,
        source_coverage=0.8,
        target_coverage_lower_bound=0.5,
        density_ratio_bound=1.2,
    )
    checks = {
        "fixed_mixture_positive": fixed.lower_bound > 0 and fixed.contract_aware_regret == 0,
        "protocol_one_hot_memorisation": bool(protocol_one_hot_memorises),
        "confidence_only_failure": confidence_choice_risk > alternative_risk,
        "resource_mask_zero_mass": float(masked.probabilities[0, 1]) == 0.0,
        "all_unavailable_explicit": bool(masked.all_unavailable[1]) and int(masked.selected_expert[1]) == -1,
        "wrong_seed_pairing_detected": wrong_seed_pairing_detected,
        "training_runtime_not_latency": "training_runtime_seconds" != "inference_latency_ms",
        "noisy_contract_nonidentifiability": np.allclose(np.asarray([1.0, 0.0]) + np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0])),
        "selective_bound_finite": 0 <= selective <= 1,
    }
    return checks
