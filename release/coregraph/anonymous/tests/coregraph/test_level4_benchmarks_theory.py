from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from coregraph.baselines.registry import BaselineStatus, level4_baselines
from coregraph.baselines.simple import SourceLogisticGate, best_fixed_expert, cheapest_feasible_expert, confidence_selection, oracle_diagnostics, random_feasible_expert, uniform_average, validation_weighted_average
from coregraph.benchmarks.fraud_contracts import canonical_fraud_contracts
from coregraph.benchmarks.graph_ood import candidate_registry, tiny_stand_in
from coregraph.benchmarks.synthetic_mechanisms import Mechanism, generate_mechanism_fixture, mechanism_registry
from coregraph.theory.regret_decomposition import RegretDecomposition
from coregraph.theory.selective_risk import selective_risk_transfer_bound
from coregraph.theory_checks import run_level4_theory_checks

ROOT = Path(__file__).resolve().parents[2]


def test_all_fifteen_mechanisms_are_deterministic_and_tiny() -> None:
    assert len(mechanism_registry()) == len(Mechanism) == 15
    hashes = []
    for mechanism in Mechanism:
        first = generate_mechanism_fixture(mechanism, size=40, seed=3)
        second = generate_mechanism_fixture(mechanism, size=40, seed=3)
        assert first.deterministic_sha256 == second.deterministic_sha256
        assert first.expert_scores.shape == (40, 3)
        hashes.append(first.deterministic_sha256)
    assert len(set(hashes)) == 15
    dynamic = generate_mechanism_fixture(Mechanism.DYNAMIC_RESOURCE_CHANGE, size=40)
    assert not np.array_equal(dynamic.feasible_mask[0], dynamic.feasible_mask[-1])


def test_three_layer_registries_are_explicit_and_results_blocked() -> None:
    fraud = canonical_fraud_contracts()
    assert len(fraud) == 6 and all(item.label_access == "NO_TARGET_LABELS_DURING_FITTING" for item in fraud)
    candidates = candidate_registry()
    assert candidates[0].family == "GOOD" and candidates[0].role == "PRIMARY"
    assert tiny_stand_in("GOOD")["official_result_claim"] == "PROHIBITED"
    with pytest.raises(KeyError):
        tiny_stand_in("unknown")


def test_baseline_registry_and_simple_methods() -> None:
    registry = level4_baselines()
    assert any(item.status is BaselineStatus.UNAVAILABLE_LICENSE for item in registry)
    assert all(not item.deployable for item in registry if "oracle" in item.baseline_id)
    scores = np.asarray([[0.1, 0.9], [0.8, 0.4], [0.2, 0.7]])
    feasible = np.asarray([[True, True], [True, False], [False, False]])
    risk = np.asarray([0.2, 0.1])
    assert np.isnan(uniform_average(scores, feasible)[-1])
    assert np.isnan(validation_weighted_average(scores, feasible, risk)[-1])
    assert best_fixed_expert(scores, feasible, risk)[1] == scores[1, 0]
    assert confidence_selection(scores, feasible)[0] in scores[0]
    assert cheapest_feasible_expert(scores, feasible, np.asarray([1.0, 2.0]))[0] == scores[0, 0]
    assert np.isnan(random_feasible_expert(scores, feasible, seed=1)[-1])
    oracle = oracle_diagnostics(scores[:2], np.asarray([0, 1]), feasible[:2])
    assert oracle["contract_oracle_expert"] == 0
    gate = SourceLogisticGate(steps=5).fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([0, 1, 1]))
    weights = gate.predict_weights(np.asarray([[0.5], [1.5]]), np.asarray([[True, True], [True, False]]))
    assert weights.shape == (2, 2) and weights[1, 1] == 0


def test_theory_checks_and_scoped_bounds() -> None:
    assert all(run_level4_theory_checks().values())
    decomposition = RegretDecomposition(0.1, 0.2, 0.1, 0.0, 0.1, 0.1, 0.2)
    assert decomposition.upper_bound == pytest.approx(0.8)
    assert "THEORETICAL" in decomposition.observable_status()["representation_error"]
    assert selective_risk_transfer_bound(source_selective_risk=0.1, source_coverage=0.8, target_coverage_lower_bound=0.4, density_ratio_bound=2.0) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        selective_risk_transfer_bound(source_selective_risk=0.1, source_coverage=0.8, target_coverage_lower_bound=0.0, density_ratio_bound=2.0)


def test_standalone_theory_entry_point_is_path_safe() -> None:
    completed = subprocess.run(
        [sys.executable, "theory/coregraph_level4/executable_checks.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["all_pass"] is True
