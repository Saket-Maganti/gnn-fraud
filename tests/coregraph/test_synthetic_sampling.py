from __future__ import annotations

import numpy as np
import pytest

from coregraph.data.synthetic import (
    SyntheticControls,
    SyntheticRegime,
    generate_contract_shift,
    oracle_expert_ranking,
)
from coregraph.experts.graph_experts import guard_full_graph_execution
from coregraph.experts.sampling import deterministic_batches, temporal_event_batches


@pytest.mark.parametrize("regime", list(SyntheticRegime))
def test_all_synthetic_regimes_are_deterministic(regime: SyntheticRegime) -> None:
    controls = SyntheticControls(num_nodes=64, seed=11)
    left = generate_contract_shift(regime, controls)
    right = generate_contract_shift(regime, controls)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.edge_index, right.edge_index)
    assert oracle_expert_ranking(left)


def test_sampling_is_deterministic_and_temporal() -> None:
    ids = np.arange(9)
    left = list(deterministic_batches(ids, batch_size=4, seed=7, shuffle=True))
    right = list(deterministic_batches(ids, batch_size=4, seed=7, shuffle=True))
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    event_ids = np.asarray([3, 1, 2])
    times = np.asarray([2, 1, 2])
    ordered = np.concatenate(list(temporal_event_batches(event_ids, times, batch_size=2)))
    assert ordered.tolist() == [1, 2, 3]


def test_full_graph_guard_fails_before_oom() -> None:
    with pytest.raises(MemoryError, match="enable neighbour"):
        guard_full_graph_execution(
            nodes=1_000_000,
            edges=10_000_000,
            memory_cap_gb=0.1,
            hidden_channels=128,
            attention=False,
        )
