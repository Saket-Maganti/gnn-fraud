from __future__ import annotations

import numpy as np

from coregraph.data.elliptic_v2 import EllipticV2Adapter
from coregraph.data.graph_views import ViewRole
from coregraph.experts.sampled_graph_expert import SampledNodeGraphExpert
from coregraph.experts.sampling import SamplingPlan


def test_one_epoch_sampled_gcn_uses_fold_specific_views(contract_factory) -> None:
    n = 12
    edges = np.asarray(
        [
            [*range(n - 1), *range(1, n)],
            [*range(1, n), *range(n - 1)],
        ]
    )
    dataset = EllipticV2Adapter.from_arrays(
        features=np.column_stack([np.arange(n), np.arange(n) % 3]).astype(float),
        labels=np.asarray([1, 2] * (n // 2)),
        node_timestamps=np.arange(1, n + 1),
        edge_index=edges,
        contract=contract_factory(),
        train_cutoff=4,
        validation_cutoff=7,
        target_cutoff=12,
    )
    expert = SampledNodeGraphExpert(
        model_name="gcn",
        expert_id="gcn_fixture",
        hidden_channels=8,
        epochs=1,
        sampling=SamplingPlan(batch_size=4, fanouts=(2,), seed=3),
    )
    expert.fit(dataset.batch_for_role(ViewRole.TRAIN))
    scores = expert.predict_scores(dataset.batch_for_role(ViewRole.TARGET))
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
