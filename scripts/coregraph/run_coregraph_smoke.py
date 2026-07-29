#!/usr/bin/env python3
"""One-step CPU gradient and expert smoke test; never loads provider data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.contracts.axes import (  # noqa: E402
    AccessRegime,
    BudgetAxis,
    BudgetSpec,
    ConstructionAxis,
    ConstructionSpec,
    ContractRole,
    ResourceAxis,
    ResourceSpec,
    SelectionAxis,
    TimeAxis,
    TimeSpec,
    VisibilityAxis,
)
from coregraph.contracts.contract import DeploymentContract  # noqa: E402
from coregraph.data.elliptic_v2 import EllipticV2Adapter  # noqa: E402
from coregraph.data.graph_views import ViewRole  # noqa: E402
from coregraph.experts.feature_experts import LogisticRegressionExpert  # noqa: E402
from coregraph.experts.sampled_graph_expert import SampledNodeGraphExpert  # noqa: E402
from coregraph.experts.sampling import SamplingPlan  # noqa: E402
from coregraph.method import CoReGraph  # noqa: E402
from coregraph.tasks.base import PredictionUnit, TaskBatch  # noqa: E402
from coregraph.utils.seeding import seed_everything  # noqa: E402


def contract(environment: str) -> DeploymentContract:
    return DeploymentContract(
        environment_id=environment,
        role=ContractRole.TARGET,
        time=TimeSpec(TimeAxis.CHRONOLOGICAL_HOLDOUT),
        visibility=VisibilityAxis.STRICT_INDUCTIVE,
        construction=ConstructionSpec(ConstructionAxis.FULL_GRAPH),
        selection=SelectionAxis.NO_TARGET_ACCESS,
        budget=BudgetSpec(BudgetAxis.FRACTIONAL_REVIEW_CAPACITY, value=0.1),
        resource=ResourceSpec(ResourceAxis.CPU),
        access_regime=AccessRegime.DG_NO_TARGET,
        dataset_id="synthetic",
        task_id="node_fraud",
    )


def main() -> int:
    seed_everything(20260729)
    rng = np.random.default_rng(20260729)
    n = 96
    features = rng.normal(size=(n, 6))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)
    train = np.arange(n) < 56
    validation = (np.arange(n) >= 56) & (np.arange(n) < 72)
    test = np.arange(n) >= 72
    batch = TaskBatch(
        identifiers=np.asarray([f"node:{index}" for index in range(n)]),
        features=features,
        labels=labels,
        label_mask=np.ones(n, dtype=bool),
        train_mask=train,
        validation_mask=validation,
        test_mask=test,
        timestamps=np.arange(n),
        graph_view=None,
        edge_attributes=None,
        prediction_unit=PredictionUnit.NODE,
        contract_id="synthetic:smoke",
    )
    logistic = LogisticRegressionExpert(seed=20260729).fit(batch)
    feature_scores = logistic.predict_scores(batch)
    graph_like_scores = 1 / (1 + np.exp(-(features[:, 0] - features[:, 1])))
    expert_scores = torch.tensor(
        np.column_stack([feature_scores, graph_like_scores]), dtype=torch.float32
    )
    model = CoReGraph(
        num_experts=2,
        diagnostic_dim=3,
        axis_dropout=0.0,
        contract_noise_std=0.0,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = model(
        contracts=[contract(f"smoke_{index % 2}") for index in range(n)],
        expert_scores=expert_scores,
        diagnostics=torch.tensor(
            np.column_stack(
                [np.nanmean(features, axis=1), np.nanstd(features, axis=1), test]
            ),
            dtype=torch.float32,
        ),
        availability_mask=torch.ones((n, 2), dtype=torch.bool),
        expert_costs=torch.tensor([1.0, 3.0]),
        expert_names=("feature", "graph_like"),
    )
    loss = torch.nn.functional.binary_cross_entropy(
        output.blended_score, torch.tensor(labels, dtype=torch.float32)
    )
    optimiser.zero_grad()
    loss.backward()
    gradient_norm = float(
        sum(parameter.grad.abs().sum() for parameter in model.parameters() if parameter.grad is not None)
    )
    optimiser.step()
    assert gradient_norm > 0
    assert torch.allclose(output.expert_weights.sum(dim=1), torch.ones(n))
    graph_n = 24
    graph_edges = np.asarray(
        [
            [*range(graph_n - 1), *range(1, graph_n)],
            [*range(1, graph_n), *range(graph_n - 1)],
        ]
    )
    graph_dataset = EllipticV2Adapter.from_arrays(
        features=features[:graph_n],
        labels=np.where(labels[:graph_n] == 1, 1, 2),
        node_timestamps=np.arange(1, graph_n + 1),
        edge_index=graph_edges,
        contract=contract("graph_smoke"),
        train_cutoff=12,
        validation_cutoff=17,
        target_cutoff=24,
    )
    graph_expert = SampledNodeGraphExpert(
        model_name="gcn",
        expert_id="gcn_smoke",
        hidden_channels=8,
        epochs=1,
        sampling=SamplingPlan(batch_size=8, fanouts=(3,), seed=20260729),
    ).fit(graph_dataset.batch_for_role(ViewRole.TRAIN))
    graph_scores = graph_expert.predict_scores(
        graph_dataset.batch_for_role(ViewRole.TARGET)
    )
    assert np.isfinite(graph_scores).all()
    report = {
        "schema": "coregraph_cpu_smoke_v1",
        "status": "PASS",
        "examples": n,
        "loss": float(loss.detach()),
        "gradient_l1": gradient_norm,
        "provider_data_used": False,
        "epochs": 1,
        "sampled_gcn_examples": graph_n,
        "sampled_gcn_predictions_finite": True,
    }
    destination = ROOT / "results/coregraph_build/CPU_ONE_EPOCH_SMOKE.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
