from __future__ import annotations

import numpy as np
import pytest
import torch

from coregraph.contracts.discovery import HybridContractEncoder, LatentContractDiscovery
from coregraph.contracts.encoders import ContractVocabulary, FactorisedContractEncoder, ProtocolOneHotEncoder
from coregraph.contracts.schema import AxisObservation, ContractObservation, ObservationState
from coregraph.contracts.uncertainty import SourceAxisStatistics
from coregraph.diagnostics.calibration import SourceCalibrationProxy
from coregraph.diagnostics.confidence import confidence_diagnostics
from coregraph.diagnostics.disagreement import disagreement_diagnostics
from coregraph.diagnostics.feature_shift import SourceFeatureReference
from coregraph.diagnostics.graph_shift import graph_shift_diagnostics
from coregraph.diagnostics.resource import ExpertResourceDiagnostic
from coregraph.evaluation.counterfactual import reroute_fixed_scores
from coregraph.evaluation.resources import ResourceMeasurement
from coregraph.evaluation.selective import selective_metrics
from coregraph.objectives.abstention import abstention_objective
from coregraph.routing.contract_router import ContractRouter
from coregraph.routing.hierarchical_router import HierarchicalRouter
from coregraph.routing.instance_router import InstanceRouter
from coregraph.routing.masks import apply_feasible_mask, compose_feasible_mask


def _contract(*, unseen: bool = False, missing: bool = False) -> ContractObservation:
    axes = {
        "time": AxisObservation(categorical="late" if unseen else "early", state=ObservationState.UNSEEN if unseen else ObservationState.OBSERVED),
        "visibility": AxisObservation(categorical="strict"),
        "construction": AxisObservation(state=ObservationState.MISSING) if missing else AxisObservation(categorical="full"),
        "selection": AxisObservation(categorical="source_only"),
        "budget": AxisObservation(continuous=0.01),
        "resource": AxisObservation(categorical="cpu"),
    }
    return ContractObservation(axes, contract_id="c")


def test_partial_contract_schema_and_source_statistics() -> None:
    source = [_contract(), _contract(missing=True)]
    vocabulary = ContractVocabulary.fit(source)
    statistics = SourceAxisStatistics.fit(source)
    encoder = FactorisedContractEncoder(vocabulary, statistics, embedding_dim=4, output_dim=8, attention=True)
    encoded = encoder([_contract(), _contract(unseen=True), _contract(missing=True)])
    assert encoded.shape == (3, 8)
    assert encoder.manifest()["normalization"] == "SOURCE_CONTRACTS_ONLY"
    assert vocabulary.index(0, "never-seen", ObservationState.UNSEEN) == 1
    assert _contract(missing=True).incomplete
    with pytest.raises(ValueError, match="missing axis cannot carry"):
        AxisObservation(categorical="x", state=ObservationState.MISSING)
    with pytest.raises(ValueError, match="contract axes must be exact"):
        ContractObservation({"time": AxisObservation(categorical="x")})


def test_protocol_baseline_and_experimental_discovery() -> None:
    one_hot = ProtocolOneHotEncoder(["strict", "isolated"], output_dim=3)
    assert one_hot(["strict", "unseen"]).shape == (2, 3)
    with pytest.raises(ValueError, match="experimental=True"):
        LatentContractDiscovery(4, 2)
    discovery = LatentContractDiscovery(4, 2, experimental=True)
    output = discovery(torch.ones(3, 4))
    hybrid = HybridContractEncoder(5, 2, 7)
    assert hybrid(torch.ones(3, 5), output).shape == (3, 7)


def test_diagnostics_are_label_free_at_deployment() -> None:
    scores = np.asarray([[0.1, 0.8], [0.9, 0.6]])
    assert confidence_diagnostics(scores)["entropy"].shape == scores.shape
    assert disagreement_diagnostics(scores)["pairwise_disagreement"].shape == (2, 1)
    graph = graph_shift_diagnostics(np.asarray([0, 1, 2]), edge_count=2, source_degree_histogram=np.asarray([1, 1, 1]))
    assert graph["isolated_node_fraction"] == pytest.approx(1 / 3)
    reference = SourceFeatureReference.fit(np.asarray([[0.0, 1.0], [1.0, 3.0]]))
    assert reference.score(np.asarray([[0.5, 2.0]])).shape == (1,)
    proxy = SourceCalibrationProxy.fit(np.asarray([0.1, 0.9]), np.asarray([0, 1]), bins=2)
    assert proxy.score(np.asarray([0.2, 0.8])).shape == (2,)
    resource = ExpertResourceDiagnostic("gcn", True, 2.0, 1.0, 0.2, 0.9, "MEASURED")
    assert resource.available


def test_resource_masks_and_all_unavailable_are_exact() -> None:
    availability = torch.tensor([[True, True, True], [True, True, True], [False, False, False]])
    memory = torch.tensor([1.0, 5.0, 2.0])
    latency = torch.tensor([2.0, 10.0, 3.0])
    feasible = compose_feasible_mask(availability, memory_gb=memory, memory_cap_gb=torch.tensor([3.0, 6.0, 3.0]), latency_ms=latency, latency_cap_ms=5.0)
    result = apply_feasible_mask(torch.tensor([[1.0, 9.0, 2.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), feasible)
    assert torch.equal(result.probabilities[0] > 0, torch.tensor([True, False, True]))
    assert result.probabilities[0].sum() == pytest.approx(1.0)
    assert torch.equal(result.probabilities[2], torch.zeros(3))
    assert int(result.selected_expert[2]) == -1
    with pytest.raises(ValueError, match="supplied together"):
        compose_feasible_mask(availability, memory_gb=memory)


def test_contract_instance_and_hierarchical_routing_modes() -> None:
    batch, experts = 4, 3
    contract = torch.randn(batch, 5)
    contract_diag = torch.randn(batch, 2)
    instances = torch.randn(batch, 6)
    instance_diag = torch.randn(batch, 2)
    feasible = torch.tensor([[True, True, False], [True, False, True], [True, True, True], [False, False, False]])
    assert ContractRouter(5, 2, experts)(contract, contract_diag, feasible).probabilities.shape == (batch, experts)
    instance = InstanceRouter(6, 2, experts)(instances, instance_diag, feasible)
    assert instance.routing.all_unavailable[-1]
    router = HierarchicalRouter(5, 6, 2, experts, max_correction=0.25, correction_threshold=0.01)
    output = router(contract_embedding=contract, contract_diagnostics=contract_diag, instance_features=instances, instance_diagnostics=instance_diag, feasible_mask=feasible, contract_group=torch.tensor([0, 0, 1, 1]))
    assert output.instance_correction.abs().max() <= 0.25 + 1e-6
    assert torch.isfinite(output.stability_penalty)
    contract_only = router(contract_embedding=contract, contract_diagnostics=contract_diag, instance_features=instances, instance_diagnostics=instance_diag, feasible_mask=feasible, contract_only=True)
    assert torch.equal(contract_only.instance_correction, torch.zeros_like(contract_only.instance_correction))


def test_abstention_selective_resource_and_counterfactual_semantics() -> None:
    total, terms = abstention_objective(torch.tensor([0.2, 0.8]), torch.tensor([0.0, 1.0]), abstention_cost=0.1, coverage_floor=0.75)
    assert torch.isfinite(total) and terms["coverage_penalty"] > 0
    zero = selective_metrics(np.asarray([0.2, 0.4]), np.asarray([True, True]))
    assert zero["selective_risk_status"] == "NOT_APPLICABLE_ZERO_COVERAGE"
    resource = ResourceMeasurement("gcn", 10, None, 2.0, 0.1, 100.0, None, 50.0, 1, 2, 4, "5 warmup", "cpu", "test", "MEASURED")
    assert resource.inference_latency_ms == 2.0
    scores = torch.tensor([[0.2, 0.9], [0.4, 0.6]])
    counter = reroute_fixed_scores(scores, torch.zeros_like(scores), torch.tensor([[True, False], [False, False]]))
    assert float(counter.weights[0, 1]) == 0.0 and bool(counter.all_unavailable[1])
