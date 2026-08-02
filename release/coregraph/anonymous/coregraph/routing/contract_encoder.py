"""Factorised, atomic, and no-contract encoder variants."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch
import torch.nn as nn

from coregraph.contracts.axes import (
    DeviceClass,
    EdgeFeaturePolicy,
    EdgeVisibility,
    HistoryPolicy,
    MeasurementStatus,
    NodeVisibility,
    Orientation,
    ReviewMode,
    SelectionAxis,
    TimeAxis,
    TopologyTransform,
)
from coregraph.contracts.contract import DeploymentContract

_FIELDS: dict[str, type[Enum]] = {
    "time_mode": TimeAxis,
    "visibility_node": NodeVisibility,
    "visibility_edge": EdgeVisibility,
    "construction_history": HistoryPolicy,
    "construction_orientation": Orientation,
    "construction_edge_features": EdgeFeaturePolicy,
    "construction_topology": TopologyTransform,
    "selection": SelectionAxis,
    "budget_review": ReviewMode,
    "resource_device": DeviceClass,
    "resource_measurement": MeasurementStatus,
}
_COORDINATE_FIELDS = {
    "time": ("time_mode",),
    "visibility": ("visibility_node", "visibility_edge"),
    "construction": (
        "construction_history",
        "construction_orientation",
        "construction_edge_features",
        "construction_topology",
    ),
    "selection": ("selection",),
    "budget": ("budget_review",),
    "resource": ("resource_device", "resource_measurement"),
}


def _categorical_values(contract: DeploymentContract) -> dict[str, str]:
    return {
        "time_mode": contract.time.mode.value,
        "visibility_node": contract.visibility.node_visibility.value,
        "visibility_edge": contract.visibility.edge_visibility.value,
        "construction_history": contract.construction.history_policy.value,
        "construction_orientation": contract.construction.orientation.value,
        "construction_edge_features": (
            contract.construction.edge_feature_policy.value
        ),
        "construction_topology": contract.construction.topology_transform.value,
        "selection": contract.selection.value,
        "budget_review": contract.budget.review_mode.value,
        "resource_device": contract.resource.device_class.value,
        "resource_measurement": contract.resource.measurement_status.value,
    }


def continuous_contract_features(contract: DeploymentContract) -> list[float]:
    costs = contract.budget.cost_matrix or ((0.0, 0.0), (0.0, 0.0))
    return [
        float(contract.time.start or 0.0),
        float(contract.time.end or 0.0),
        float(contract.time.window or 0),
        float(contract.visibility.target_node_availability),
        float(contract.visibility.target_edge_availability),
        float(contract.visibility.label_free_target_covariates),
        float(contract.visibility.test_time_graph_access),
        float(contract.visibility.historical_only),
        float(contract.construction.recent_window or 0),
        float(contract.construction.degree_cap or 0),
        float(contract.budget.review_fraction or 0.0),
        float(contract.budget.fixed_k or 0),
        float(contract.budget.abstention_capacity or 0.0),
        float(contract.budget.latency_allowance_ms or 0.0),
        *(float(value) for row in costs for value in row),
        float(contract.resource.memory_cap_gb or 0.0),
        float(contract.resource.latency_cap_ms or 0.0),
        float(len(contract.resource.unavailable_experts)),
    ]


class FactorisedContractEncoder(nn.Module):
    """One learned representation per top-level coordinate."""

    def __init__(
        self,
        embedding_dim: int = 8,
        output_dim: int = 32,
        *,
        pairwise_interactions: bool = True,
        axis_dropout: float = 0.0,
        contract_noise_std: float = 0.0,
    ):
        super().__init__()
        if embedding_dim <= 0 or output_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if not 0 <= axis_dropout < 1 or contract_noise_std < 0:
            raise ValueError("invalid dropout/noise configuration")
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.pairwise_interactions = pairwise_interactions
        self.axis_dropout = axis_dropout
        self.contract_noise_std = contract_noise_std
        self.value_to_index: dict[str, dict[str, int]] = {}
        for name, enum_type in _FIELDS.items():
            self.value_to_index[name] = {
                str(member.value): index
                for index, member in enumerate(enum_type)
            }
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(len(mapping), embedding_dim)
                for name, mapping in self.value_to_index.items()
            }
        )
        pair_count = (
            len(_COORDINATE_FIELDS) * (len(_COORDINATE_FIELDS) - 1) // 2
            if pairwise_interactions
            else 0
        )
        continuous_dim = 21
        input_dim = (
            len(_COORDINATE_FIELDS) * embedding_dim
            + pair_count
            + continuous_dim
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def encode_indices(
        self,
        contracts: Sequence[DeploymentContract],
        *,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        values = [_categorical_values(contract) for contract in contracts]
        return {
            name: torch.tensor(
                [self.value_to_index[name][row[name]] for row in values],
                dtype=torch.long,
                device=device,
            )
            for name in _FIELDS
        }

    def forward(self, contracts: Sequence[DeploymentContract]) -> torch.Tensor:
        if not contracts:
            raise ValueError("contract encoder requires at least one contract")
        device = next(self.parameters()).device
        indices = self.encode_indices(contracts, device=device)
        coordinate_vectors: list[torch.Tensor] = []
        for coordinate, fields in _COORDINATE_FIELDS.items():
            del coordinate
            parts = [self.embeddings[name](indices[name]) for name in fields]
            vector = torch.stack(parts, dim=0).mean(dim=0)
            if self.training and self.axis_dropout > 0:
                drop = (
                    torch.rand((len(contracts), 1), device=device)
                    < self.axis_dropout
                )
                vector = torch.where(drop, torch.zeros_like(vector), vector)
            coordinate_vectors.append(vector)
        interactions: list[torch.Tensor] = []
        if self.pairwise_interactions:
            for left in range(len(coordinate_vectors)):
                for right in range(left + 1, len(coordinate_vectors)):
                    interactions.append(
                        (
                            coordinate_vectors[left]
                            * coordinate_vectors[right]
                        ).sum(dim=-1, keepdim=True)
                    )
        continuous = torch.tensor(
            [continuous_contract_features(contract) for contract in contracts],
            dtype=torch.float32,
            device=device,
        )
        if self.training and self.contract_noise_std > 0:
            continuous = (
                continuous
                + torch.randn_like(continuous) * self.contract_noise_std
            )
        joined = torch.cat(
            [*coordinate_vectors, *interactions, continuous],
            dim=-1,
        )
        return self.projection(joined)


class AtomicContractEncoder(nn.Module):
    """Seen-contract baseline; unknown target IDs map to a dedicated token."""

    def __init__(self, seen_contract_ids: Sequence[str], output_dim: int = 32):
        super().__init__()
        self.mapping = {
            contract_id: index + 1
            for index, contract_id in enumerate(sorted(set(seen_contract_ids)))
        }
        self.embedding = nn.Embedding(len(self.mapping) + 1, output_dim)

    def forward(self, contracts: Sequence[DeploymentContract]) -> torch.Tensor:
        device = self.embedding.weight.device
        ids = torch.tensor(
            [self.mapping.get(contract.contract_id, 0) for contract in contracts],
            dtype=torch.long,
            device=device,
        )
        return self.embedding(ids)


class NoContractEncoder(nn.Module):
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.anchor = nn.Parameter(torch.zeros(output_dim))

    def forward(self, contracts: Sequence[DeploymentContract]) -> torch.Tensor:
        return self.anchor.unsqueeze(0).expand(len(contracts), -1)
