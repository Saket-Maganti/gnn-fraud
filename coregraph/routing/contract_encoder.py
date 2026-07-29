"""Factorised, atomic, and no-contract encoder variants."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from coregraph.contracts.axes import (
    BudgetAxis,
    ConstructionAxis,
    ResourceAxis,
    SelectionAxis,
    TimeAxis,
    VisibilityAxis,
)
from coregraph.contracts.contract import DeploymentContract

AxisEnum = (
    type[TimeAxis]
    | type[VisibilityAxis]
    | type[ConstructionAxis]
    | type[SelectionAxis]
    | type[BudgetAxis]
    | type[ResourceAxis]
)

_AXES: dict[str, AxisEnum] = {
    "time": TimeAxis,
    "visibility": VisibilityAxis,
    "construction": ConstructionAxis,
    "selection": SelectionAxis,
    "budget": BudgetAxis,
    "resource": ResourceAxis,
}


def _axis_values(contract: DeploymentContract) -> dict[str, str]:
    return {
        "time": contract.time.mode.value,
        "visibility": contract.visibility.value,
        "construction": contract.construction.mode.value,
        "selection": contract.selection.value,
        "budget": contract.budget.mode.value,
        "resource": contract.resource.mode.value,
    }


def continuous_contract_features(contract: DeploymentContract) -> list[float]:
    return [
        float(contract.time.start or 0.0),
        float(contract.time.end or 0.0),
        float(contract.time.window or 0),
        float(contract.construction.recent_window or 0),
        float(contract.construction.degree_cap or 0),
        float(contract.budget.value or 0.0),
        float(contract.resource.memory_gb or 0.0),
        float(contract.resource.latency_ms or 0.0),
    ]


class FactorisedContractEncoder(nn.Module):
    """One embedding per axis with optional pairwise interactions."""

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
        self.value_to_index: dict[str, dict[str, int]] = {
            name: {member.value: index for index, member in enumerate(enum)}
            for name, enum in _AXES.items()
        }
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(len(mapping), embedding_dim)
                for name, mapping in self.value_to_index.items()
            }
        )
        pair_count = len(_AXES) * (len(_AXES) - 1) // 2 if pairwise_interactions else 0
        input_dim = len(_AXES) * embedding_dim + pair_count + 8
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
        values = [_axis_values(contract) for contract in contracts]
        return {
            name: torch.tensor(
                [self.value_to_index[name][row[name]] for row in values],
                dtype=torch.long,
                device=device,
            )
            for name in _AXES
        }

    def forward(self, contracts: Sequence[DeploymentContract]) -> torch.Tensor:
        if not contracts:
            raise ValueError("contract encoder requires at least one contract")
        device = next(self.parameters()).device
        indices = self.encode_indices(contracts, device=device)
        encoded: list[torch.Tensor] = []
        unknown_index = {
            name: mapping["unknown"] for name, mapping in self.value_to_index.items()
        }
        for name in _AXES:
            idx = indices[name]
            if self.training and self.axis_dropout > 0:
                drop = torch.rand(idx.shape, device=device) < self.axis_dropout
                idx = torch.where(
                    drop,
                    torch.full_like(idx, unknown_index[name]),
                    idx,
                )
            encoded.append(self.embeddings[name](idx))
        interactions: list[torch.Tensor] = []
        if self.pairwise_interactions:
            for left in range(len(encoded)):
                for right in range(left + 1, len(encoded)):
                    interactions.append((encoded[left] * encoded[right]).sum(dim=-1, keepdim=True))
        continuous = torch.tensor(
            [continuous_contract_features(contract) for contract in contracts],
            dtype=torch.float32,
            device=device,
        )
        if self.training and self.contract_noise_std > 0:
            continuous = continuous + torch.randn_like(continuous) * self.contract_noise_std
        joined = torch.cat([*encoded, *interactions, continuous], dim=-1)
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
