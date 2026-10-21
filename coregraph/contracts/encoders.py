"""Level-4 contract encoders with source-only normalization and unknown tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from coregraph.contracts.interactions import bounded_pairwise_interactions
from coregraph.contracts.schema import CONTRACT_AXES, ContractObservation, ObservationState
from coregraph.contracts.uncertainty import SourceAxisStatistics


@dataclass(frozen=True)
class ContractVocabulary:
    values: tuple[tuple[str, ...], ...]

    @classmethod
    def fit(cls, contracts: Sequence[ContractObservation]) -> "ContractVocabulary":
        if not contracts:
            raise ValueError("contract vocabulary requires source contracts")
        rows: list[tuple[str, ...]] = []
        for name in CONTRACT_AXES:
            observed_values: set[str] = set()
            for item in contracts:
                value = item.axes[name].categorical
                if value is not None:
                    observed_values.add(value)
            observed = sorted(observed_values)
            rows.append(("__MISSING__", "__UNKNOWN__", *observed))
        return cls(tuple(rows))

    def index(self, axis: int, value: str | None, state: ObservationState) -> int:
        if state is ObservationState.MISSING:
            return 0
        if value is None:
            return 1
        try:
            return self.values[axis].index(value)
        except ValueError:
            return 1


class FactorisedContractEncoder(nn.Module):
    """Embed each axis independently, then compose with bounded interactions."""

    def __init__(
        self,
        vocabulary: ContractVocabulary,
        statistics: SourceAxisStatistics,
        *,
        embedding_dim: int = 16,
        output_dim: int = 32,
        interactions: bool = True,
        attention: bool = False,
        interaction_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.vocabulary = vocabulary
        self.statistics = statistics
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.interactions = interactions
        self.attention_enabled = attention
        self.interaction_bound = interaction_bound
        self.category_embeddings = nn.ModuleList(
            nn.Embedding(len(values), embedding_dim) for values in vocabulary.values
        )
        self.state_embedding = nn.Embedding(len(ObservationState), embedding_dim)
        self.numeric_projection = nn.ModuleList(
            nn.Linear(3, embedding_dim) for _ in CONTRACT_AXES
        )
        self.attention = (
            nn.MultiheadAttention(embedding_dim, 1, batch_first=True)
            if attention
            else None
        )
        pair_count = len(CONTRACT_AXES) * (len(CONTRACT_AXES) - 1) // 2
        input_dim = len(CONTRACT_AXES) * embedding_dim + (pair_count if interactions else 0)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _axis_tensor(
        self,
        contracts: Sequence[ContractObservation],
        axis_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        name = CONTRACT_AXES[axis_index]
        observations = [contract.axes[name] for contract in contracts]
        categories = torch.tensor(
            [self.vocabulary.index(axis_index, item.categorical, item.state) for item in observations],
            dtype=torch.long,
            device=device,
        )
        state_indices = torch.tensor(
            [list(ObservationState).index(item.state) for item in observations],
            dtype=torch.long,
            device=device,
        )
        numeric_rows = []
        for item in observations:
            if item.continuous is None:
                normalized, outside, present = 0.0, 0.0, 0.0
            else:
                normalized, inferred_outside = self.statistics.normalize(
                    axis_index, item.continuous
                )
                outside = float(
                    inferred_outside or item.state is ObservationState.OUT_OF_RANGE
                )
                present = 1.0
            numeric_rows.append((normalized * item.confidence, outside, present))
        numeric = torch.tensor(numeric_rows, dtype=torch.float32, device=device)
        categorical = self.category_embeddings[axis_index](categories)
        state = self.state_embedding(state_indices)
        confidence = torch.tensor(
            [item.confidence for item in observations],
            dtype=torch.float32,
            device=device,
        )[:, None]
        return categorical * confidence + state + self.numeric_projection[axis_index](numeric)

    def forward(self, contracts: Sequence[ContractObservation]) -> torch.Tensor:
        if not contracts:
            raise ValueError("contract encoder requires at least one contract")
        device = next(self.parameters()).device
        axes = torch.stack(
            [self._axis_tensor(contracts, index, device) for index in range(len(CONTRACT_AXES))],
            dim=1,
        )
        if self.attention is not None:
            axes, _ = self.attention(axes, axes, axes, need_weights=False)
        pieces = [axes.flatten(start_dim=1)]
        if self.interactions:
            pieces.append(
                bounded_pairwise_interactions(axes, bound=self.interaction_bound)
            )
        return self.projection(torch.cat(pieces, dim=-1))

    def manifest(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "axes": list(CONTRACT_AXES),
            "embedding_dim": self.embedding_dim,
            "output_dim": self.output_dim,
            "interactions": self.interactions,
            "attention": self.attention_enabled,
            "unknown_token": "__UNKNOWN__",
            "missing_token": "__MISSING__",
            "normalization": "SOURCE_CONTRACTS_ONLY",
        }


class UncertaintyAwareContractEncoder(FactorisedContractEncoder):
    """Named Level-4 variant; uncertainty is encoded by state and confidence."""


class FlatContractMLP(FactorisedContractEncoder):
    """Flat ablation with interactions and axis attention disabled."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.update({"interactions": False, "attention": False})
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]


class ProtocolOneHotEncoder(nn.Module):
    """Memorization baseline that maps unseen protocol IDs to UNKNOWN."""

    def __init__(self, source_protocols: Sequence[str], output_dim: int = 16) -> None:
        super().__init__()
        values = ("__UNKNOWN__", *sorted(set(source_protocols)))
        self.index = {value: position for position, value in enumerate(values)}
        self.embedding = nn.Embedding(len(values), output_dim)

    def forward(self, protocol_ids: Sequence[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        indices = torch.tensor(
            [self.index.get(value, 0) for value in protocol_ids],
            dtype=torch.long,
            device=device,
        )
        return self.embedding(indices)
