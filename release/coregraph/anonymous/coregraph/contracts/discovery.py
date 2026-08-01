"""Experimental label-free latent contract discovery and hybrid encoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LatentContractOutput:
    factors: torch.Tensor
    uncertainty: torch.Tensor
    out_of_support: torch.Tensor


class LatentContractDiscovery(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        experimental: bool = False,
    ) -> None:
        super().__init__()
        if not experimental:
            raise ValueError("latent contract discovery requires experimental=True")
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("discovery dimensions must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(latent_dim * 2, 8)),
            nn.ReLU(),
            nn.Linear(max(latent_dim * 2, 8), latent_dim + 2),
        )

    def forward(self, label_free_diagnostics: torch.Tensor) -> LatentContractOutput:
        if label_free_diagnostics.ndim != 2:
            raise ValueError("contract discovery expects [batch,diagnostics]")
        encoded = self.encoder(label_free_diagnostics)
        return LatentContractOutput(
            factors=encoded[:, :-2],
            uncertainty=torch.sigmoid(encoded[:, -2]),
            out_of_support=torch.sigmoid(encoded[:, -1]),
        )


class HybridContractEncoder(nn.Module):
    def __init__(self, supplied_dim: int, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(supplied_dim + latent_dim + 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        supplied: torch.Tensor,
        discovered: LatentContractOutput,
    ) -> torch.Tensor:
        if supplied.shape[0] != discovered.factors.shape[0]:
            raise ValueError("supplied and discovered contracts must align")
        joined = torch.cat(
            [
                supplied,
                discovered.factors,
                discovered.uncertainty[:, None],
                discovered.out_of_support[:, None],
            ],
            dim=-1,
        )
        return self.projection(joined)
