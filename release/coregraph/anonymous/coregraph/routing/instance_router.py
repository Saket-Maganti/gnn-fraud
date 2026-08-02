"""Instance-level correction router with invocation accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from coregraph.routing.masks import MaskedRouting, apply_feasible_mask


@dataclass(frozen=True)
class InstanceRoutingOutput:
    routing: MaskedRouting
    expected_invocations: torch.Tensor


class InstanceRouter(nn.Module):
    def __init__(self, instance_dim: int, diagnostic_dim: int, num_experts: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(instance_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        instance_features: torch.Tensor,
        label_free_diagnostics: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> InstanceRoutingOutput:
        if instance_features.ndim != 2 or label_free_diagnostics.ndim != 2:
            raise ValueError("instance router inputs must be matrices")
        if instance_features.shape[0] != label_free_diagnostics.shape[0]:
            raise ValueError("instance router batches must align")
        logits = self.network(torch.cat([instance_features, label_free_diagnostics], dim=-1))
        routing = apply_feasible_mask(logits, feasible_mask)
        expected = (routing.probabilities > 0).float().sum(dim=-1)
        return InstanceRoutingOutput(routing, expected)
