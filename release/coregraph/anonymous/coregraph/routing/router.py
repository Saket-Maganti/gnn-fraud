"""CoReRouter with expert-aware tokens, masks, costs, and abstention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from coregraph.objectives.scores import ScoreType, validate_scores
from coregraph.routing.fallback import FallbackStrategy


@dataclass(frozen=True)
class RouterOutput:
    expert_weights: torch.Tensor
    selected_expert: torch.Tensor
    blended_score: torch.Tensor
    score_type: ScoreType
    abstention_probability: torch.Tensor
    routing_entropy: torch.Tensor
    expected_compute: torch.Tensor
    all_experts_unavailable: torch.Tensor
    explanation_records: tuple[dict[str, Any], ...]


class CoReRouter(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        contract_dim: int,
        diagnostic_dim: int,
        per_expert_diagnostic_dim: int = 0,
        expert_identity_dim: int = 8,
        expert_family_dim: int = 0,
        num_expert_families: int = 0,
        hidden_dim: int = 64,
        mode: str = "mlp",
        fallback: FallbackStrategy = FallbackStrategy.FEATURE_ONLY_SAFE,
        feature_expert_index: int = 0,
        unavailable_score_sentinel: float = 0.0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("router requires at least one expert")
        if min(
            contract_dim,
            diagnostic_dim,
            per_expert_diagnostic_dim,
            expert_identity_dim,
            expert_family_dim,
        ) < 0:
            raise ValueError("router dimensions cannot be negative")
        if mode not in {"linear", "mlp", "attention"}:
            raise ValueError("router mode must be linear mlp or attention")
        if expert_family_dim > 0 and num_expert_families < 1:
            raise ValueError("family embeddings require at least one family")
        self.num_experts = num_experts
        self.contract_dim = contract_dim
        self.diagnostic_dim = diagnostic_dim
        self.per_expert_diagnostic_dim = per_expert_diagnostic_dim
        self.expert_identity_dim = expert_identity_dim
        self.expert_family_dim = expert_family_dim
        self.num_expert_families = num_expert_families
        self.mode = mode
        self.fallback = fallback
        self.feature_expert_index = feature_expert_index
        self.unavailable_score_sentinel = float(unavailable_score_sentinel)
        self.expert_identity = (
            nn.Embedding(num_experts, expert_identity_dim)
            if expert_identity_dim
            else None
        )
        self.expert_family = (
            nn.Embedding(num_expert_families, expert_family_dim)
            if expert_family_dim
            else None
        )
        self.identity_token_offset = 1
        self.family_token_offset = (
            self.identity_token_offset + expert_identity_dim
        )
        self.per_expert_diagnostic_offset = (
            self.family_token_offset + expert_family_dim
        )
        self.contract_token_offset = (
            self.per_expert_diagnostic_offset
            + per_expert_diagnostic_dim
        )
        self.shared_diagnostic_offset = (
            self.contract_token_offset + contract_dim
        )
        self.cost_token_offset = (
            self.shared_diagnostic_offset + diagnostic_dim
        )
        self.availability_token_offset = self.cost_token_offset + 1
        token_dim = self.availability_token_offset + 1
        if mode == "linear":
            self.scorer: nn.Module = nn.Linear(token_dim, 1)
        else:
            self.scorer = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.attention: nn.MultiheadAttention | None
        if mode == "attention":
            self.attention = nn.MultiheadAttention(
                token_dim,
                num_heads=1,
                batch_first=True,
            )
        else:
            self.attention = None
        self.abstention_head = nn.Sequential(
            nn.Linear(contract_dim + diagnostic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _indices(
        self,
        value: torch.Tensor | None,
        *,
        batch: int,
        upper: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            indices = torch.arange(self.num_experts, device=device)
        else:
            indices = value.to(device=device, dtype=torch.long)
        if indices.shape == (self.num_experts,):
            indices = indices.unsqueeze(0).expand(batch, -1)
        if indices.shape != (batch, self.num_experts):
            raise ValueError(f"{name} must have shape [experts] or [batch,experts]")
        if bool((indices < 0).any()) or bool((indices >= upper).any()):
            raise ValueError(f"{name} contains an out-of-range index")
        return indices

    def _cost_matrix(
        self,
        expert_costs: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        costs = expert_costs.to(device=device, dtype=torch.float32)
        if costs.shape == (self.num_experts,):
            costs = costs.unsqueeze(0).expand(batch, -1)
        if costs.shape != (batch, self.num_experts):
            raise ValueError(
                "expert_costs must have shape [experts] or [batch,experts]"
            )
        if not torch.isfinite(costs).all() or bool((costs < 0).any()):
            raise ValueError("expert costs must be finite and non-negative")
        return costs

    def forward(
        self,
        *,
        expert_scores: torch.Tensor,
        score_type: ScoreType,
        contract_embedding: torch.Tensor,
        shared_diagnostics: torch.Tensor,
        per_expert_diagnostics: torch.Tensor,
        availability_mask: torch.Tensor,
        expert_costs: torch.Tensor,
        expert_identity_indices: torch.Tensor | None = None,
        expert_family_indices: torch.Tensor | None = None,
        expert_names: Sequence[str] | None = None,
    ) -> RouterOutput:
        if expert_scores.ndim != 2 or expert_scores.shape[1] != self.num_experts:
            raise ValueError("expert_scores must have shape [batch,num_experts]")
        validate_scores(expert_scores, score_type)
        batch = expert_scores.shape[0]
        if contract_embedding.shape != (batch, self.contract_dim):
            raise ValueError("contract embedding shape mismatch")
        if shared_diagnostics.shape != (batch, self.diagnostic_dim):
            raise ValueError("shared diagnostic shape mismatch")
        if per_expert_diagnostics.shape != (
            batch,
            self.num_experts,
            self.per_expert_diagnostic_dim,
        ):
            raise ValueError("per-expert diagnostic shape mismatch")
        if availability_mask.shape != expert_scores.shape:
            raise ValueError("availability mask shape mismatch")
        mask = availability_mask.bool()
        costs = self._cost_matrix(
            expert_costs,
            batch=batch,
            device=expert_scores.device,
        )
        repeated_contract = contract_embedding[:, None, :].expand(
            -1,
            self.num_experts,
            -1,
        )
        repeated_shared = shared_diagnostics[:, None, :].expand(
            -1,
            self.num_experts,
            -1,
        )
        parts = [expert_scores[:, :, None]]
        if self.expert_identity is not None:
            identity_indices = self._indices(
                expert_identity_indices,
                batch=batch,
                upper=self.num_experts,
                device=expert_scores.device,
                name="expert_identity_indices",
            )
            parts.append(self.expert_identity(identity_indices))
        if self.expert_family is not None:
            if expert_family_indices is None:
                raise ValueError("expert family embeddings require family indices")
            family_indices = self._indices(
                expert_family_indices,
                batch=batch,
                upper=self.num_expert_families,
                device=expert_scores.device,
                name="expert_family_indices",
            )
            parts.append(self.expert_family(family_indices))
        parts.extend(
            [
                per_expert_diagnostics,
                repeated_contract,
                repeated_shared,
                costs[:, :, None],
                mask.float()[:, :, None],
            ]
        )
        tokens = torch.cat(parts, dim=-1)
        no_experts = ~mask.any(dim=-1)
        if self.attention is not None:
            valid_indices = torch.where(~no_experts)[0]
            if valid_indices.numel():
                valid_tokens = tokens.index_select(0, valid_indices)
                valid_mask = mask.index_select(0, valid_indices)
                attended, _ = self.attention(
                    valid_tokens,
                    valid_tokens,
                    valid_tokens,
                    key_padding_mask=~valid_mask,
                    need_weights=False,
                )
                tokens = tokens.index_copy(0, valid_indices, attended)
        routing_logits = self.scorer(tokens).squeeze(-1)
        masked_logits = routing_logits.masked_fill(
            ~mask & ~no_experts[:, None],
            -torch.inf,
        )
        safe_logits = torch.where(
            no_experts[:, None],
            torch.zeros_like(masked_logits),
            masked_logits,
        )
        weights = torch.softmax(safe_logits, dim=-1) * mask.float()
        # Multiplication above makes every unavailable expert exactly zero and
        # every all-unavailable row exactly zero while retaining a valid graph.
        context = torch.cat([contract_embedding, shared_diagnostics], dim=-1)
        abstention_probability = torch.sigmoid(
            self.abstention_head(context)
        ).squeeze(-1)
        abstention_probability = torch.where(
            no_experts,
            torch.ones_like(abstention_probability),
            abstention_probability,
        )
        blended = (weights * expert_scores).sum(dim=-1)
        sentinel = blended.new_full(blended.shape, self.unavailable_score_sentinel)
        blended = torch.where(no_experts, sentinel, blended)
        entropy = -(
            weights * torch.log(weights.clamp_min(1e-12))
        ).sum(dim=-1)
        expected_compute = (weights * costs).sum(dim=-1)
        names = tuple(
            expert_names
            or [f"expert_{index}" for index in range(self.num_experts)]
        )
        if len(names) != self.num_experts:
            raise ValueError("one expert name is required per expert")
        selected = weights.argmax(dim=-1)
        selected = torch.where(
            no_experts,
            torch.full_like(selected, -1),
            selected,
        )
        records = tuple(
            {
                "selected_expert": (
                    None
                    if bool(no_experts[row])
                    else names[int(selected[row].item())]
                ),
                "weights": {
                    names[index]: float(
                        weights[row, index].detach().cpu()
                    )
                    for index in range(self.num_experts)
                },
                "unavailable": [
                    names[index]
                    for index in range(self.num_experts)
                    if not bool(mask[row, index])
                ],
                "all_experts_unavailable": bool(no_experts[row]),
                "score_type": score_type.value,
                "routing_entropy": float(entropy[row].detach().cpu()),
                "expected_compute": float(
                    expected_compute[row].detach().cpu()
                ),
            }
            for row in range(batch)
        )
        return RouterOutput(
            expert_weights=weights,
            selected_expert=selected,
            blended_score=blended,
            score_type=score_type,
            abstention_probability=abstention_probability,
            routing_entropy=entropy,
            expected_compute=expected_compute,
            all_experts_unavailable=no_experts,
            explanation_records=records,
        )
