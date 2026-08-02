"""Deterministic sampled node-classification expert for scalable GNN execution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from coregraph.experts.base import Expert, OfficialStatus, ResourceRequirements
from coregraph.experts.sampling import SamplingPlan, deterministic_batches, sample_one_hop
from coregraph.tasks.base import TaskBatch, TaskType
from coregraph.utils.seeding import seed_everything
from models.registry import build_model


def _numeric_identifiers(batch: TaskBatch) -> np.ndarray:
    values = []
    for identifier in batch.identifiers:
        token = str(identifier).rsplit(":", 1)[-1]
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(
                "sampled node expert requires integer node identifiers"
            ) from exc
    return np.asarray(values, dtype=int)


@dataclass
class SampledNodeGraphExpert(Expert):
    model_name: str
    expert_id: str
    hidden_channels: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    sampling: SamplingPlan = field(
        default_factory=lambda: SamplingPlan(batch_size=1024)
    )
    device: str = "cpu"
    official_status: OfficialStatus = OfficialStatus.VALIDATED_REIMPLEMENTATION
    supported_tasks: tuple[TaskType, ...] = (TaskType.NODE_CLASSIFICATION,)
    _model: torch.nn.Module | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("sampled graph expert supports cpu or cuda")

    def _sample_subgraph(
        self,
        batch: TaskBatch,
        seed_nodes: np.ndarray,
        *,
        rng_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        if batch.graph_view is None:
            raise ValueError("sampled graph expert requires a fold-specific graph view")
        view = batch.graph_view
        edges = np.asarray(view.edge_index, dtype=int)
        selected_nodes = set(int(node) for node in seed_nodes)
        frontier = np.asarray(sorted(selected_nodes), dtype=int)
        chosen_edges: set[int] = set()
        for layer, fanout in enumerate(self.sampling.fanouts):
            edge_ids = sample_one_hop(
                edges,
                frontier,
                fanout=fanout,
                rng_seed=self.sampling.seed + rng_offset + layer,
            )
            chosen_edges.update(int(value) for value in edge_ids)
            if len(edge_ids) == 0:
                break
            frontier = np.unique(edges[0, edge_ids])
            selected_nodes.update(int(node) for node in frontier)
        nodes = np.asarray(sorted(selected_nodes), dtype=int)
        if chosen_edges:
            edge_ids_array = np.asarray(sorted(chosen_edges), dtype=int)
            selected_edges = edges[:, edge_ids_array]
        else:
            selected_edges = np.empty((2, 0), dtype=int)
            edge_ids_array = np.asarray([], dtype=int)
        mapping = {int(node): index for index, node in enumerate(nodes)}
        relabelled = np.asarray(
            [
                [mapping[int(source)] for source in selected_edges[0]],
                [mapping[int(target)] for target in selected_edges[1]],
            ],
            dtype=int,
        )
        row_by_node = {
            int(node): index for index, node in enumerate(_numeric_identifiers(batch))
        }
        try:
            feature_rows = np.asarray([row_by_node[int(node)] for node in nodes], dtype=int)
            seed_positions = np.asarray([mapping[int(node)] for node in seed_nodes], dtype=int)
        except KeyError as exc:
            raise ValueError("graph-view node is absent from task batch") from exc
        edge_attributes = (
            np.asarray(view.edge_attributes)[edge_ids_array]
            if view.edge_attributes is not None and len(edge_ids_array)
            else None
        )
        return (
            torch.tensor(batch.features[feature_rows], dtype=torch.float32, device=self.device),
            torch.tensor(relabelled, dtype=torch.long, device=self.device),
            (
                torch.tensor(edge_attributes, dtype=torch.float32, device=self.device)
                if edge_attributes is not None
                else torch.empty((0, 0), dtype=torch.float32, device=self.device)
            ),
            seed_positions,
        )

    def fit(self, batch: TaskBatch) -> "SampledNodeGraphExpert":
        if batch.graph_view is None or batch.graph_view.role.value != "train":
            raise ValueError("fit requires the explicit training GraphView")
        seed_everything(self.sampling.seed)
        built = build_model(
            self.model_name,
            in_channels=batch.features.shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=3,
        )
        if built.category == "temporal":
            raise ValueError("use the official event-temporal adapter for temporal models")
        self._model = built.model.to(self.device)
        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        global_ids = _numeric_identifiers(batch)
        visible = np.isin(global_ids, batch.graph_view.visible_node_ids)
        train_nodes = global_ids[batch.train_mask & batch.label_mask & visible]
        if len(train_nodes) == 0:
            raise ValueError("training graph view contains no labelled train nodes")
        row_by_node = {int(node): index for index, node in enumerate(global_ids)}
        self._model.train()
        optimiser.zero_grad()
        accumulation = 0
        for epoch in range(self.epochs):
            for batch_index, seeds in enumerate(
                deterministic_batches(
                    train_nodes,
                    batch_size=self.sampling.batch_size,
                    seed=self.sampling.seed + epoch,
                    shuffle=True,
                )
            ):
                x, edge_index, edge_attr, seed_positions = self._sample_subgraph(
                    batch,
                    seeds,
                    rng_offset=epoch * 100_000 + batch_index * 100,
                )
                logits = self._model(
                    x,
                    edge_index,
                    edge_attr if edge_attr.numel() else None,
                )
                label_rows = [row_by_node[int(node)] for node in seeds]
                labels = torch.tensor(
                    batch.labels[label_rows],
                    dtype=torch.long,
                    device=self.device,
                )
                loss = torch.nn.functional.cross_entropy(
                    logits[torch.tensor(seed_positions, device=self.device)],
                    labels,
                )
                (loss / self.sampling.gradient_accumulation).backward()
                accumulation += 1
                if accumulation % self.sampling.gradient_accumulation == 0:
                    optimiser.step()
                    optimiser.zero_grad()
        if accumulation % self.sampling.gradient_accumulation:
            optimiser.step()
            optimiser.zero_grad()
        return self

    def predict_scores(self, batch: TaskBatch) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit sampled graph expert before prediction")
        if batch.graph_view is None:
            raise ValueError("prediction requires an explicit graph view")
        global_ids = _numeric_identifiers(batch)
        visible = np.isin(global_ids, batch.graph_view.visible_node_ids)
        scores = np.full(len(batch.identifiers), np.nan, dtype=float)
        row_by_node = {int(node): index for index, node in enumerate(global_ids)}
        self._model.eval()
        with torch.no_grad():
            for batch_index, seeds in enumerate(
                deterministic_batches(
                    global_ids[visible],
                    batch_size=self.sampling.batch_size,
                    seed=self.sampling.seed,
                    shuffle=False,
                )
            ):
                x, edge_index, edge_attr, seed_positions = self._sample_subgraph(
                    batch,
                    seeds,
                    rng_offset=10_000_000 + batch_index * 100,
                )
                logits = self._model(
                    x,
                    edge_index,
                    edge_attr if edge_attr.numel() else None,
                )
                probabilities = torch.softmax(logits[seed_positions], dim=-1)[:, 1]
                for node, score in zip(seeds, probabilities.cpu().numpy(), strict=True):
                    scores[row_by_node[int(node)]] = float(score)
        return scores

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            min_memory_gb=4.0,
            expected_latency_ms=50.0,
            device_classes=("cpu", "single_t4", "dual_t4"),
            requires_graph=True,
            cost_provenance="DRY_RUN_ESTIMATE",
        )
