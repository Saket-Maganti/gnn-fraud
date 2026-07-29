"""Fold-specific graph visibility with machine-checkable provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from coregraph.contracts.axes import ConstructionAxis, VisibilityAxis
from coregraph.contracts.contract import DeploymentContract


class ViewRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TARGET = "target"


@dataclass(frozen=True)
class GraphView:
    visible_node_ids: np.ndarray
    edge_index: np.ndarray
    directed: bool
    edge_attributes: Optional[np.ndarray]
    edge_timestamps: Optional[np.ndarray]
    time_cutoff: Optional[float]
    source_mask: np.ndarray
    target_mask: np.ndarray
    construction: ConstructionAxis
    contract: DeploymentContract
    provenance: Tuple[Tuple[str, str], ...]
    role: ViewRole

    def __post_init__(self) -> None:
        nodes = np.asarray(self.visible_node_ids)
        edges = np.asarray(self.edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if len(self.source_mask) != len(nodes) or len(self.target_mask) != len(nodes):
            raise ValueError("view masks must align to visible nodes")
        if np.any(self.source_mask & self.target_mask):
            raise ValueError("source and target masks cannot overlap")
        if self.edge_attributes is not None and len(self.edge_attributes) != edges.shape[1]:
            raise ValueError("edge attributes must align to edges")
        if self.edge_timestamps is not None and len(self.edge_timestamps) != edges.shape[1]:
            raise ValueError("edge timestamps must align to edges")
        if edges.size:
            visible = set(nodes.tolist())
            if any(int(v) not in visible for v in edges.reshape(-1)):
                raise ValueError("edge endpoint is not visible in this graph view")
        if self.time_cutoff is not None and self.edge_timestamps is not None:
            if np.any(self.edge_timestamps > self.time_cutoff):
                raise ValueError("graph view contains a future edge beyond its cutoff")

    @property
    def node_set(self) -> set[int]:
        return set(int(v) for v in self.visible_node_ids)

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class GraphViewBundle:
    train: GraphView
    validation: GraphView
    target: GraphView


def _filter_edges(
    edge_index: np.ndarray,
    visible_nodes: np.ndarray,
    edge_timestamps: Optional[np.ndarray],
    cutoff: Optional[float],
    *,
    recent_window: Optional[int] = None,
) -> np.ndarray:
    visible_mask = np.zeros(int(max(edge_index.max(initial=-1), visible_nodes.max(initial=-1))) + 1, dtype=bool)
    visible_mask[visible_nodes.astype(int)] = True
    keep = visible_mask[edge_index[0]] & visible_mask[edge_index[1]]
    if edge_timestamps is not None and cutoff is not None:
        keep &= edge_timestamps <= cutoff
        if recent_window is not None:
            keep &= edge_timestamps > cutoff - recent_window
    return keep


def _view(
    *,
    role: ViewRole,
    node_ids: np.ndarray,
    node_timestamps: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamps: Optional[np.ndarray],
    edge_attributes: Optional[np.ndarray],
    source_limit: float,
    target_start: float,
    cutoff: float,
    contract: DeploymentContract,
    allow_all_nodes: bool,
) -> GraphView:
    if contract.visibility in {
        VisibilityAxis.MISSING_GRAPH,
        VisibilityAxis.ISOLATED_INDUCTIVE,
    }:
        visible_nodes = node_ids[node_timestamps <= cutoff]
        keep = np.zeros(edge_index.shape[1], dtype=bool)
    else:
        visible_nodes = node_ids if allow_all_nodes else node_ids[node_timestamps <= cutoff]
        recent = (
            contract.construction.recent_window
            if contract.construction.mode is ConstructionAxis.RECENT_WINDOW
            else None
        )
        keep = _filter_edges(
            edge_index,
            visible_nodes,
            edge_timestamps,
            None if allow_all_nodes else cutoff,
            recent_window=recent,
        )
    visible_times = node_timestamps[np.isin(node_ids, visible_nodes)]
    source_mask = visible_times <= source_limit
    target_mask = visible_times >= target_start
    directed = contract.construction.mode is ConstructionAxis.DIRECTED
    return GraphView(
        visible_node_ids=visible_nodes.copy(),
        edge_index=edge_index[:, keep].copy(),
        directed=directed,
        edge_attributes=(
            edge_attributes[keep].copy() if edge_attributes is not None else None
        ),
        edge_timestamps=(
            edge_timestamps[keep].copy() if edge_timestamps is not None else None
        ),
        time_cutoff=None if allow_all_nodes else float(cutoff),
        source_mask=source_mask.astype(bool),
        target_mask=target_mask.astype(bool),
        construction=contract.construction.mode,
        contract=contract,
        provenance=(
            ("builder", "coregraph.data.graph_views.build_temporal_graph_views"),
            ("future_structure", "permitted" if allow_all_nodes else "forbidden"),
            ("labels_used", "none"),
        ),
        role=role,
    )


def build_temporal_graph_views(
    *,
    node_ids: np.ndarray,
    node_timestamps: np.ndarray,
    edge_index: np.ndarray,
    edge_timestamps: Optional[np.ndarray],
    edge_attributes: Optional[np.ndarray],
    train_cutoff: float,
    validation_cutoff: float,
    target_cutoff: float,
    contract: DeploymentContract,
) -> GraphViewBundle:
    """Construct train/validation/target views under an explicit contract."""

    node_ids = np.asarray(node_ids, dtype=int)
    node_timestamps = np.asarray(node_timestamps)
    edge_index = np.asarray(edge_index, dtype=int)
    if len(node_ids) != len(node_timestamps):
        raise ValueError("node ids and timestamps must align")
    if not train_cutoff < validation_cutoff < target_cutoff:
        raise ValueError("temporal cutoffs must satisfy train < validation < target")
    if edge_timestamps is not None:
        edge_timestamps = np.asarray(edge_timestamps)
    if edge_attributes is not None:
        edge_attributes = np.asarray(edge_attributes)

    transductive = contract.visibility is VisibilityAxis.TRANSDUCTIVE_STRUCTURE
    train = _view(
        role=ViewRole.TRAIN,
        node_ids=node_ids,
        node_timestamps=node_timestamps,
        edge_index=edge_index,
        edge_timestamps=edge_timestamps,
        edge_attributes=edge_attributes,
        source_limit=train_cutoff,
        target_start=float(np.nextafter(float(train_cutoff), np.inf)),
        cutoff=train_cutoff,
        contract=contract,
        allow_all_nodes=transductive,
    )
    validation = _view(
        role=ViewRole.VALIDATION,
        node_ids=node_ids,
        node_timestamps=node_timestamps,
        edge_index=edge_index,
        edge_timestamps=edge_timestamps,
        edge_attributes=edge_attributes,
        source_limit=train_cutoff,
        target_start=float(np.nextafter(float(train_cutoff), np.inf)),
        cutoff=validation_cutoff,
        contract=contract,
        allow_all_nodes=transductive,
    )
    target = _view(
        role=ViewRole.TARGET,
        node_ids=node_ids,
        node_timestamps=node_timestamps,
        edge_index=edge_index,
        edge_timestamps=edge_timestamps,
        edge_attributes=edge_attributes,
        source_limit=validation_cutoff,
        target_start=float(np.nextafter(float(validation_cutoff), np.inf)),
        cutoff=target_cutoff,
        contract=contract,
        allow_all_nodes=transductive
        or contract.visibility is VisibilityAxis.TEST_TIME_GRAPH_AVAILABLE,
    )
    return GraphViewBundle(train=train, validation=validation, target=target)


def audit_view_bundle(
    bundle: GraphViewBundle,
    *,
    test_node_ids: np.ndarray,
    train_cutoff: float,
    validation_cutoff: float,
) -> list[str]:
    violations: list[str] = []
    if bundle.train.edge_timestamps is not None and np.any(
        bundle.train.edge_timestamps > train_cutoff
    ):
        violations.append("future_edges_in_train")
    transductive = (
        bundle.validation.contract.visibility is VisibilityAxis.TRANSDUCTIVE_STRUCTURE
    )
    if not transductive and bundle.validation.node_set.intersection(
        int(v) for v in test_node_ids
    ):
        violations.append("test_nodes_in_validation_graph")
    if bundle.validation.edge_timestamps is not None and not transductive:
        if np.any(bundle.validation.edge_timestamps > validation_cutoff):
            violations.append("test_period_edges_in_validation_graph")
    return violations
