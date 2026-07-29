"""Protocol contract definitions for FraudShiftBench."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolContract:
    name: str
    train_time_access: str
    validation_time_access: str
    test_time_access: str
    graph_access_at_train: str
    graph_access_at_inference: str
    temporal_ordering: str
    label_access: str
    feature_scaling_mode: str
    checkpoint_selection_rule: str
    permitted_claims: list[str] = field(default_factory=list)
    prohibited_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProtocolContract":
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ProtocolContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_protocol_contracts() -> list[ProtocolContract]:
    return [
        ProtocolContract(
            name="transductive_static",
            train_time_access="training labels only",
            validation_time_access="validation labels only",
            test_time_access="test labels forbidden",
            graph_access_at_train="full static graph may be visible",
            graph_access_at_inference="full static graph may be visible",
            temporal_ordering="not deployment-faithful unless explicitly temporal",
            label_access="unknown labels masked; future test labels forbidden",
            feature_scaling_mode="must disclose whether fit on train-only or full population",
            checkpoint_selection_rule="validation-only checkpoint selection",
            permitted_claims=["static benchmark comparison", "transductive sensitivity"],
            prohibited_claims=["deployment-time inductive performance", "future-label access"],
        ),
        ProtocolContract(
            name="strict_inductive_temporal",
            train_time_access="past train window only",
            validation_time_access="validation window after train",
            test_time_access="future test window only at inference",
            graph_access_at_train="train-period graph only",
            graph_access_at_inference="test-period expansion without future labels",
            temporal_ordering="chronological",
            label_access="future labels forbidden",
            feature_scaling_mode="train-only scaling",
            checkpoint_selection_rule="validation-only checkpoint selection",
            permitted_claims=["temporal deployment proxy", "protocol-dependent ranking"],
            prohibited_claims=["oracle transductive access", "test-period model selection"],
        ),
        ProtocolContract(
            name="rolling_deployment",
            train_time_access="rolling historical windows",
            validation_time_access="previous held-out window",
            test_time_access="next production-like block",
            graph_access_at_train="historical graph only",
            graph_access_at_inference="incremental graph expansion",
            temporal_ordering="rolling chronological",
            label_access="labels available only after delay",
            feature_scaling_mode="per-window train-only scaling",
            checkpoint_selection_rule="predeclared validation window",
            permitted_claims=["deployment-contract robustness", "review-budget behavior"],
            prohibited_claims=["single static leaderboard finality"],
        ),
        ProtocolContract(
            name="isolated_feature_control",
            train_time_access="same train labels as paired graph protocol",
            validation_time_access="same validation labels as paired graph protocol",
            test_time_access="same test labels as paired graph protocol",
            graph_access_at_train="no graph structure",
            graph_access_at_inference="no graph structure",
            temporal_ordering="matched to paired protocol",
            label_access="matched to paired protocol",
            feature_scaling_mode="matched train-only scaler",
            checkpoint_selection_rule="matched validation-only selection",
            permitted_claims=["graph utility comparison under matched access"],
            prohibited_claims=["causal graph harm without intervention evidence"],
        ),
    ]
