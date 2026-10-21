"""Expert registry that never promotes diagnostic approximations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from coregraph.experts.base import Expert, OfficialStatus
from coregraph.experts.feature_experts import FeatureMLPExpert, LogisticRegressionExpert
from coregraph.experts.graph_experts import LegacyGraphExpert
from coregraph.experts.sampled_graph_expert import SampledNodeGraphExpert
from coregraph.experts.tree_experts import HistGBExpert, RandomForestExpert

Factory = Callable[..., Expert]


@dataclass
class ExpertRegistry:
    factories: dict[str, Factory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.factories:
            self.factories.update(
                {
                    "feature_mlp": lambda **kwargs: FeatureMLPExpert(**kwargs),
                    "logistic_regression": lambda **kwargs: LogisticRegressionExpert(**kwargs),
                    "histgb": lambda **kwargs: HistGBExpert(**kwargs),
                    "random_forest": lambda **kwargs: RandomForestExpert(**kwargs),
                    "gcn": lambda **kwargs: SampledNodeGraphExpert(
                        model_name="gcn", expert_id="gcn_sampled", **kwargs
                    ),
                    "graphsage": lambda **kwargs: SampledNodeGraphExpert(
                        model_name="sage", expert_id="graphsage_sampled", **kwargs
                    ),
                    "gin": lambda **kwargs: LegacyGraphExpert(
                        model_name="gin", expert_id="gin", **kwargs
                    ),
                    "gine": lambda **kwargs: LegacyGraphExpert(
                        model_name="gine", expert_id="gine", **kwargs
                    ),
                    "recent_window_sage": lambda **kwargs: SampledNodeGraphExpert(
                        model_name="sage",
                        expert_id="recent_window_sage",
                        **kwargs,
                    ),
                }
            )

    def register(self, name: str, factory: Factory) -> None:
        if name in self.factories:
            raise ValueError(f"expert {name!r} already registered")
        self.factories[name] = factory

    def build(self, name: str, **kwargs: object) -> Expert:
        if name not in self.factories:
            raise KeyError(f"unknown expert {name!r}; available={sorted(self.factories)}")
        expert = self.factories[name](**kwargs)
        if (
            expert.official_status is OfficialStatus.OFFICIAL_CODE
            and "Legacy" in type(expert).__name__
        ):
            raise ValueError("legacy wrapper cannot be promoted to OFFICIAL_CODE")
        return expert
