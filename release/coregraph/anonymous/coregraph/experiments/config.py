"""Lightweight validated configuration equivalent to a Hydra composition."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

from coregraph.contracts.axes import AccessRegime
from coregraph.contracts.serialization import to_primitive

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class HardwareConfig:
    device: str = "cpu"
    memory_cap_gb: float = 8.0
    mixed_precision: bool = False
    gradient_accumulation: int = 1
    workers: int = 0

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "single_t4", "dual_t4", "cuda", "mps"}:
            raise ValueError(f"unsupported hardware device {self.device!r}")
        if self.memory_cap_gb <= 0 or self.gradient_accumulation < 1 or self.workers < 0:
            raise ValueError("invalid hardware resource values")


@dataclass(frozen=True)
class OutputConfig:
    schema: str = "coregraph_result_v2"
    prediction_export: bool = True
    telemetry: bool = True
    output_root: str = "results/coregraph_pilot"


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    task: str
    source_contracts: Tuple[str, ...]
    target_contracts: Tuple[str, ...]
    access_regime: AccessRegime
    experts: Tuple[str, ...]
    router: str
    objective: str
    metrics: Tuple[str, ...]
    seed: int
    hardware: HardwareConfig
    data_checksum: str
    code_commit: str
    dependency_lock: str
    dataset_manifest: str
    output: OutputConfig = field(default_factory=OutputConfig)
    smoke: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        for name in ("dataset", "task", "router", "objective", "data_checksum", "code_commit"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"run config field {name} cannot be empty")
        if not self.source_contracts or not self.target_contracts:
            raise ValueError("run config requires source and target contracts")
        if set(self.source_contracts) & set(self.target_contracts):
            raise ValueError("source and target contract sets must be disjoint")
        if not self.experts or not self.metrics:
            raise ValueError("run config requires experts and metrics")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.dry_run:
            if not _GIT_COMMIT.fullmatch(self.code_commit):
                raise ValueError("executable run config requires an exact 40-hex code commit")
            for name in ("data_checksum", "dependency_lock", "dataset_manifest"):
                if not _SHA256.fullmatch(str(getattr(self, name))):
                    raise ValueError(
                        f"executable run config requires a SHA-256 value for {name}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        hardware = payload.get("hardware", {})
        output = payload.get("output", {})
        return cls(
            dataset=str(payload["dataset"]),
            task=str(payload["task"]),
            source_contracts=tuple(payload["source_contracts"]),
            target_contracts=tuple(payload["target_contracts"]),
            access_regime=AccessRegime(payload["access_regime"]),
            experts=tuple(payload["experts"]),
            router=str(payload["router"]),
            objective=str(payload["objective"]),
            metrics=tuple(payload["metrics"]),
            seed=int(payload["seed"]),
            hardware=HardwareConfig(**hardware),
            data_checksum=str(payload["data_checksum"]),
            code_commit=str(payload["code_commit"]),
            dependency_lock=str(payload["dependency_lock"]),
            dataset_manifest=str(payload["dataset_manifest"]),
            output=OutputConfig(**output),
            smoke=bool(payload.get("smoke", False)),
            dry_run=bool(payload.get("dry_run", False)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("run configuration YAML must decode to a mapping")
        return cls.from_dict(payload)
