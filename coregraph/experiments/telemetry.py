"""Lightweight CPU/GPU resource telemetry."""

from __future__ import annotations

import platform
import resource
import time
from dataclasses import dataclass


@dataclass
class Telemetry:
    started: float = 0.0
    completed: float = 0.0
    max_rss_kb: int = 0
    platform: str = ""

    def __enter__(self) -> "Telemetry":
        self.started = time.perf_counter()
        self.platform = platform.platform()
        return self

    def __exit__(self, *args: object) -> None:
        self.completed = time.perf_counter()
        self.max_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "wall_seconds": max(0.0, self.completed - self.started),
            "max_rss_kb": self.max_rss_kb,
            "platform": self.platform,
            "measurement": "MEASURED",
        }
