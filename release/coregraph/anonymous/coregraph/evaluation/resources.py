"""Resource-measurement records; training time cannot impersonate latency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceMeasurement:
    expert_id: str
    parameter_count: int | None
    flops_or_proxy: float | None
    inference_latency_ms: float | None
    routing_latency_ms: float | None
    peak_cpu_memory_mb: float | None
    peak_gpu_memory_mb: float | None
    throughput_per_second: float | None
    expert_invocations: int
    review_budget_used: int
    batch_size: int
    warmup_policy: str
    hardware_identity: str
    software_environment: str
    status: str

    def __post_init__(self) -> None:
        numeric = (
            self.parameter_count,
            self.flops_or_proxy,
            self.inference_latency_ms,
            self.routing_latency_ms,
            self.peak_cpu_memory_mb,
            self.peak_gpu_memory_mb,
            self.throughput_per_second,
            self.expert_invocations,
            self.review_budget_used,
            self.batch_size,
        )
        if any(value is not None and value < 0 for value in numeric):
            raise ValueError("resource measurements cannot be negative")
        if self.batch_size < 1:
            raise ValueError("resource measurement batch size must be positive")
        if self.status == "MEASURED" and self.inference_latency_ms is None:
            raise ValueError("measured status requires inference latency")
