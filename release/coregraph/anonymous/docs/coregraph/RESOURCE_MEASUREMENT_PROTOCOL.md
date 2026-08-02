# Resource measurement protocol

Status: `PROTOCOL_FROZEN_MEASUREMENTS_PENDING`.

Every future resource record includes parameter count, FLOPs or a named justified proxy, expert inference latency, router latency, peak CPU and GPU memory, throughput, invocation count, review usage, batch size, warmup policy, hardware identity, software environment, code SHA, config hash, and measurement status.

Latency is measured around inference only after declared warmup and synchronization. End-to-end training runtime, notebook wall time, archive import time, and data-download time cannot substitute for inference latency. Router latency is reported separately and together with the invoked expert path. Peak memory must name the process boundary and measurement tool. Estimates and measurements are never pooled without a status column.

Required conditions are: all experts available; one graph expert unavailable; all graph experts unavailable; tight memory; tight latency; tight review budget; combined graph/resource shift; and dynamic batch-level availability. Resource constraints form the feasible mask before routing. An OOM becomes `RESOURCE_BLOCKED` with the attempted batch and envelope; it is not a zero metric or silently skipped run.

Measurements are repeated under a fixed batch size and software environment. Warmup count, timed iterations, synchronization, and estimator uncertainty are recorded in the run manifest. Numeric hardware claims remain `BLOCKED_RESOURCE_UNMEASURED` until this protocol produces validated records.
