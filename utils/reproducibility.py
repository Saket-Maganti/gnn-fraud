"""
utils/reproducibility.py

Global determinism and provenance helpers.

The RUNS harnesses previously seeded ``random`` / ``numpy`` / ``torch`` ad hoc
(see the old ``set_seed`` in ``experiments/_multi_harness.py`` and the inline
seeding in ``scripts/runs_harness_common.py``), but set **none** of the flags
that actually make a run bit-reproducible: cuDNN determinism, the global
deterministic-algorithms switch, ``PYTHONHASHSEED``, or the CUBLAS workspace
config required for deterministic CUDA matmuls.

``set_global_determinism`` is the single entry point every runner should call
once, at the very top, before any model or data is constructed. It returns a
seeded ``torch.Generator`` that callers can hand to DataLoaders / samplers so
shuffling is reproducible too.

Nothing here changes metric values or training logic — it only removes
nondeterminism, which is exactly what an ACM artifact-evaluation reviewer wants
to see. Determinism is requested in ``warn_only`` mode by default so that an op
without a deterministic kernel on a given backend (e.g. some scatter/segment
reductions on MPS) degrades to a warning rather than crashing a long sweep.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Mapping, Optional

import numpy as np


def set_global_determinism(
    seed: int,
    *,
    deterministic_torch: bool = True,
    warn_only: bool = True,
    cudnn_benchmark: bool = False,
) -> "Any":
    """Seed every RNG and request deterministic kernels.

    Call once per process before constructing models or loading data.

    Parameters
    ----------
    seed
        Master seed applied to ``random``, ``numpy`` and ``torch`` (CPU + all
        CUDA devices).
    deterministic_torch
        If ``True`` (default) call ``torch.use_deterministic_algorithms`` and
        set cuDNN to deterministic mode.
    warn_only
        Passed through to ``torch.use_deterministic_algorithms``; when ``True``
        an op lacking a deterministic implementation warns instead of raising.
        Keep ``True`` for sweeps, flip to ``False`` to *prove* full determinism.
    cudnn_benchmark
        Leave ``False`` for determinism. ``True`` lets cuDNN autotune (faster,
        nondeterministic) and is only useful for throughput benchmarking.

    Returns
    -------
    torch.Generator
        A CPU generator seeded from ``seed`` for use with DataLoaders/samplers.
    """
    # Process-level hash seed must be set before interpreter hashing matters;
    # exported so re-exec'd subprocesses inherit it too.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CUDA GEMMs (cublas) under deterministic mode.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    import torch  # local import keeps this module importable without torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic_torch)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            # Older torch without the warn_only kwarg.
            torch.use_deterministic_algorithms(True)
        except Exception:  # noqa: BLE001 - never let determinism setup crash a run
            pass

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    """Stable short hash of a run configuration for provenance binding.

    Produces the same digest for the same logical config regardless of key
    order. Non-JSON-serialisable values fall back to ``repr``. Attach the result
    to every result row so an artifact can be traced to the exact config that
    produced it.
    """
    payload = json.dumps(dict(config), sort_keys=True, default=repr)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def determinism_report() -> dict:
    """Snapshot of the current determinism-relevant environment + torch flags.

    Safe to embed in run metadata; returns plain JSON-serialisable values and
    never raises even if torch is unavailable.
    """
    report: dict = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        try:
            report["deterministic_algorithms"] = (
                torch.are_deterministic_algorithms_enabled()
            )
        except Exception:  # noqa: BLE001
            report["deterministic_algorithms"] = None
        if hasattr(torch.backends, "cudnn"):
            report["cudnn_deterministic"] = bool(
                torch.backends.cudnn.deterministic
            )
            report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:  # noqa: BLE001
        report["torch_version"] = None
    return report
