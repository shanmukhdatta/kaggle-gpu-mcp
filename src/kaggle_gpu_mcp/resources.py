"""Deterministic resource selection; never silently run GPU-required jobs on CPU."""
from typing import Literal

Mode = Literal["cpu", "gpu", "auto"]


def choose(mode: Mode, workload: str = "pytorch", estimated_rows: int = 0) -> dict:
    if mode not in ("cpu", "gpu", "auto"):
        raise ValueError("mode must be cpu, gpu or auto")
    if mode == "auto":
        use_gpu = workload.lower() in {"pytorch", "tensorflow", "transformer", "deep-learning"}
        if workload.lower() == "debug" or (workload.lower() == "numpy" and estimated_rows < 500_000):
            use_gpu = False
        selected = "gpu" if use_gpu else "cpu"
    else:
        selected = mode
    return {
        "requested": mode,
        "selected": selected,
        "accelerator": "NvidiaTeslaT4" if selected == "gpu" else None,
        "reason": "Heuristic selection; provider quota and actual device must be verified at runtime",
    }
