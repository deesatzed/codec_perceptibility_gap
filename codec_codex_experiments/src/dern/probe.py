"""Distinction probe: cheap, deterministic, non-semantic key for a trajectory.

The key is a tuple of small integers (quantized coarse statistics). It carries
NO human-text form — it is an opaque region label the experience graph keys on.
Cost is reported in abstract 'units' (Stage A has no real energy; the actuator
converts units -> simulated cost vector).
"""
from __future__ import annotations

import numpy as np

# Quantization bins per statistic; coarse so similar trajectories share a key.
_BINS = 4


def _quantize(value: float, lo: float, hi: float, bins: int = _BINS) -> int:
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    return int(np.clip(int(frac * bins), 0, bins - 1))


def distinction_key(traj: np.ndarray) -> tuple:
    """traj: (T, N) single-trajectory array -> structured int key.

    Uses cheap shape statistics: per-step energy spread, late-vs-early variance
    ratio (a coarse 'is it settling or chaotic' signal), and mean abs level.
    """
    x = np.asarray(traj, dtype=float)
    half = x.shape[0] // 2 or 1
    early_var = float(x[:half].var())
    late_var = float(x[half:].var())
    spread = float(x.std())
    level = float(np.abs(x).mean())
    ratio = late_var / (early_var + 1e-8)
    return (
        _quantize(spread, 0.0, 6.0),
        _quantize(ratio, 0.0, 3.0),
        _quantize(level, 0.0, 4.0),
    )


def probe_cost_units() -> float:
    """Abstract cost of running the probe (small but nonzero; metered as overhead)."""
    return 0.05
