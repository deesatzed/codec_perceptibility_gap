"""Measured cost record + honesty manifest (Stage B).

Tokens, wall-clock, and active-parameter-seconds are MEASURED on real hardware
with no privilege. Joules are NEVER tagged 'measured' here — they are
'unavailable' unless the sudo-gated energy probe supplies a real value, or
'derived' if estimated from active-param-seconds via a published constant.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.dern_b.mlx_backend import GenResult


def cost_record(gen: GenResult, overhead_units: float = 0.0,
                joules: Optional[float] = None, joules_source: str = "unavailable") -> Dict[str, Any]:
    assert joules_source in {"measured", "derived", "unavailable", "simulated"}
    rec = {
        "total_tokens": gen.prompt_tokens + gen.gen_tokens,
        "gen_tokens": gen.gen_tokens,
        "wall_seconds": round(gen.wall_seconds, 4),
        "active_param_seconds": round(gen.active_param_seconds, 1),
        "overhead": float(overhead_units),
        "joules": joules,
        "_tags": {
            "total_tokens": "measured",
            "gen_tokens": "measured",
            "wall_seconds": "measured",
            "active_param_seconds": "measured",
            "overhead": "measured",
            "joules": joules_source,  # never 'measured' unless probe supplied it
        },
    }
    return rec


def net_savings(baseline: Dict[str, Any], chosen: Dict[str, Any], dim: str) -> float:
    """Net savings on a single measured dimension, overhead charged to chosen side.

    overhead is in active-param-second-equivalent units only when dim is
    'active_param_seconds'; for token/latency dims it is treated as 0 (the probe
    is pure-Python and adds no tokens/model-latency).
    """
    oh = chosen["overhead"] if dim == "active_param_seconds" else 0.0
    return round(baseline[dim] - (chosen[dim] + oh), 4)
