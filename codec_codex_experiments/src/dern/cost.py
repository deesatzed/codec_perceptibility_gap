"""Simulated broad cost vector + honest net-savings accounting (Stage A).

Every dimension is tagged 'simulated' in Stage A — no real energy is measured.
The tags are the honesty manifest: Stage B replaces specific tags with
'measured'/'telemetry-derived'; nothing here may ever claim 'measured'.
"""
from __future__ import annotations

from typing import Any, Dict

from src.dern.configs import compute_cost_units

ALL_SIMULATED = "simulated"

# Posture -> thermal multiplier. 'low_fan' models the "runs cool, don't spin up"
# experiential insight: lower fan/clock posture => lower thermal/operational cost.
_POSTURE_THERMAL = {"default": 1.0, "low_fan": 0.4, "race_to_idle": 0.7}


def cost_vector(config: Dict[str, Any], posture: str, overhead_units: float) -> Dict[str, Any]:
    compute = compute_cost_units(config)
    thermal = compute * _POSTURE_THERMAL.get(posture, 1.0)
    dims = {
        "compute": compute,
        "thermal_slope": round(thermal, 4),
        "dvfs": 1.0 if posture == "default" else 0.6,
        "idle": 0.0 if posture == "race_to_idle" else 0.2,
        "carbon": compute * 0.1,
        "overhead": float(overhead_units),
    }
    dims["_tags"] = {k: ALL_SIMULATED for k in dims}
    return dims


def net_savings(baseline: Dict[str, Any], chosen: Dict[str, Any]) -> float:
    """Net = baseline (compute+thermal) - chosen (compute+thermal+overhead).

    Overhead (probe+controller+verifier) is charged to the CHOSEN side, never
    hidden — a route that costs more to decide than it saves nets negative.
    """
    base = baseline["compute"] + baseline["thermal_slope"]
    chos = chosen["compute"] + chosen["thermal_slope"] + chosen["overhead"]
    return round(base - chos, 4)
