"""Compute configurations DERN can route to, cheapest -> most expensive.

A config is a dict describing how much 'machine' to run. On synthetic agents we
model 'how much compute' as feature richness + quantization granularity, which
is a faithful proxy for distinction-resolving power (proof-ladder S1/S6 showed
fewer distinctions => higher error). Each config has an abstract compute cost.
"""
from __future__ import annotations

from typing import Any, Dict, List

# channel: which feature builder; k: quantization levels (more = finer distinctions)
COMPUTE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "cheap", "channel": "native", "k": 3, "cost": 1.0},
    {"name": "mid", "channel": "engineered", "k": 6, "cost": 2.5},
    {"name": "full", "channel": "direct", "k": 16, "cost": 5.0},
]


def full_config() -> Dict[str, Any]:
    return COMPUTE_CONFIGS[-1]


def compute_cost_units(config: Dict[str, Any]) -> float:
    return float(config["cost"])
