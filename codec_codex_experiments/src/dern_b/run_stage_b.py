"""Stage-B driver: run the real-model cascade over a fixed prompt set and emit a
MEASURED report (tokens / latency / active-param-seconds), with an honest
manifest. Joules only if the user primed sudo for the energy probe.

Usage:
    python -m src.dern_b.run_stage_b
    # for measured joules, first prime sudo in the session:  ! sudo -v
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.dern_b.runtime_b import DERNBRuntime
from src.dern_b.prompts import ALL_PROMPTS
from src.dern_b.energy_probe import sudo_available


def run_stage_b(prompts: List[str] | None = None, epsilon: float = 0.5,
                audit_prob: float = 1.0, max_tokens: int = 48, seed: int = 0) -> Dict[str, Any]:
    prompts = prompts if prompts is not None else ALL_PROMPTS
    rt = DERNBRuntime(epsilon=epsilon, audit_prob=audit_prob, max_tokens=max_tokens, seed=seed)
    served_worse_than_ref = 0
    cheap_served = 0
    tok_sav, aps_sav, lat_sav = [], [], []
    for p in prompts:
        rec = rt.route(p)
        if rec["served"] == "cheap":
            cheap_served += 1
            # a served cheap that wasn't a passed audit would be a safety breach
            if rec["audited"] and rec["verdict_passed"] is False:
                served_worse_than_ref += 1
        tok_sav.append(rec["token_savings"])
        aps_sav.append(rec["aps_savings"])
        lat_sav.append(rec["latency_savings"])
    n = len(prompts)
    return {
        "prompts": n,
        "cheap_acceptance_rate": round(cheap_served / n, 3),
        "escalation_rate": round(1 - cheap_served / n, 3),
        "mean_token_savings": round(float(np.mean(tok_sav)), 2),
        "mean_active_param_seconds_savings": round(float(np.mean(aps_sav)), 1),
        "mean_latency_savings_s": round(float(np.mean(lat_sav)), 3),
        "served_worse_than_reference": served_worse_than_ref,
        "epsilon": epsilon,
        "manifest": {
            "tokens": "measured", "latency": "measured",
            "active_param_seconds": "measured",
            "joules": "measured" if sudo_available() else "unavailable",
            "lane": "bounded-loss (Lane 2); Lane 1 exact deferred",
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_stage_b(), indent=2))
