"""Stage-A driver + acceptance report (no energy claim; simulated cost only)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime


def run_stage_a(family: str, n_requests: int, cfg: Dict[str, Any],
                eps_tolerance: float = 0.4, seed: int = 0) -> Dict[str, Any]:
    src = SOURCE_REGISTRY[family]
    rt = DERNRuntime(epsilon=0.15, audit_prob=1.0, eps_tolerance=eps_tolerance, seed=seed)
    nets, unverified, exact_violations = [], 0, 0
    for s in range(n_requests):
        rec = rt.route(src, cfg, seed=s)
        nets.append(rec["net_savings"])
        if not rec["verified"]:
            unverified += 1
        if rec["proof"] == "exact" and rec["delta"] != 0.0:
            exact_violations += 1
    return {
        "family": family,
        "requests": n_requests,
        "mean_net_savings": round(float(np.mean(nets)), 4),
        "unverified_serves": unverified,
        "lane1_exactness_violations": exact_violations,
        "ledger_size": len(rt.ledger),
        "all_cost_dims_simulated": all(
            t == "simulated" for t in rt.ledger[-1]["cost_vector"]["_tags"].values()
        ) if rt.ledger else True,
    }


if __name__ == "__main__":
    from src.proof_ladder import normalize_cfg
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    for fam in ("linear_oscillator", "nonlinear_oscillator", "henon_map"):
        print(run_stage_a(fam, 120, cfg))
