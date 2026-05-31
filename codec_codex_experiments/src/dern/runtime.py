"""DERN runtime loop (Stage A).

route(): probe -> graph lookup -> controller propose -> VERIFIER -> actuator/ledger
-> graph+controller update. Any rejection falls back to full compute and serves
the verified full result. Reward + trust are minted only from the verifier verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.dern.probe import distinction_key, probe_cost_units
from src.dern.configs import COMPUTE_CONFIGS, full_config, compute_cost_units
from src.dern.verifier import verify_lane1, verify_lane2
from src.dern.cost import cost_vector, net_savings
from src.dern.graph import ExperienceGraph
from src.dern.controller import OnlineController
from src.dern.breaker import TrustBreaker

_POSTURES = ["default", "low_fan", "race_to_idle"]


class DERNRuntime:
    def __init__(self, epsilon: float = 0.1, audit_prob: float = 1.0,
                 eps_tolerance: float = 0.3, seed: int = 0) -> None:
        names = [c["name"] for c in COMPUTE_CONFIGS]
        self._by_name = {c["name"]: c for c in COMPUTE_CONFIGS}
        self.controller = OnlineController(actions=names, seed=seed, epsilon=epsilon)
        self.graph = ExperienceGraph()
        self.breaker = TrustBreaker()
        self.audit_prob = float(audit_prob)
        self.eps_tolerance = float(eps_tolerance)
        self.ledger: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)

    def route(self, src, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
        # 1) probe one representative trajectory
        xprobe, _ = src.sample(1, cfg, seed)
        key = distinction_key(xprobe[0])
        overhead = probe_cost_units()

        # 2) forced full while breaker tripped or region locked
        forced_full = self.breaker.must_force_full() or self.graph.is_locked(key)

        # 3) choose config (graph replay if trusted, else controller)
        edge = None if forced_full else self.graph.lookup(key)
        if forced_full:
            chosen_name = "full"
        elif edge is not None:
            chosen_name = edge.config["name"]
        else:
            chosen_name = self.controller.choose()
        chosen = self._by_name[chosen_name]
        posture = "low_fan"  # Stage-A simplest non-default posture; controller-extendable

        # 4) VERIFY (controller cannot influence this)
        if chosen_name == "full":
            verdict = verify_lane1(src, cfg, full_config(), seed)  # exact, trivially passes
            served = full_config()
        else:
            v1 = verify_lane1(src, cfg, chosen, seed)
            if v1.passed:
                verdict, served = v1, chosen
            else:
                # Lane 2 audit (sampled)
                v2 = verify_lane2(src, cfg, chosen, seed, self.eps_tolerance)
                if v2.passed:
                    verdict, served = v2, chosen
                else:
                    # rejection -> fall back to full (never serve unverified)
                    verdict = verify_lane1(src, cfg, full_config(), seed)
                    served = full_config()

        # 5) cost + net savings (overhead always charged to chosen side)
        baseline_cv = cost_vector(full_config(), "default", 0.0)
        served_cv = cost_vector(served, posture, overhead)
        net = net_savings(baseline_cv, served_cv)

        # 6) learning signal — minted ONLY from verdict + cost
        passed = verdict.passed and served["name"] == chosen["name"]
        reward = net if passed else -compute_cost_units(full_config())
        self.controller.update(chosen_name, reward=reward)
        self.graph.record(key, chosen, compute_cost_units(chosen), verdict.proof,
                          passed=passed and chosen_name != "full")
        self.breaker.observe(passed=verdict.passed)
        self.breaker.tick()

        rec = {
            "key": key, "chosen_config": chosen, "served_config": served,
            "lane": verdict.lane, "verified": verdict.passed, "proof": verdict.proof,
            "delta": verdict.delta, "cost_vector": served_cv, "net_savings": net,
            "reward": reward, "forced_full": forced_full,
        }
        self.ledger.append(rec)
        return rec
