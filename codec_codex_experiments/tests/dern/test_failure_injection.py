import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime
from src.dern.graph import ExperienceGraph
from src.dern.breaker import TrustBreaker


def _cfg():
    return normalize_cfg({"n_train": 200, "n_test": 90, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_drift_evicts_trusted_edge_and_locks_region():
    g = ExperienceGraph()
    g.record((1, 1, 1), {"name": "cheap"}, 2.0, "bounded", passed=True)
    g.record((1, 1, 1), {"name": "cheap"}, 2.0, "bounded", passed=False)
    assert g.lookup((1, 1, 1)) is None
    assert g.is_locked((1, 1, 1)) is True


def test_breaker_trips_under_sustained_failures_forces_full():
    b = TrustBreaker(window=6, fail_rate_trip=0.5, cooldown=4)
    for _ in range(4):
        b.observe(passed=False)
    assert b.tripped and b.must_force_full()


def test_locked_region_forces_full_in_runtime():
    rt = DERNRuntime(epsilon=0.0, audit_prob=1.0, eps_tolerance=0.4, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec0 = rt.route(src, _cfg(), seed=0)
    rt.graph._locked.add(rec0["key"])
    rec1 = rt.route(src, _cfg(), seed=0)
    assert rec1["forced_full"] is True
    assert rec1["served_config"]["name"] == "full"


def test_overhead_can_make_route_net_negative_and_is_not_hidden():
    from src.dern.cost import cost_vector, net_savings
    base = cost_vector({"name": "full", "channel": "direct", "k": 16, "cost": 5.0}, "default", 0.0)
    chosen = cost_vector({"name": "cheap", "channel": "native", "k": 3, "cost": 1.0}, "default", 100.0)
    assert net_savings(base, chosen) < 0
