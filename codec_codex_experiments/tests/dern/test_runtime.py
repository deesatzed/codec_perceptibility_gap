import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime


def _cfg():
    return normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_route_returns_verified_outcome_and_ledger_record():
    rt = DERNRuntime(epsilon=0.2, audit_prob=1.0, eps_tolerance=0.5, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec = rt.route(src, _cfg(), seed=0)
    assert rec["verified"] is True
    assert rec["lane"] in (1, 2)
    assert rec["chosen_config"]["name"] in {"cheap", "mid", "full"}
    assert "net_savings" in rec and "cost_vector" in rec
    assert all(t == "simulated" for t in rec["cost_vector"]["_tags"].values())


def test_rejected_cheap_falls_back_to_full_never_unverified():
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(8):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["verified"] is True
        if rec["chosen_config"]["name"] != "full":
            assert rec["served_config"]["name"] == "full"


def test_ledger_accumulates_one_record_per_route():
    rt = DERNRuntime(epsilon=0.2, audit_prob=1.0, eps_tolerance=0.5, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(5):
        rt.route(src, _cfg(), seed=s)
    assert len(rt.ledger) == 5
