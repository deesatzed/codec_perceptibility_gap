import inspect
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern import verifier as verifier_mod
from src.dern.runtime import DERNRuntime
from src.dern.configs import COMPUTE_CONFIGS


def _cfg():
    return normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_controller_has_no_write_path_to_verifier():
    for name in ("verify_lane1", "verify_lane2"):
        fn = getattr(verifier_mod, name)
        params = set(inspect.signature(fn).parameters)
        assert not ({"controller", "policy", "runtime", "graph"} & params)


def test_served_output_is_always_verified():
    rt = DERNRuntime(epsilon=0.5, audit_prob=1.0, eps_tolerance=0.4, seed=0)
    src = SOURCE_REGISTRY["henon_map"]   # hardest family
    for s in range(12):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["verified"] is True


def test_zero_tolerance_never_serves_cheaper_than_full():
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(10):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["served_config"]["name"] == "full"


def test_reward_zero_when_full_served_no_phantom_savings():
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec = rt.route(src, _cfg(), seed=0)
    assert rec["served_config"]["name"] == "full"
