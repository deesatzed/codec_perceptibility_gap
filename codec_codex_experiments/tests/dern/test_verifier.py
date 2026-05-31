import inspect
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.configs import COMPUTE_CONFIGS, full_config
from src.dern.verifier import verify_lane1, verify_lane2, Verdict


def _cfg():
    return normalize_cfg({"n_train": 300, "n_test": 120, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_lane1_full_config_is_exact():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    v = verify_lane1(src, cfg, full_config(), seed=0)
    assert v.passed is True and v.proof == "exact" and v.delta == 0.0


def test_lane1_cheaper_config_rejected_unless_identical():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    v = verify_lane1(src, cfg, COMPUTE_CONFIGS[0], seed=0)
    assert v.passed is False


def test_lane2_passes_within_epsilon_fails_outside():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    loose = verify_lane2(src, cfg, COMPUTE_CONFIGS[1], seed=0, epsilon=1.0)
    tight = verify_lane2(src, cfg, COMPUTE_CONFIGS[0], seed=0, epsilon=0.0)
    assert loose.passed is True
    assert tight.passed is False        # zero tolerance => any deviation fails


def test_verifier_signature_takes_no_controller_state():
    for fn in (verify_lane1, verify_lane2):
        params = set(inspect.signature(fn).parameters)
        assert "controller" not in params and "policy" not in params
