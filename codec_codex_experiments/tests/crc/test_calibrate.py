import numpy as np
from src.crc.calibrate import calibrate, MIN_N


def _arrays(n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    wrong = rng.integers(0, 2, n).astype(float)
    if signal:          # disagreement tracks wrongness
        disagree = wrong * 0.5 + rng.normal(0, 0.05, n) + 0.1
    else:               # disagreement is pure noise wrt wrongness
        disagree = rng.normal(0.3, 0.1, n)
    return disagree, wrong


def test_refuses_below_min_n():
    d, w = _arrays(MIN_N - 1)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "UNVALIDATED" and out["threshold"] is None


def test_refuses_when_signal_is_noise():
    d, w = _arrays(200, signal=False)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "UNVALIDATED"   # must not clear permutation null


def test_validates_and_thresholds_when_signal_real():
    d, w = _arrays(200, signal=True)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "VALID"
    assert out["threshold"] is not None
    assert 0.0 <= out["precision"] <= 1.0 and 0.0 <= out["recall"] <= 1.0


def test_correlated_error_guardrail_warns():
    d, w = _arrays(200, signal=True)
    out = calibrate(d, w, pairwise_err_corr=0.7)
    assert out["hollow_agreement_warning"] is True
