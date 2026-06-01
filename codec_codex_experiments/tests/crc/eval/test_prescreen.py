"""Gate 1 tests — the primary flaw-catcher. Error-scarce data must be REFUSED.
This gate alone would have stopped both prior misleading CRC runs."""
import numpy as np
from src.crc.eval.prescreen import prescreen_error_rate, MIN_N
from src.crc.eval import status as st


def test_prescreen_refuses_error_scarce():
    # 10% error over 40 items -> nothing to predict -> refuse, number nulled
    w = [1.0] * 4 + [0.0] * 36
    out = prescreen_error_rate(w)
    assert out["status"] == st.ERROR_SCARCE_UNVALIDATED
    assert out["gate_pass"] is False
    assert out["error_rate"] == 0.1   # measured rate reported, but gate fails


def test_prescreen_admits_error_rich():
    rng = np.random.default_rng(0)
    w = (rng.random(40) < 0.45).astype(float).tolist()
    out = prescreen_error_rate(w)
    assert out["status"] == st.OK and out["gate_pass"] is True


def test_prescreen_boundary_is_inclusive():
    # exactly 30% -> admit (>= is the rule; regression guard against >/>= flip)
    w = [1.0] * 12 + [0.0] * 28   # 12/40 = 0.30
    out = prescreen_error_rate(w)
    assert out["gate_pass"] is True and out["status"] == st.OK


def test_prescreen_insufficient_n():
    out = prescreen_error_rate([1.0, 0.0, 1.0])   # < MIN_N
    assert out["status"] == st.INSUFFICIENT_N and out["error_rate"] is None


def test_prescreen_refuses_all_wrong():
    out = prescreen_error_rate([1.0] * 40)   # single class -> insufficient
    assert out["gate_pass"] is False
