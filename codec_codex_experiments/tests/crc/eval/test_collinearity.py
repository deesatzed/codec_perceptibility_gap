"""Gate 2 tests — the false-positive catcher (2-family collinearity)."""
import numpy as np
from src.crc.eval.collinearity import collinearity_guard
from src.crc.eval import status as st


def test_blocks_two_families():
    rng = np.random.default_rng(0)
    d = rng.random(40).tolist()
    w = (rng.random(40) < 0.5).astype(float).tolist()
    out = collinearity_guard(2, d, w)
    assert out["status"] == st.COLLINEAR_UNVALIDATED and out["gate_pass"] is False


def test_blocks_high_corr_even_with_three_families():
    # difficulty ~= ensemble_wrong -> partial undefined -> refuse
    w = ([1.0] * 20 + [0.0] * 20)
    d = list(w)                       # identical -> corr 1.0
    out = collinearity_guard(3, d, w)
    assert out["status"] == st.COLLINEAR_UNVALIDATED
    assert out["corr"] is None or out["corr"] >= 0.95
    assert out["gate_pass"] is False


def test_allows_three_distinct():
    rng = np.random.default_rng(1)
    w = (rng.random(60) < 0.5).astype(float)
    # difficulty correlated but NOT identical to wrong
    d = (w * 0.5 + rng.random(60) * 0.5)
    out = collinearity_guard(3, d.tolist(), w.tolist())
    assert out["gate_pass"] is True and out["status"] == st.OK


def test_degenerate_variance_refused():
    out = collinearity_guard(3, [0.5] * 40, [1.0] * 20 + [0.0] * 20)
    assert out["status"] == st.COLLINEAR_UNVALIDATED
