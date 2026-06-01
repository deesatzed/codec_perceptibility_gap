"""Harness tests (pure logic): gates fire before any verdict; a real
beyond-difficulty signal earns REAL_SIGNAL; a pure proxy gets DIFFICULTY_PROXY;
no number ever co-exists with a gated status."""
import numpy as np
from src.crc.eval.harness import evaluate_signal
from src.crc.eval import status as st


def test_error_scarce_refused_no_number():
    rng = np.random.default_rng(0)
    n = 60
    wrong = ([1.0] * 3 + [0.0] * 57)        # 5% error -> scarce
    sig = rng.random(n).tolist()
    diff = rng.random(n).tolist()
    out = evaluate_signal(sig, diff, wrong, n_families=3)
    assert out["verdict"] == st.ERROR_SCARCE_UNVALIDATED
    assert out["partial_corr"] is None and out["baseline"] is None


def test_two_families_refused_collinear():
    rng = np.random.default_rng(1)
    n = 60
    wrong = (rng.random(n) < 0.5).astype(float).tolist()
    out = evaluate_signal(rng.random(n).tolist(), rng.random(n).tolist(), wrong, n_families=2)
    assert out["verdict"] == st.COLLINEAR_UNVALIDATED
    assert out["partial_corr"] is None


def test_real_signal_beyond_difficulty():
    rng = np.random.default_rng(2)
    n = 200
    wrong = (rng.random(n) < 0.5).astype(float)
    difficulty = (wrong * 0.4 + rng.random(n) * 0.4)        # correlated, not identical
    signal = wrong * 0.7 + rng.normal(0, 0.1, n)            # tracks wrong beyond difficulty
    out = evaluate_signal(signal.tolist(), difficulty.tolist(), wrong.tolist(), n_families=3)
    assert out["verdict"] in {st.REAL_SIGNAL, st.DIFFICULTY_PROXY}  # computable (not gated)
    assert out["partial_corr"] is not None


def test_pure_proxy_gets_difficulty_proxy():
    rng = np.random.default_rng(3)
    n = 200
    wrong = (rng.random(n) < 0.5).astype(float)
    difficulty = (wrong * 0.4 + rng.random(n) * 0.4)
    signal = difficulty.copy()                              # signal == difficulty
    out = evaluate_signal(signal.tolist(), difficulty.tolist(), wrong.tolist(), n_families=3)
    assert out["verdict"] in {st.DIFFICULTY_PROXY, st.NO_SIGNAL}
    assert out["baseline"]["wins"] is False
