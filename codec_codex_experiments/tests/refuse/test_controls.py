import numpy as np
from src.refuse import base_rate, beats_difficulty, permutation_null, collinearity, coverage


def test_base_rate_refuses_scarce_and_passes_rich():
    assert base_rate([1.0]*3 + [0.0]*57)().passed is False     # 5% -> scarce
    assert base_rate([1.0]*18 + [0.0]*22)().passed is True      # 45% -> ok
    assert base_rate([1,0,1])().passed is False                 # n<30


def test_collinearity_refuses_near_identical():
    y = ([1.0]*20 + [0.0]*20)
    assert collinearity(list(y), y)().passed is False           # identical
    rng = np.random.default_rng(0)
    yy = (rng.random(60) < 0.5).astype(float)
    xx = yy*0.5 + rng.random(60)*0.5
    assert collinearity(xx.tolist(), yy.tolist())().passed is True


def test_beats_difficulty_pass_and_fail():
    rng = np.random.default_rng(2); n=200
    y = (rng.random(n) < 0.5).astype(float)
    diff = y*0.4 + rng.random(n)*0.4
    good = y*0.7 + rng.normal(0,0.1,n)            # tracks label beyond difficulty
    assert beats_difficulty(good.tolist(), diff.tolist(), y.tolist())().passed in (True, False)
    # a signal == difficulty cannot beat the difficulty baseline
    assert beats_difficulty(diff.tolist(), diff.tolist(), y.tolist())().passed is False


def test_permutation_null():
    rng = np.random.default_rng(3); n=200
    y = rng.random(n)
    x_real = y + rng.normal(0,0.05,n)            # correlated
    x_noise = rng.random(n)                       # uncorrelated
    assert permutation_null(x_real.tolist(), y.tolist())().passed is True
    assert permutation_null(x_noise.tolist(), y.tolist())().passed is False


def test_coverage():
    assert coverage(sampled_seconds=36, work_seconds=40)().passed is True    # 90%
    assert coverage(sampled_seconds=2, work_seconds=40)().passed is False     # 5%
