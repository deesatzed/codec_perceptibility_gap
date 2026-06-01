import numpy as np
from src.crc.guardrail import pairwise_error_correlation


def test_independent_failures_low_corr():
    rng = np.random.default_rng(0)
    errs = rng.integers(0, 2, (3, 200)).astype(float)
    assert pairwise_error_correlation(errs) < 0.3


def test_together_failures_high_corr():
    base = (np.random.default_rng(1).integers(0, 2, 200)).astype(float)
    errs = np.stack([base, base, base])
    assert pairwise_error_correlation(errs) > 0.9
