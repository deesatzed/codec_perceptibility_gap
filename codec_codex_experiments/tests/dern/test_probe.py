import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.probe import distinction_key, probe_cost_units


def _cfg():
    return normalize_cfg({"n_train": 200, "n_test": 90, "T": 6.0, "dt": 0.1, "seeds": [0]})


def test_key_is_deterministic_and_nonsemantic():
    cfg = _cfg()
    traj, _ = SOURCE_REGISTRY["linear_oscillator"].sample(10, cfg, 0)
    k1 = distinction_key(traj[0])
    k2 = distinction_key(traj[0])
    assert k1 == k2                      # deterministic
    assert isinstance(k1, tuple)         # non-semantic structured key
    assert all(isinstance(x, int) for x in k1)


def test_key_separates_easy_from_hard():
    # Linear (easy: recoverable params) vs Henon (hard: chaotic, near-unrecoverable).
    # Their key distributions must differ, or the probe carries no signal.
    cfg = _cfg()
    lin, _ = SOURCE_REGISTRY["linear_oscillator"].sample(60, cfg, 0)
    hen, _ = SOURCE_REGISTRY["henon_map"].sample(60, cfg, 0)
    lin_keys = {distinction_key(lin[i]) for i in range(len(lin))}
    hen_keys = {distinction_key(hen[i]) for i in range(len(hen))}
    overlap = len(lin_keys & hen_keys) / max(len(lin_keys | hen_keys), 1)
    assert overlap < 0.5, f"probe cannot separate families (overlap={overlap:.2f})"


def test_probe_cost_is_positive_and_small():
    assert 0 < probe_cost_units() < 1.0
