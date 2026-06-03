import numpy as np
from src.refuse import Battery, improvement_beats_noise
from src.refuse.examples.example_plus_one_percent import _accuracies


def test_noisy_small_gain_refused():
    a, b = _accuracies(true_gain=0.01, noise=0.02, seed=0)
    r = Battery([improvement_beats_noise(a.tolist(), b.tolist())]).evaluate()
    assert r.status == "REFUSED" and "includes 0" in r.reason


def test_real_gain_verified():
    a, b = _accuracies(true_gain=0.06, noise=0.01, seed=0)
    r = Battery([improvement_beats_noise(a.tolist(), b.tolist())]).evaluate()
    assert r.status == "VERIFIED"


def test_unpaired_inputs_refused():
    r = Battery([improvement_beats_noise([0.8, 0.81], [0.82])]).evaluate()
    assert r.status == "REFUSED"
