import numpy as np
from src.crc.eval.mc_logprob import (
    option_letters, model_choice, distribution_disagreement,
    family_correct, ensemble_wrong, difficulty,
)


def test_option_letters():
    assert option_letters(3) == ["A", "B", "C"]


def test_model_choice_argmax():
    assert model_choice(np.array([0.1, 0.7, 0.2])) == 1


def test_identical_distributions_zero_disagreement():
    p = np.array([0.2, 0.5, 0.3])
    assert distribution_disagreement({"a": p, "b": p, "c": p}) < 1e-9


def test_scattered_distributions_more_disagreement():
    close = {"a": np.array([0.6, 0.4]), "b": np.array([0.55, 0.45])}
    far = {"a": np.array([0.9, 0.1]), "b": np.array([0.1, 0.9])}
    assert distribution_disagreement(far) > distribution_disagreement(close)


def test_ensemble_wrong_and_difficulty():
    gold = 0
    # 2 of 3 families pick the wrong option -> ensemble wrong, difficulty 2/3
    probs = {"a": np.array([0.8, 0.2]), "b": np.array([0.2, 0.8]), "c": np.array([0.1, 0.9])}
    assert family_correct(probs["a"], gold) is True
    assert ensemble_wrong(probs, gold) == 1.0
    assert abs(difficulty(probs, gold) - 2 / 3) < 1e-9
