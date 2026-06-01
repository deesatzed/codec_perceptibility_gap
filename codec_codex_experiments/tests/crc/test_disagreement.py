import numpy as np
from src.crc.disagreement import answer_disagreement


def test_identical_answers_zero_disagreement():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert answer_disagreement(v) == 0.0


def test_scattered_answers_higher_than_close():
    close = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.05]])
    far = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert answer_disagreement(far) > answer_disagreement(close)


def test_single_family_is_zero():
    assert answer_disagreement(np.array([[1.0, 2.0]])) == 0.0
