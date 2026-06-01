from src.crc.correctness import is_correct


def test_exact_match_case_insensitive():
    assert is_correct("Paris", "paris") is True
    assert is_correct("Lyon", "Paris") is False


def test_substring_answer_counts_correct():
    assert is_correct("The capital is Paris.", "Paris") is True


def test_empty_key_is_false():
    assert is_correct("anything", "") is False
