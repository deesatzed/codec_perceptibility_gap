from src.dern_b.probe_text import distinction_key, probe_cost_units


def test_key_deterministic_and_int_tuple():
    k1 = distinction_key("What is 2 + 2?")
    k2 = distinction_key("What is 2 + 2?")
    assert k1 == k2
    assert isinstance(k1, tuple) and all(isinstance(x, int) for x in k1)


def test_key_separates_short_factual_from_long_reasoning():
    short = distinction_key("Capital of France?")
    long = distinction_key(
        "Explain step by step why the derivative of the chaotic map diverges and "
        "reason carefully through each intermediate term in the expansion at length"
    )
    assert short != long


def test_probe_cost_small_positive():
    assert 0 < probe_cost_units() < 0.1
