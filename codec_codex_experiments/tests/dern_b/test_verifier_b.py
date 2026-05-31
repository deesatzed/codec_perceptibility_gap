import inspect
from src.dern_b.verifier_b import verify_against_reference, token_f1, VerdictB


def test_identical_answers_pass_at_zero_epsilon():
    v = verify_against_reference("Paris", "Paris", epsilon=0.0, metric="exact_match")
    assert v.passed is True and v.delta == 0.0


def test_divergent_answers_fail_tight_epsilon():
    v = verify_against_reference("Paris", "Berlin is the capital", epsilon=0.1)
    assert v.passed is False and v.proof == "rejected"


def test_close_answers_pass_loose_epsilon():
    v = verify_against_reference(
        "The capital of France is Paris.", "Paris is the capital of France.",
        epsilon=0.5,
    )
    assert v.passed is True


def test_token_f1_bounds():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b c", "x y z") == 0.0


def test_verifier_takes_no_controller_state():
    params = set(inspect.signature(verify_against_reference).parameters)
    assert not ({"controller", "policy", "runtime", "graph"} & params)
