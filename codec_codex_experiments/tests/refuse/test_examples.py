"""The worked examples must keep demonstrating the intended contrast."""
from src.refuse.examples.example_difficulty import _case
from src.refuse.examples.example_carbon import (
    control_choice_not_outcome_determining, PROJECT_LOSS, CONTROL_LOSSES,
)
from src.refuse.contract import Battery


def test_difficulty_real_signal_verified_proxy_refused():
    assert _case(real_signal=True).status == "VERIFIED"
    assert _case(real_signal=False).status == "REFUSED"


def test_carbon_control_dependent_claim_is_refused():
    r = Battery([control_choice_not_outcome_determining(PROJECT_LOSS, CONTROL_LOSSES)]).evaluate()
    assert r.status == "REFUSED"
    assert "control-dependent" in r.reason


def test_carbon_robust_claim_would_verify():
    # a hypothetical project that lost far less than ALL controls -> robust -> verified
    r = Battery([control_choice_not_outcome_determining(0.05, [1.0, 1.2, 0.9, 1.1])]).evaluate()
    assert r.status == "VERIFIED"
