from src.refuse import Battery, Verified, Refused
from src.refuse.contract import CheckResult


def _pass(name="ok"):
    return lambda: CheckResult(name, True, "fine", {})


def _fail(name="bad"):
    return lambda: CheckResult(name, False, "nope", {"k": 1})


def test_all_pass_returns_verified():
    r = Battery([_pass("a"), _pass("b")]).evaluate(value=42)
    assert isinstance(r, Verified) and r.value == 42 and r.status == "VERIFIED"
    assert len(r.receipts) == 2


def test_any_fail_returns_refused_with_first_reason():
    r = Battery([_pass("a"), _fail("b"), _fail("c")]).evaluate(value=42)
    assert isinstance(r, Refused) and r.status == "REFUSED"
    assert r.failed == "b" and r.reason == "nope"
    assert len(r.evidence) == 3            # all checks recorded, passed + failed


def test_default_is_refuse_not_a_silent_number():
    # the whole point: a failing control yields NO value, a typed refusal instead
    r = Battery([_fail()]).evaluate(value=99)
    assert isinstance(r, Refused)
    assert not hasattr(r, "value")


def test_empty_battery_rejected():
    import pytest
    with pytest.raises(ValueError):
        Battery([])
