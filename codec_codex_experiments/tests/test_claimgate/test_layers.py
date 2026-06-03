"""Tests for the Rust/Zig-informed contract layers."""
import pytest
from claimgate import Battery, Verified, Refused, RefusedError, Earned, require_verified
from claimgate import CheckResult


def _ok():
    return lambda: CheckResult("ok", True, "fine", {})


def _no():
    return lambda: CheckResult("no", False, "nope", {})


# ---- Layer 1: enforced extraction ----
def test_unwrap_verified_returns_value():
    r = Battery([_ok()]).evaluate(value=7)
    assert r.unwrap() == 7 and r.expect("should pass") == 7 and r.is_verified()


def test_unwrap_refused_raises_expect_carries_msg():
    r = Battery([_no()]).evaluate(value=7)
    assert r.is_refused()
    with pytest.raises(RefusedError):
        r.unwrap()
    with pytest.raises(RefusedError, match="custom context"):
        r.expect("custom context")


def test_match_forces_both_arms():
    v = Battery([_ok()]).evaluate(value=10)
    out_v = v.match(verified=lambda x: x * 2, refused=lambda r: -1)
    assert out_v == 20
    f = Battery([_no()]).evaluate(value=10)
    out_f = f.match(verified=lambda x: x * 2, refused=lambda r: -1)
    assert out_f == -1


# ---- Layer 2: typestate / Earned brand ----
def test_earned_minted_only_from_verified():
    e = Battery([_ok()]).evaluate(value=5).earned()
    assert isinstance(e, Earned) and e.get() == 5
    with pytest.raises(RefusedError):
        Battery([_no()]).evaluate(value=5).earned()


def test_require_verified_rejects_raw_number():
    e = Battery([_ok()]).evaluate(value=5).earned()
    assert require_verified(e) == 5                 # branded ok
    assert require_verified(Battery([_ok()]).evaluate(value=9)) == 9   # Verified ok
    with pytest.raises(RefusedError):
        require_verified(3.14)                       # raw unverified number rejected


def test_downstream_function_can_require_earned():
    def deploy(metric):
        val = require_verified(metric)   # boundary guard: only earned numbers in
        return f"deployed at {val}"
    good = Battery([_ok()]).evaluate(value=0.9).earned()
    assert deploy(good) == "deployed at 0.9"
    with pytest.raises(RefusedError):
        deploy(0.9)                       # someone passed a raw number -> refused


# ---- Layer 3: explicit pipeline ----
def test_and_then_short_circuits_on_refusal():
    calls = []
    def step(x):
        calls.append(x)
        return Battery([_ok()]).evaluate(value=x + 1)
    # verified -> step runs
    Battery([_ok()]).evaluate(value=1).and_then(step)
    assert calls == [1]
    # refused -> step is NOT run (short-circuit)
    calls.clear()
    Battery([_no()]).evaluate(value=1).and_then(step)
    assert calls == []


def test_map_transforms_verified_passes_refused():
    v = Battery([_ok()]).evaluate(value=10).map(lambda x: x * 3)
    assert v.unwrap() == 30 and len(v.receipts) == 1   # receipts carried forward
    f = Battery([_no()]).evaluate(value=10).map(lambda x: x * 3)
    assert f.is_refused()                               # refusal passes through unchanged
