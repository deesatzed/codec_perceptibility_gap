from src.dern.breaker import TrustBreaker


def test_breaker_trips_on_high_fail_rate():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=3)
    for _ in range(3):
        b.observe(passed=False)
    assert b.tripped is True


def test_breaker_forces_full_compute_while_tripped():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=3)
    for _ in range(3):
        b.observe(passed=False)
    assert b.must_force_full() is True


def test_breaker_resets_after_cooldown():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=2)
    for _ in range(3):
        b.observe(passed=False)
    assert b.tripped is True
    b.tick(); b.tick()
    assert b.tripped is False
