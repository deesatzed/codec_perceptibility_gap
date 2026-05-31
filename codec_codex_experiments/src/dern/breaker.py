"""Trust circuit breaker — bounds online-learning instability (from CIO-II).

Tripping degrades the system to full compute + audit (the known-good baseline)
and freezes adaptation; it resets after a cool-down with the failure history
cleared. Worst case under instability is therefore the baseline, never worse.
"""
from __future__ import annotations

from collections import deque


class TrustBreaker:
    def __init__(self, window: int = 20, fail_rate_trip: float = 0.3, cooldown: int = 10) -> None:
        self._hist: deque[bool] = deque(maxlen=window)
        self._fail_rate_trip = fail_rate_trip
        self._cooldown = cooldown
        self._cooldown_left = 0
        self.tripped = False

    def observe(self, passed: bool) -> None:
        self._hist.append(bool(passed))
        if not self.tripped and len(self._hist) >= 1:
            fails = sum(1 for p in self._hist if not p)
            if fails / len(self._hist) >= self._fail_rate_trip:
                self.tripped = True
                self._cooldown_left = self._cooldown

    def tick(self) -> None:
        if self.tripped:
            self._cooldown_left -= 1
            if self._cooldown_left <= 0:
                self.tripped = False
                self._hist.clear()

    def must_force_full(self) -> bool:
        return self.tripped
