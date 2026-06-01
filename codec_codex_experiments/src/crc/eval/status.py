"""Typed status vocabulary for the difficulty-disentangled evaluation.

A failed precondition returns a typed status, NEVER a number. The hard invariant
(enforced by tests): a non-OK status and a numeric verdict field must never
co-exist for the same gated quantity. Mirrors calibrate.py's UNVALIDATED.
"""
from __future__ import annotations

# gate / precondition statuses
OK = "OK"
ERROR_SCARCE_UNVALIDATED = "ERROR_SCARCE_UNVALIDATED"   # ensemble error below pre-screen floor
COLLINEAR_UNVALIDATED = "COLLINEAR_UNVALIDATED"         # <3 families or difficulty==wrong
INSUFFICIENT_N = "INSUFFICIENT_N"                       # too few items / single class

# per-signal verdicts (only reachable once gates pass)
REAL_SIGNAL = "REAL_SIGNAL"                # beats difficulty baseline, CI clears 0
DIFFICULTY_PROXY = "DIFFICULTY_PROXY"      # does not beat difficulty
NO_SIGNAL = "NO_SIGNAL"                    # no predictive value at all

GATED = {ERROR_SCARCE_UNVALIDATED, COLLINEAR_UNVALIDATED, INSUFFICIENT_N}
VERDICTS = {REAL_SIGNAL, DIFFICULTY_PROXY, NO_SIGNAL}
ALL = GATED | VERDICTS | {OK}
