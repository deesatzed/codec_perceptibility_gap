"""claimgate — gate a quantitative claim; refuse to report a number until it's earned.

Wrap a quantitative claim in adversarial controls; the result is either
Verified(value, receipts) or Refused(reason, evidence). This is RIGOR PACKAGED
FOR REUSE, not a novel technique — every control is standard statistics
(base-rate, difficulty/triviality baseline, permutation null, bootstrap-CI
separation, sampling coverage, collinearity). The value is the composition + the
default-to-refuse contract, demonstrated on real worked examples.

Origin: distilled from a research program (energy/reliability/agent/carbon audits)
where this discipline repeatedly caught overclaims — including its own.

Quick use:
    from claimgate import Battery, base_rate, beats_difficulty
    b = Battery([
        base_rate(wrong, floor=0.30),
        beats_difficulty(signal, difficulty, wrong),
    ])
    result = b.evaluate()            # Verified(...) or Refused(...)
"""
__version__ = "0.1.0"

from .contract import (
    Verified, Refused, Result, Control, Battery, CheckResult,
    Earned, RefusedError, require_verified,
)
from .controls import (
    base_rate,
    beats_difficulty,
    permutation_null,
    collinearity,
    coverage,
    improvement_beats_noise,
)

__all__ = [
    "Verified", "Refused", "Result", "Control", "Battery", "CheckResult",
    "Earned", "RefusedError", "require_verified",
    "base_rate", "beats_difficulty", "permutation_null", "collinearity", "coverage",
    "improvement_beats_noise",
]
