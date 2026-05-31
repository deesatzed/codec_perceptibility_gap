"""Validity gates for the Stage-B energy-savings number (ultrathink mitigation).

A savings number is emitted ONLY if all gates pass; otherwise the harness reports
gate status, never a number — the same honest-failure discipline used for the
sudo/coverage path. Thresholds are validity floors tied to measured noise, NOT
values tuned to admit a borderline saving.

Gates:
- coverage_parity: both conditions clear the probe's 60% coverage floor AND are
  within 15% of each other (mismatched windows aren't comparable).
- significant_saving: K>=5 repeats AND |mean| > stdev (a saving inside run-to-run
  noise is not a saving).
- audit_overhead_charged: the published cascade total must include the audit
  double-run cost (a single audited route runs BOTH models -> more than reference).
- idle_drift_ok: idle baselines before/after the run must agree within tolerance,
  else thermal/background drift could fake the saving sign.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List, Optional

COVERAGE_FLOOR = 0.60          # per-condition (matches energy_probe single-window guard)
COVERAGE_PARITY = 0.15         # max relative gap between the two conditions' coverage
MIN_REPEATS = 5                # variance gate
IDLE_DRIFT_REL = 0.25          # max relative idle-baseline drift across the run


def coverage_parity_ok(cov_ref: float, cov_casc: float) -> bool:
    if cov_ref < COVERAGE_FLOOR or cov_casc < COVERAGE_FLOOR:
        return False
    denom = max(cov_ref, cov_casc, 1e-9)
    return abs(cov_ref - cov_casc) / denom <= COVERAGE_PARITY


def significant_saving(deltas: List[float]) -> Optional[float]:
    """Return mean delta iff K>=MIN_REPEATS and |mean| > stdev, else None.

    deltas are per-repeat paired (reference - cascade) savings. A saving smaller
    than the across-repeat spread is indistinguishable from noise -> None.
    """
    if len(deltas) < MIN_REPEATS:
        return None
    m = mean(deltas)
    sd = pstdev(deltas)
    if abs(m) <= sd:
        return None
    return round(m, 3)


def audit_overhead_charged(published_cascade_total: float, served_total: float,
                           audit_extra: float) -> bool:
    """The published cascade total must include the audit double-run overhead.
    (served_total alone undercounts; true total = served + audit_extra.)"""
    return published_cascade_total + 1e-6 >= served_total + audit_extra


def idle_drift_ok(idle_pre_watts: float, idle_post_watts: float) -> bool:
    denom = max(idle_pre_watts, 1e-9)
    return abs(idle_pre_watts - idle_post_watts) / denom <= IDLE_DRIFT_REL


def evaluate_gates(cov_ref: float, cov_casc: float, deltas: List[float],
                   published_cascade_total: float, served_total: float, audit_extra: float,
                   idle_pre_watts: float, idle_post_watts: float) -> Dict:
    """Run all gates. Returns verdict + per-gate status. A savings number is only
    publishable when verdict == 'VALID'."""
    sig = significant_saving(deltas)
    gates = {
        "coverage_parity": coverage_parity_ok(cov_ref, cov_casc),
        "significant_saving": sig is not None,
        "audit_overhead_charged": audit_overhead_charged(published_cascade_total, served_total, audit_extra),
        "idle_drift_ok": idle_drift_ok(idle_pre_watts, idle_post_watts),
        "k_repeats": len(deltas) >= MIN_REPEATS,
    }
    valid = all(gates.values())
    return {
        "verdict": "VALID" if valid else "GATED_NO_NUMBER",
        "gates": gates,
        "significant_saving_J": sig if valid else None,
        "thresholds": {
            "coverage_floor": COVERAGE_FLOOR, "coverage_parity": COVERAGE_PARITY,
            "min_repeats": MIN_REPEATS, "idle_drift_rel": IDLE_DRIFT_REL,
        },
        "note": ("A savings number is emitted only when verdict==VALID. Thresholds "
                 "are validity floors tied to measured noise, not tuned to pass."),
    }
