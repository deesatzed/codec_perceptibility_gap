"""Validity-gate tests (ultrathink). Pure logic — these are the ship-gate that
turns a silent-wrong-number into a loud refusal to emit one."""
from src.dern_b.gates_b import (
    coverage_parity_ok, significant_saving, audit_overhead_charged,
    idle_drift_ok, evaluate_gates,
)


def test_coverage_parity_requires_floor_and_match():
    assert coverage_parity_ok(0.95, 0.92) is True      # both high, close
    assert coverage_parity_ok(0.95, 0.55) is False     # one below 60% floor
    assert coverage_parity_ok(0.95, 0.61) is False     # both clear floor but >15% apart


def test_significant_saving_needs_k_and_signal_over_noise():
    assert significant_saving([5.0]) is None                    # too few repeats
    assert significant_saving([1, -2, 3, -1, 2]) is None        # mean~0.6 < stdev~1.9 -> noise
    v = significant_saving([20, 22, 19, 21, 18])                # mean 20, stdev ~1.4
    assert v is not None and abs(v - 20) < 1.5


def test_audit_overhead_must_be_charged():
    # published total includes audit extra -> ok
    assert audit_overhead_charged(published_cascade_total=150.0, served_total=100.0, audit_extra=50.0) is True
    # published total omits the audit double-run -> NOT ok (savings would be overstated)
    assert audit_overhead_charged(published_cascade_total=100.0, served_total=100.0, audit_extra=50.0) is False


def test_idle_drift_gate():
    assert idle_drift_ok(0.66, 0.70) is True       # ~6% drift, fine
    assert idle_drift_ok(0.66, 1.20) is False      # ~80% drift -> thermal/background, reject


def test_evaluate_gates_emits_number_only_when_all_pass():
    good = evaluate_gates(cov_ref=0.9, cov_casc=0.88, deltas=[20, 22, 19, 21, 18],
                          published_cascade_total=150.0, served_total=100.0, audit_extra=50.0,
                          idle_pre_watts=0.66, idle_post_watts=0.69)
    assert good["verdict"] == "VALID" and good["significant_saving_J"] is not None

    # one gate red (noise-sized saving) -> no number
    bad = evaluate_gates(cov_ref=0.9, cov_casc=0.88, deltas=[1, -2, 3, -1, 2],
                         published_cascade_total=150.0, served_total=100.0, audit_extra=50.0,
                         idle_pre_watts=0.66, idle_post_watts=0.69)
    assert bad["verdict"] == "GATED_NO_NUMBER" and bad["significant_saving_J"] is None
