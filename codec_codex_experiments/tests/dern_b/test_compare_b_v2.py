"""Tests for the paired/gated v2 comparison harness (ultrathink mitigation)."""
import pytest

from src.dern_b.compare_b_v2 import (
    run_comparison_v2, summarize_classes, PairTrial, _classify,
)
from src.dern_b.energy_probe import sudo_available


def test_no_sudo_returns_gated_no_number():
    # Without sudo, the harness must refuse to emit a number (honest failure).
    if not sudo_available():
        out = run_comparison_v2(["What is 2 plus 2?"], K=5, max_tokens=8)
        assert out["verdict"] == "GATED_NO_NUMBER"
        assert "measured_paired_saving_J_per_round" not in out or \
               out.get("measured_paired_saving_J_per_round") is None


def test_classify_route_class():
    assert _classify({"forced_ref": True, "replay_route": False}) == "forced_ref"
    assert _classify({"forced_ref": False, "replay_route": True}) == "replay"
    assert _classify({"forced_ref": False, "replay_route": False}) == "first_encounter"


def test_summarize_separates_first_encounter_from_replay():
    # first-encounter cascade must be reportable as MORE expensive than reference
    # (it ran both models). This is the no-free-savings invariant at the summary level.
    trials = [
        PairTrial("p1", "first_encounter", True, ref_joules=10.0, casc_joules=13.0,
                  delta_joules=-3.0, ref_cov=0.9, casc_cov=0.9),
        PairTrial("p2", "replay", True, ref_joules=10.0, casc_joules=4.0,
                  delta_joules=6.0, ref_cov=0.9, casc_cov=0.9),
    ]
    s = summarize_classes(trials)
    assert s["first_encounter"]["n"] == 1 and s["replay"]["n"] == 1
    # first-encounter cascade (13) > its reference (10): audit double-run charged
    assert s["first_encounter"]["mean_cascade_J"] > s["first_encounter"]["mean_reference_J"]


@pytest.mark.heavy
def test_v2_runs_end_to_end_and_gates(monkeypatch):
    # Real models + (if primed) sudo. Asserts the harness produces a verdict and,
    # when VALID, a number; when GATED, no number — never a silent stretch.
    out = run_comparison_v2(["What is the capital of France? One word.",
                             "What is the capital of Japan? One word."],
                            K=5, max_tokens=24)
    assert out["verdict"] in {"VALID", "GATED_NO_NUMBER"}
    if out["verdict"] == "GATED_NO_NUMBER":
        assert out.get("measured_paired_saving_J_per_round") is None
