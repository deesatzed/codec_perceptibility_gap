import pytest
from src.dern_b.prompts import REPEATED_REGION_STREAM
from src.dern_b.probe_text import distinction_key


def test_repeated_region_stream_actually_repeats_regions():
    # The whole point of the replay test: many prompts must SHARE a probe key,
    # else the experience graph never replays and amortization can't be shown.
    keys = [distinction_key(p) for p in REPEATED_REGION_STREAM]
    n_unique = len(set(keys))
    n_total = len(keys)
    # there must be substantially fewer regions than prompts (real repetition)
    assert n_unique < n_total * 0.6, f"{n_unique} regions for {n_total} prompts — not enough repetition"


def test_compare_module_imports_and_exposes_api():
    import src.dern_b.compare_b as c
    assert hasattr(c, "run_comparison")
    assert hasattr(c, "always_reference_pass")
    assert hasattr(c, "cascade_pass")
    assert hasattr(c, "epsilon_sweep")


@pytest.mark.heavy
def test_comparison_runs_and_baseline_not_below_cascade_on_aps():
    # Real models, aps-only (no sudo): the always-reference baseline aps must be
    # >= cascade served aps (cascade serves cheap on >=1 prompt -> less served aps).
    import src.dern_b.compare_b as c
    out = c.run_comparison(prompts=REPEATED_REGION_STREAM[:6], max_tokens=32,
                           epsilons=[0.4], seed=0, measure_joules=False)
    assert out["cascade_detail"]["served_worse_than_reference"] == 0
    assert out["baseline_active_param_seconds"] >= out["cascade_active_param_seconds"]
