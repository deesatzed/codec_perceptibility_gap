from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.run_stage_a import run_stage_a


def test_controller_converges_below_baseline_on_easy_family():
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    report = run_stage_a(family="linear_oscillator", n_requests=120, cfg=cfg,
                         eps_tolerance=0.5, seed=0)
    assert report["mean_net_savings"] > 0.0
    assert report["unverified_serves"] == 0
    assert report["lane1_exactness_violations"] == 0


def test_hard_family_stays_safe_even_if_no_savings():
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    report = run_stage_a(family="henon_map", n_requests=80, cfg=cfg,
                         eps_tolerance=0.3, seed=0)
    assert report["unverified_serves"] == 0
    assert report["lane1_exactness_violations"] == 0
