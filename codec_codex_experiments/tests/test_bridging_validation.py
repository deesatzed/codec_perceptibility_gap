"""Tests for the bridging validation (critic1 criticism 2).

Guards two properties: (1) it never touches the reserved confirmatory holdout,
and (2) it actually compares both estimator classes and emits the margin-
sensitivity verdict the critique asked for.
"""
from src.proof_ladder import normalize_cfg
from src.bridging_validation import run_bridging_validation


def _cfg(**kw):
    base = {
        "n_train": 200,
        "n_test": 90,
        "T": 6.0,
        "dt": 0.1,
        "seeds": [0, 1],
        "sources": ["linear_oscillator", "nonlinear_oscillator"],
        "delta_eq": 0.04,
        "mlp_hidden": [32],
        "mlp_iter": 120,
    }
    base.update(kw)
    return normalize_cfg(base)


def test_bridging_never_touches_holdout():
    res = run_bridging_validation(_cfg())
    assert res["holdout_touched"] is False


def test_bridging_compares_both_estimators():
    res = run_bridging_validation(_cfg())
    assert res["estimators_compared"] == ["ridge", "mlp"]
    for fam, v in res["per_family"].items():
        be = v["by_estimator"]
        assert "ridge" in be and "mlp" in be
        for est in ("ridge", "mlp"):
            assert "direct_ceiling_mean" in be[est]
            assert "throughput_delta_max" in be[est]
        assert isinstance(v["margin_fragile"], bool)
        assert "ceiling_shift_ridge_to_mlp" in v


def test_bridging_emits_aggregate_verdict():
    res = run_bridging_validation(_cfg())
    assert isinstance(res["any_margin_fragile"], bool)
    assert res["max_abs_ceiling_shift"] >= 0.0
