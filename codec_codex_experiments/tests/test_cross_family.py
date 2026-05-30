"""Phase 2-5 tests for cross-family generalization (F5 mitigation)."""
from src.proof_ladder import normalize_cfg, run_ladder
from src.cross_family import run_cross_family, LADDER_GATES
from src.sources import SOURCE_REGISTRY


def _cfg(**kw):
    base = {"n_train": 200, "n_test": 90, "T": 6.0, "dt": 0.1, "seeds": [0, 1], "estimator": "ridge"}
    base.update(kw)
    return normalize_cfg(base)


def test_ladder_runs_per_family():
    cfg = _cfg()
    for key, src in SOURCE_REGISTRY.items():
        if getattr(src, "target_kind", "continuous") != "continuous":
            continue
        rep = run_ladder(cfg, source=src)
        assert {"S0", "S8"}.issubset(rep.keys())
        assert isinstance(rep["S0"]["passed"], bool)


def test_cross_family_matrix_wellformed():
    cfg = _cfg(sources=["linear_oscillator", "nonlinear_oscillator"])
    res = run_cross_family(cfg)
    assert res["families_run"] == ["linear_oscillator", "nonlinear_oscillator"]
    gm = res["gate_matrix"]
    per_gate = gm["per_gate_pass_by_family"]
    # every gate present, every family has a boolean
    for gate in LADDER_GATES:
        assert gate in per_gate
        for fam in res["families_run"]:
            assert isinstance(per_gate[gate][fam], bool)
    # partition is exhaustive and disjoint
    robust = set(gm["family_robust_gates"])
    specific = set(gm["family_specific_gates"])
    failing = set(gm["failing_on_all_families"])
    assert robust | specific | failing == set(LADDER_GATES)
    assert not (robust & specific) and not (robust & failing) and not (specific & failing)


def test_nonlinear_s7_nondegenerate():
    """The nonlinear family must produce a non-trivial S7 slope (not all-equal
    error across dimensions). This is the F5 motivation: the linear family could
    not generate a difficulty gradient. We assert the slope spans a real range,
    NOT that the gate passes (gate outcome is reported honestly, not forced)."""
    cfg = _cfg()
    rep = run_ladder(cfg, source=SOURCE_REGISTRY["nonlinear_oscillator"])
    trained = [rep["S7"]["slope"][d]["trained"] for d in ("4D", "6D", "8D")]
    assert max(trained) - min(trained) > 0.02, f"S7 slope degenerate: {trained}"
