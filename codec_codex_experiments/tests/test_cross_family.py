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


def test_verification_tracks_independent():
    """critic1 #1: the three lanes are reported and each is independent. A lane's
    pass status must depend only on its own gates, not on other lanes."""
    cfg = _cfg()
    rep = run_ladder(cfg)
    tracks = rep["tracks"]
    assert set(tracks) == {"information_theoretic", "complexity_scaling", "auditability_verification"}
    # union of lane gates == all ladder gates, disjoint
    seen = []
    for t in tracks.values():
        seen += t["gates"]
        # lane_passed == all its own gates passed
        assert t["lane_passed"] == all(t["per_gate_passed"].values())
        assert t["n_passed"] == sum(t["per_gate_passed"].values())
    assert sorted(seen) == sorted(LADDER_GATES)
    assert len(seen) == len(set(seen)), "a gate is assigned to more than one lane"


def test_henon_family_registered_and_bounded():
    """critic1 #3: the discrete-time chaotic family exists, stays bounded (no
    mass saturation at the clip wall), and never leaks its target."""
    import numpy as np
    cfg = _cfg()
    h = SOURCE_REGISTRY["henon_map"]
    traj, theta = h.sample(cfg["n_train"], cfg, 0)
    assert np.isfinite(traj).all()
    sat = (np.abs(traj) >= 9.99).mean()
    assert sat < 0.10, f"henon orbits saturate too often ({sat:.3f}); coupling unstable"
    # leakage guard (mirrors test_sources, kept here so the family can't be added
    # to the cross-family matrix without this check)
    for feats in (h.native_channel(traj), h.expanded_physics_channel(traj)):
        for j in range(feats.shape[1]):
            if np.std(feats[:, j]) < 1e-12:
                continue
            for t in range(theta.shape[1]):
                if np.std(theta[:, t]) < 1e-12:
                    continue
                assert abs(np.corrcoef(feats[:, j], theta[:, t])[0, 1]) < 0.999


def test_cross_family_audit_survival_reported():
    """critic1 #3 headline: the cross-family bundle reports CodecGuard audit
    survival per family, including the non-oscillatory henon family."""
    cfg = _cfg(sources=["linear_oscillator", "henon_map"])
    res = run_cross_family(cfg)
    surv = res["codecguard_audit_survival"]
    assert set(surv) == {"linear_oscillator", "henon_map"}
    for fam, s in surv.items():
        assert isinstance(s["audit_signal_survives"], bool)
        assert "loco_corr" in s and "catch_rate" in s


def test_nonlinear_s7_nondegenerate():
    """The nonlinear family must produce a non-trivial S7 slope (not all-equal
    error across dimensions). This is the F5 motivation: the linear family could
    not generate a difficulty gradient. We assert the slope spans a real range,
    NOT that the gate passes (gate outcome is reported honestly, not forced)."""
    cfg = _cfg()
    rep = run_ladder(cfg, source=SOURCE_REGISTRY["nonlinear_oscillator"])
    trained = [rep["S7"]["slope"][d]["trained"] for d in ("4D", "6D", "8D")]
    assert max(trained) - min(trained) > 0.02, f"S7 slope degenerate: {trained}"
