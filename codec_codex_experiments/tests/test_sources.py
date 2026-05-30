"""Phase 1+ tests for the SourceFamily seam (F5 mitigation).

`test_no_target_leakage` is the critical guard: it makes the F1 leakage class
un-reintroducible as new families are added.
"""
import numpy as np

from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY, resolve_sources


def _cfg():
    return normalize_cfg({"n_train": 120, "n_test": 60, "T": 6.0, "dt": 0.1, "seeds": [0]})


def test_source_registry_shapes():
    cfg = _cfg()
    for key, src in SOURCE_REGISTRY.items():
        traj, theta = src.sample(cfg["n_train"], cfg, 0)
        assert traj.ndim == 3, f"{key}: trajectory must be (B,T,N)"
        assert traj.shape[0] == cfg["n_train"], f"{key}: batch mismatch"
        assert theta.shape[0] == cfg["n_train"], f"{key}: theta batch mismatch"
        assert theta.shape[1] == len(src.target_labels), (
            f"{key}: target_labels ({len(src.target_labels)}) != theta dim ({theta.shape[1]})"
        )
        assert src.target_kind in {"continuous", "categorical", "mixed"}


def test_no_target_leakage():
    """No native/expanded_physics channel column may reproduce a target column.

    Enforces the F1 anti-leakage invariant for every registered family. For
    continuous targets we check max |corr| < 0.999; an exact-copy leak would
    correlate ~1.0.
    """
    cfg = _cfg()
    for key, src in SOURCE_REGISTRY.items():
        traj, theta = src.sample(cfg["n_train"], cfg, 0)
        if src.target_kind != "continuous":
            # Categorical families validated in their own module; skip corr test.
            continue
        channels = {
            "native": src.native_channel(traj),
            "expanded_physics": src.expanded_physics_channel(traj),
        }
        for cname, feats in channels.items():
            for j in range(feats.shape[1]):
                fcol = feats[:, j]
                if np.std(fcol) < 1e-12:
                    continue
                for t in range(theta.shape[1]):
                    tcol = theta[:, t]
                    if np.std(tcol) < 1e-12:
                        continue
                    r = abs(np.corrcoef(fcol, tcol)[0, 1])
                    assert r < 0.999, (
                        f"{key}.{cname}[{j}] leaks target '{src.target_labels[t]}' (|corr|={r:.4f})"
                    )


def test_resolve_sources_default_linear():
    cfg = _cfg()
    srcs = resolve_sources(cfg)
    assert [s.key for s in srcs] == ["linear_oscillator"]
