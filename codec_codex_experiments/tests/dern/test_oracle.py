import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.configs import COMPUTE_CONFIGS, full_config, compute_cost_units
from src.dern.oracle import recover_distinctions, recovery_error


def _cfg():
    return normalize_cfg({"n_train": 300, "n_test": 120, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_configs_ordered_by_cost():
    costs = [compute_cost_units(c) for c in COMPUTE_CONFIGS]
    assert costs == sorted(costs), "configs must be listed cheapest->most expensive"
    assert compute_cost_units(full_config()) == max(costs), "full config is most expensive"


def test_full_config_recovers_easy_family_well():
    cfg = _cfg()
    src = SOURCE_REGISTRY["linear_oscillator"]
    err = recovery_error(src, cfg, full_config(), seed=0)
    assert err < 0.6, f"full compute should recover linear distinctions (nrmse={err:.3f})"


def test_cheaper_config_no_better_than_full_on_easy():
    cfg = _cfg()
    src = SOURCE_REGISTRY["linear_oscillator"]
    full_err = recovery_error(src, cfg, full_config(), seed=0)
    cheap_err = recovery_error(src, cfg, COMPUTE_CONFIGS[0], seed=0)
    assert cheap_err + 1e-9 >= full_err - 0.25, "cheap must not dramatically beat full"
