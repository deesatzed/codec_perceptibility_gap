"""Full-compute oracle + recovery scoring on synthetic agents.

Ground-truth distinctions are the sampled theta. recovery_error runs a config's
channel through the existing proof-ladder estimator and returns nRMSE vs theta.
This is the 'did we resolve the distinctions' measure the verifier trusts.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.proof_ladder import (
    native_feats,
    engineered_feats,
    direct_feats,
    quantize,
    fit_predict,
    nrmse,
)

_CHANNELS = {
    "native": native_feats,
    "engineered": engineered_feats,
    "direct": direct_feats,
}


def _features(traj: np.ndarray, channel: str) -> np.ndarray:
    return _CHANNELS[channel](traj)


def recover_distinctions(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int):
    """Return (theta_test, predicted_theta) for a config on a source family."""
    xtr, ttr = src.sample(int(cfg["n_train"]), cfg, seed)
    xte, tte = src.sample(int(cfg["n_test"]), cfg, seed + 500)
    Ftr = _features(xtr, config["channel"])
    Fte = _features(xte, config["channel"])
    k = int(config["k"])
    Qtr = quantize(Ftr, k, Ftr)
    Qte = quantize(Fte, k, Ftr)
    pred = fit_predict(Qtr, ttr, Qte, cfg, seed)
    return tte, pred


def recovery_error(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int) -> float:
    tte, pred = recover_distinctions(src, cfg, config, seed)
    return nrmse(tte, pred)
