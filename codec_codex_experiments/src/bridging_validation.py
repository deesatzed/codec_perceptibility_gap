#!/usr/bin/env python3
"""Bridging validation (critic1 criticism 2).

The critique flagged an architectural blind spot: the developmental phase uses
simple estimators (Ridge / KNN) but the confirmatory phase will introduce
nonlinear estimators (MLP / small transformer) that are *not yet run*. A much
stronger estimator can shift the direct machine ceiling left and make the
throughput-invariance margin (``delta_eq``, the absolute nRMSE band, the TOST
analog) either trivially easy or unreachable -- a calibration risk to lock into
a pre-registration.

This module derisks that *without touching the reserved confirmatory holdout*.
It re-runs two ladder probes -- S0 (direct machine ceiling) and S2
(throughput-invariance margin) -- under both the developmental estimator
(``ridge``) and the proposed confirmatory estimator (``mlp``, already wired in
``fit_predict``), on the EXPOSED developmental source families only. It reports:

  * the provisional machine ceiling per estimator, and how far it moves; and
  * the S2 throughput-invariance delta per estimator, so we can see whether the
    chosen ``delta_eq`` is comfortable or marginal once the estimator is upgraded.

No new data are drawn from any held-out set; the same ``make_dataset`` developmental
sampling used everywhere else is reused. The estimator is the ONLY thing that
changes between the paired runs.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .proof_ladder import (
    make_dataset,
    native_feats,
    direct_feats,
    quantize,
    fit_predict,
    nrmse,
    normalize_cfg,
)
from .sources import resolve_sources


def _direct_ceiling(cfg: Dict[str, Any], source: Any, seed: int) -> float:
    """S0-style direct machine ceiling: best recovery from raw down-sampled
    observables under the active estimator."""
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed, source=source)
    yp = fit_predict(direct_feats(xtr), ttr, direct_feats(xte), cfg, seed)
    return nrmse(tte, yp)


def _throughput_delta(cfg: Dict[str, Any], source: Any, seed: int) -> Dict[str, float]:
    """S2-style throughput-invariance probe: replicate the same quantized native
    channel 1x vs 8x (redundant bits, no new information) and measure the nRMSE
    gap. A faithful estimator should be ~invariant; the gap is what ``delta_eq``
    must comfortably exceed."""
    xtr, ttr, xte, tte = make_dataset(2, cfg, seed, source=source)
    Ntr, Nte = native_feats(xtr), native_feats(xte)
    Ftr = quantize(Ntr, 8, Ntr)
    Fte = quantize(Nte, 8, Ntr)

    def replicate(F: np.ndarray, m: int) -> np.ndarray:
        rng = np.random.default_rng(100 + m)
        return np.concatenate([F + 0.01 * rng.standard_normal(F.shape) for _ in range(m)], axis=1)

    t1 = nrmse(tte, fit_predict(replicate(Ftr, 1), ttr, replicate(Fte, 1), cfg, seed))
    t8 = nrmse(tte, fit_predict(replicate(Ftr, 8), ttr, replicate(Fte, 8), cfg, seed))
    return {"thru1x": float(t1), "thru8x": float(t8), "delta": float(abs(t1 - t8))}


def run_bridging_validation(raw_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compare ridge (developmental) vs mlp (proposed confirmatory) on every
    exposed developmental family. Returns ceiling-shift and margin-sensitivity
    evidence. The reserved confirmatory holdout is never read."""
    cfg = normalize_cfg(raw_cfg)
    seeds = [int(s) for s in cfg["seeds"]]
    delta_eq = float(cfg.get("delta_eq", 0.04))

    families = [f for f in resolve_sources(cfg) if getattr(f, "target_kind", "continuous") == "continuous"]
    # Always include the linear family even if cfg lists only one source.
    keys_seen = {f.key for f in families}
    if not families:
        families = resolve_sources(normalize_cfg({"source": "linear_oscillator"}))

    estimators = ["ridge", "mlp"]
    per_family: Dict[str, Any] = {}
    for fam in families:
        per_est: Dict[str, Any] = {}
        for est in estimators:
            ecfg = dict(cfg)
            ecfg["estimator"] = est
            ceilings = [_direct_ceiling(ecfg, fam, s) for s in seeds]
            deltas = [_throughput_delta(ecfg, fam, s) for s in seeds]
            per_est[est] = {
                "direct_ceiling_mean": round(float(np.mean(ceilings)), 4),
                "direct_ceiling_sd": round(float(np.std(ceilings)), 4),
                "throughput_delta_mean": round(float(np.mean([d["delta"] for d in deltas])), 4),
                "throughput_delta_max": round(float(np.max([d["delta"] for d in deltas])), 4),
                "throughput_delta_within_delta_eq": bool(
                    np.max([d["delta"] for d in deltas]) <= delta_eq
                ),
            }
        ridge_c = per_est["ridge"]["direct_ceiling_mean"]
        mlp_c = per_est["mlp"]["direct_ceiling_mean"]
        per_family[fam.key] = {
            "by_estimator": per_est,
            "ceiling_shift_ridge_to_mlp": round(mlp_c - ridge_c, 4),
            "ceiling_shift_pct": round(100.0 * (mlp_c - ridge_c) / (ridge_c + 1e-9), 1),
            # The margin is "fragile" if upgrading the estimator flips whether the
            # throughput delta stays inside delta_eq, OR moves the delta by more
            # than half the band -- i.e. delta_eq is sensitive to estimator class.
            "margin_fragile": bool(
                per_est["ridge"]["throughput_delta_within_delta_eq"]
                != per_est["mlp"]["throughput_delta_within_delta_eq"]
                or abs(per_est["mlp"]["throughput_delta_mean"] - per_est["ridge"]["throughput_delta_mean"])
                > 0.5 * delta_eq
            ),
        }

    any_fragile = any(v["margin_fragile"] for v in per_family.values())
    max_shift = max((abs(v["ceiling_shift_ridge_to_mlp"]) for v in per_family.values()), default=0.0)
    return {
        "delta_eq": delta_eq,
        "estimators_compared": estimators,
        "per_family": per_family,
        "any_margin_fragile": bool(any_fragile),
        "max_abs_ceiling_shift": round(float(max_shift), 4),
        "holdout_touched": False,
        "interpretation": (
            "Provisional machine ceiling under the proposed confirmatory estimator, "
            "established on EXPOSED developmental families only (no holdout peeking). "
            "If `any_margin_fragile` is true, the throughput-invariance band (delta_eq) "
            "is sensitive to estimator class and should be re-derived from the "
            "confirmatory estimator's empirical floor before freezing pre-registration, "
            "NOT carried over from the ridge developmental floor."
        ),
    }
