"""Phase-1 calibrator. Proves disagreement predicts wrongness BEFORE a threshold
is issued: correlation + bootstrap CI + permutation null. Surfaces the
correlated-error (hollow-agreement) caveat. Refuses (UNVALIDATED) if unproven."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.codec_contest import bootstrap_corr_ci

MIN_N = 30                 # variance gate: corr on a handful of points is noise
HOLLOW_AGREEMENT = 0.5     # pairwise family-error corr above this => agreement suspect


def _perm_null_p95(d: np.ndarray, w: np.ndarray, n_perm: int = 500, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        vals.append(float(np.corrcoef(d[rng.permutation(len(d))], w)[0, 1]))
    return float(np.percentile(vals, 95))


def _best_threshold(d: np.ndarray, w: np.ndarray):
    """Sweep candidate thresholds; pick the one maximizing F1 of (d>=t) vs wrong."""
    best = (None, -1.0, 0.0, 0.0)
    for t in np.quantile(d, np.linspace(0.1, 0.9, 17)):
        flag = d >= t
        tp = float((flag & (w > 0)).sum())
        fp = float((flag & (w == 0)).sum())
        fn = float((~flag & (w > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (float(t), f1, prec, rec)
    return best


def calibrate(disagree: np.ndarray, wrong: np.ndarray, pairwise_err_corr: float) -> Dict[str, Any]:
    d = np.asarray(disagree, dtype=float)
    w = np.asarray(wrong, dtype=float)
    hollow = bool(pairwise_err_corr >= HOLLOW_AGREEMENT)
    if len(d) < MIN_N or len(np.unique(w)) < 2:
        return {"verdict": "UNVALIDATED",
                "reason": f"need >= {MIN_N} pts and both right+wrong present",
                "threshold": None, "hollow_agreement_warning": hollow}
    corr = float(np.corrcoef(d, w)[0, 1])
    ci = bootstrap_corr_ci(d, w, n_boot=300)
    null_p95 = _perm_null_p95(d, w)
    clears_null = bool(corr > null_p95 and ci["ci95"][0] > 0)
    if not clears_null:
        return {"verdict": "UNVALIDATED",
                "reason": "disagreement does not predict wrongness (fails permutation null/CI)",
                "corr": round(corr, 3), "null_p95": round(null_p95, 3), "ci95": ci["ci95"],
                "threshold": None, "hollow_agreement_warning": hollow}
    t, f1, prec, rec = _best_threshold(d, w)
    return {
        "verdict": "VALID",
        "corr": round(corr, 3), "null_p95": round(null_p95, 3), "ci95": ci["ci95"],
        "threshold": round(t, 4), "f1": round(f1, 3),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "calibration_range": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "pairwise_error_corr": round(float(pairwise_err_corr), 3),
        "hollow_agreement_warning": hollow,
    }
