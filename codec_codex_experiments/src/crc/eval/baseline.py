"""Gate 4 — naive difficulty baseline (per arXiv 2509.19372).

The bar every label-free signal must clear: can question DIFFICULTY alone predict
wrongness? A signal "wins" only if its AUC beats the difficulty-only AUC by more
than noise (bootstrap CI of the gap clears zero) — not a point estimate.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _auc(scores: np.ndarray, wrong: np.ndarray) -> float:
    if len(np.unique(wrong)) < 2:
        return float("nan")
    return float(roc_auc_score(wrong.astype(int), scores))


def beats_difficulty_baseline(signal_scores, difficulty, wrong, n_boot: int = 300,
                              seed: int = 17) -> Dict[str, Any]:
    s = np.asarray(signal_scores, float)
    d = np.asarray(difficulty, float)
    w = np.asarray(wrong, float)
    auc_s = _auc(s, w)
    auc_b = _auc(d, w)
    if not (np.isfinite(auc_s) and np.isfinite(auc_b)):
        return {"auc_signal": None, "auc_baseline": None, "gap_ci95": None, "wins": False}
    rng = np.random.default_rng(seed)
    n = len(w)
    gaps = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(w[idx])) < 2:
            continue
        gaps.append(_auc(s[idx], w[idx]) - _auc(d[idx], w[idx]))
    if not gaps:
        return {"auc_signal": round(auc_s, 3), "auc_baseline": round(auc_b, 3),
                "gap_ci95": None, "wins": False}
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "auc_signal": round(auc_s, 3), "auc_baseline": round(auc_b, 3),
        "gap_mean": round(float(np.mean(gaps)), 3),
        "gap_ci95": [round(float(lo), 3), round(float(hi), 3)],
        "wins": bool(lo > 0.0),   # CI strictly above zero => beats difficulty beyond noise
    }
