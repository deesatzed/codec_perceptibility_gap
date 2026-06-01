"""Correlated-error guardrail: mean pairwise correlation of per-question
right/wrong across family pairs. High => families fail together => agreement is
hollow. Mirrors codec_contest.mean_pairwise_codec_error_correlation."""
from __future__ import annotations

import numpy as np


def pairwise_error_correlation(family_errors: np.ndarray) -> float:
    """family_errors: (n_families, n_questions) of 0/1 wrong flags."""
    e = np.asarray(family_errors, dtype=float)
    if e.shape[0] < 2:
        return float("nan")
    # guard against a family with zero variance (all right or all wrong)
    with np.errstate(invalid="ignore"):
        cm = np.corrcoef(e)
    mask = ~np.eye(cm.shape[0], dtype=bool)
    offdiag = cm[mask]
    if not np.any(np.isfinite(offdiag)):
        return float("nan")          # no defined pairwise correlation (all zero-variance)
    return float(np.nanmean(offdiag))
