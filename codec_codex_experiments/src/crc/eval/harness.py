"""Difficulty-disentangled evaluation harness. For a label-free signal, computes:
raw corr with wrongness, partial corr controlling for difficulty (reused from
two_arm), and whether it beats the naive difficulty baseline (per 2509.19372).
All gated: error-scarce or collinear inputs -> typed status, never a number.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.crc.two_arm import disagreement_beyond_difficulty
from src.crc.eval.prescreen import prescreen_error_rate
from src.crc.eval.collinearity import collinearity_guard
from src.crc.eval.baseline import beats_difficulty_baseline
from src.crc.eval import status as st


def evaluate_signal(signal_scores: List[float], difficulty: List[float],
                    wrong: List[float], n_families: int) -> Dict[str, Any]:
    """One shared evaluator every signal routes through (fairness). Emits a typed
    status when a gate fails; a numeric verdict only when all gates pass."""
    pre = prescreen_error_rate(wrong)
    if not pre["gate_pass"]:
        return {"verdict": pre["status"], "error_rate": pre.get("error_rate"),
                "partial_corr": None, "baseline": None}
    col = collinearity_guard(n_families, difficulty, wrong)
    if not col["gate_pass"]:
        return {"verdict": col["status"], "error_rate": pre["error_rate"],
                "partial_corr": None, "baseline": None, "collinearity": col}

    ctrl = disagreement_beyond_difficulty(np.asarray(signal_scores), np.asarray(wrong),
                                          np.asarray(difficulty))
    base = beats_difficulty_baseline(signal_scores, difficulty, wrong)
    partial = ctrl["partial_corr_controlling_difficulty"]

    if base["wins"] and partial is not None and partial > 0.0:
        verdict = st.REAL_SIGNAL
    elif partial is not None:
        verdict = st.DIFFICULTY_PROXY
    else:
        verdict = st.NO_SIGNAL
    return {
        "verdict": verdict,
        "error_rate": pre["error_rate"],
        "raw_corr": ctrl["raw_corr_disagree_wrong"],
        "partial_corr": partial,
        "baseline": base,
    }
