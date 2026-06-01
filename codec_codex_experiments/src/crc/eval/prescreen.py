"""Gate 1 — benchmark error-rate pre-screen (the primary flaw-catcher).

Both prior CRC runs were misled by ERROR SCARCITY: on ~6%-error data nothing
discriminates, so any signal looks non-predictive (false negative). This gate
REFUSES to report a verdict unless the ensemble actually errs enough to test
against. A refusal is the honest deliverable, not a number.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.crc.eval import status as st

MIN_N = 30
ERROR_FLOOR = 0.30          # need >=30% wrong, or there's nothing to predict
ERROR_CEIL = 0.95           # ~all wrong is also degenerate


def prescreen_error_rate(ensemble_wrong: List[float], threshold: float = ERROR_FLOOR) -> Dict[str, Any]:
    """ensemble_wrong: 0/1 per question (1 = ensemble got it wrong).
    Returns {status, error_rate, n, gate_pass}. error_rate is None unless computable."""
    w = np.asarray(ensemble_wrong, dtype=float)
    n = len(w)
    if n < MIN_N or len(np.unique(w)) < 2:
        return {"status": st.INSUFFICIENT_N, "error_rate": None, "n": n, "gate_pass": False}
    err = float(np.mean(w))
    if err < threshold or err > ERROR_CEIL:
        return {"status": st.ERROR_SCARCE_UNVALIDATED, "error_rate": round(err, 3),
                "n": n, "gate_pass": False}
    return {"status": st.OK, "error_rate": round(err, 3), "n": n, "gate_pass": True}
