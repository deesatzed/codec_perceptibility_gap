"""Gate 2 — collinearity guard (the false-positive catcher).

With 2 decoder families, "majority wrong" is algebraically identical to
"difficulty >= 0.5", so the difficulty-controlled partial is undefined/degenerate
— that is exactly how the earlier 2-family precision=1.0 false positive slipped
through. This gate refuses to compute the partial unless there are >=3 families
AND difficulty and ensemble-wrong are genuinely distinct variables.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.crc.eval import status as st

MIN_FAMILIES = 3
CORR_BOUND = 0.95


def collinearity_guard(n_families: int, difficulty: List[float],
                       ensemble_wrong: List[float], corr_bound: float = CORR_BOUND) -> Dict[str, Any]:
    """Returns {status, corr, gate_pass}. corr = |corr(difficulty, ensemble_wrong)|."""
    if n_families < MIN_FAMILIES:
        return {"status": st.COLLINEAR_UNVALIDATED, "corr": None, "gate_pass": False,
                "reason": f"needs >= {MIN_FAMILIES} families (got {n_families}); "
                          f"with fewer, majority-wrong == difficulty"}
    d = np.asarray(difficulty, dtype=float)
    w = np.asarray(ensemble_wrong, dtype=float)
    if np.std(d) < 1e-12 or np.std(w) < 1e-12:
        return {"status": st.COLLINEAR_UNVALIDATED, "corr": None, "gate_pass": False,
                "reason": "degenerate variance (difficulty or wrong is constant)"}
    c = abs(float(np.corrcoef(d, w)[0, 1]))
    if not np.isfinite(c) or c >= corr_bound:
        return {"status": st.COLLINEAR_UNVALIDATED, "corr": (round(c, 3) if np.isfinite(c) else None),
                "gate_pass": False, "reason": "difficulty ~= ensemble-wrong; partial undefined"}
    return {"status": st.OK, "corr": round(c, 3), "gate_pass": True}
