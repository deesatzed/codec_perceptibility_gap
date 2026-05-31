"""Unmodifiable verifier: the only authority that can declare a config safe.

Lane 1 (exact): a config passes only if its recovered distinctions are
byte-identical to full compute (delta == 0). This models speculative decoding's
exact verification: a non-full config can only pass if it happens to produce the
identical result, which for distinct configs it will not -> it is rejected and
the caller falls back to full compute. Loss is impossible.

Lane 2 (bounded): a config passes iff its recovery error exceeds full compute's
by at most epsilon. This is the sampled full-compute spot-check.

Both functions are PURE w.r.t. controller state — they accept no controller or
policy argument, so the learner cannot influence the verdict. This is enforced
by test_verifier_signature_takes_no_controller_state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.dern.configs import full_config
from src.dern.oracle import recover_distinctions, recovery_error


@dataclass(frozen=True)
class Verdict:
    passed: bool
    lane: int
    delta: float
    epsilon: Optional[float]
    proof: str  # "exact" | "bounded" | "rejected"


def verify_lane1(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int) -> Verdict:
    _, pred_full = recover_distinctions(src, cfg, full_config(), seed)
    _, pred_cfg = recover_distinctions(src, cfg, config, seed)
    identical = bool(np.array_equal(pred_full, pred_cfg))
    return Verdict(
        passed=identical, lane=1,
        delta=0.0 if identical else float(np.abs(pred_full - pred_cfg).mean()),
        epsilon=None, proof="exact" if identical else "rejected",
    )


def verify_lane2(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int, epsilon: float) -> Verdict:
    full_err = recovery_error(src, cfg, full_config(), seed)
    cfg_err = recovery_error(src, cfg, config, seed)
    delta = float(cfg_err - full_err)
    passed = bool(delta <= epsilon)
    return Verdict(
        passed=passed, lane=2, delta=delta, epsilon=float(epsilon),
        proof="bounded" if passed else "rejected",
    )
