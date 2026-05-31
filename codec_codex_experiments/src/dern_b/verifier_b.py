"""Bounded-loss verifier (Lane 2): the reference model's answer is the authority.

A cheap answer passes iff its agreement with the reference answer is >= (1 - eps)
on a task-appropriate metric. PURE w.r.t. controller state (no controller/policy
argument) — the learner cannot influence the verdict. This is the runtime
authority that mints reward and trust.

NOTE: this is Lane 2 (bounded). Lane 1 (exact, byte-identical via white-box token
acceptance) is deferred — it needs control the local mlx generate loop does not
currently expose at the token-acceptance level for cross-model speculation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class VerdictB:
    passed: bool
    delta: float          # 1 - agreement
    epsilon: float
    metric: str
    proof: str            # "bounded" | "rejected"


def _tokens(s: str) -> list:
    return _WORD.findall(s.lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(tb)
    rec = overlap / len(ta)
    return 2 * prec * rec / (prec + rec)


def verify_against_reference(cheap_text: str, ref_text: str, epsilon: float,
                             metric: str = "token_f1") -> VerdictB:
    if metric == "exact_match":
        agreement = 1.0 if cheap_text.strip().lower() == ref_text.strip().lower() else 0.0
    else:
        agreement = token_f1(cheap_text, ref_text)
    delta = 1.0 - agreement
    passed = bool(delta <= epsilon)
    return VerdictB(
        passed=passed, delta=float(delta), epsilon=float(epsilon),
        metric=metric, proof="bounded" if passed else "rejected",
    )
