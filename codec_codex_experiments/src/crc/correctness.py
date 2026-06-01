"""Phase-1 correctness vs an answer key. Crisp answers: normalized exact or
substring match. (Embedding-similarity matching deferred until needed.)"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_correct(answer: str, key: str) -> bool:
    a, k = _norm(answer), _norm(key)
    if not k:
        return False
    return k == a or k in a.split() or k in a
