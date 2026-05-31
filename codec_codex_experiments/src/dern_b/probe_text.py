"""Distinction probe over a prompt: cheap, deterministic, non-semantic key.

No model call. Quantizes a few surface cues into a small int tuple — an opaque
region label the experience graph keys on. The key's only job is to group
prompts that tend to need the same compute tier (short factual vs long
reasoning), which it does without any human-semantic interpretation.
"""
from __future__ import annotations

import re

_DIGIT = re.compile(r"\d")
_CODE = re.compile(r"[{};=()\[\]]|def |class |import |```")
_QUESTION = re.compile(r"\b(why|how|explain|prove|derive|reason|step)\b", re.I)


def _len_bucket(n_words: int) -> int:
    for i, edge in enumerate((8, 24, 64)):
        if n_words <= edge:
            return i
    return 3


def distinction_key(prompt: str) -> tuple:
    words = prompt.split()
    return (
        _len_bucket(len(words)),
        1 if _DIGIT.search(prompt) else 0,
        1 if _CODE.search(prompt) else 0,
        1 if _QUESTION.search(prompt) else 0,
    )


def probe_cost_units() -> float:
    return 0.001  # pure-Python surface scan; negligible vs a model forward pass
