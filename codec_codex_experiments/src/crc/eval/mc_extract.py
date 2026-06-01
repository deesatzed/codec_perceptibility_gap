"""Robust multiple-choice letter extraction from verbose model output.

Models answer verbosely and in different reasoning-trace formats (granite ignores
'one word'; gemma/qwen emit thinking channels). We first strip the trace via the
generalized extract_final_answer, then pull the chosen option letter. Returns None
on ambiguity; None is scored wrong AND counted in an unextractable_rate that is
reported, never hidden.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.dern_b.mlx_backend import extract_final_answer

# ordered strategies, most explicit first
_P_ANSWER_IS = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-J])\)?\b", re.I)
_P_PAREN = re.compile(r"\(([A-J])\)")
_P_LETTER_LINE = re.compile(r"(?:^|\n)\s*\(?([A-J])\)?[\.\):]?\s*(?:$|\n)")


def extract_choice(raw_text: str, options: List[str]) -> Optional[str]:
    tail = extract_final_answer(raw_text)
    # 1) explicit "answer is X" (take the last such mention)
    m = list(_P_ANSWER_IS.finditer(tail))
    if m:
        return m[-1].group(1).upper()
    # 2) a parenthesized letter, if exactly one distinct letter appears
    pl = {g.upper() for g in _P_PAREN.findall(tail)}
    if len(pl) == 1:
        return pl.pop()
    # 3) a standalone letter on its own line
    ll = {g.upper() for g in _P_LETTER_LINE.findall(tail)}
    if len(ll) == 1:
        return ll.pop()
    # 4) option-text substring: exactly one option appears verbatim in the tail
    hits = []
    low = tail.lower()
    for i, opt in enumerate(options):
        if opt and opt.lower() in low:
            hits.append(chr(ord("A") + i))
    if len(hits) == 1:
        return hits[0]
    return None   # ambiguous / unextractable


def score_mc(raw_text: str, options: List[str], gold_letter: str) -> Tuple[bool, bool]:
    """Returns (is_correct, extracted_ok). extracted_ok=False when ambiguous."""
    choice = extract_choice(raw_text, options)
    if choice is None:
        return False, False
    return (choice.upper() == gold_letter.upper()), True
