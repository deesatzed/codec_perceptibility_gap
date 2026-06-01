"""mlx-lm backend: load a real local model once, generate with measured timing.

No mock. Active params are read from the loaded model's parameter tree (the true
array-element count, including quantized weights), so active_param_seconds is a
real compute proxy, not a guess.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

CHEAP_PATH = "/Volumes/WS4TB/models/mlx-community/gemma-4-e4b-it-OptiQ-4bit"
REF_PATH = "/Volumes/WS4TB/models/mlx-community/gemma-4-31B-it-OptiQ-4bit"

# Different model families wrap the answer in different reasoning-trace formats.
# The final answer is whatever follows the LAST thinking-channel terminator. We
# strip the trace so downstream comparison sees ANSWERS, not reasoning. This is
# output normalization — the model's own final segment is returned verbatim.
#   gemma-4 OptiQ:  <|channel>thought ... <channel|>ANSWER
#   qwen3.6 OptiQ:  Thinking Process: ... </think>\n\nANSWER
# Known terminators, tried in order; we take the text after the LAST match.
_TERMINATORS = [
    re.compile(r"<\s*channel\s*\|?\s*>"),   # gemma harmony close
    re.compile(r"</\s*think\s*>", re.I),    # qwen </think>
    re.compile(r"</\s*thought\s*>", re.I),  # variant
]


def extract_final_answer(text: str) -> str:
    """Return the post-thinking final answer if a known reasoning-trace terminator
    is present, else the text unchanged. Always returns the model's own tokens
    verbatim (we only split on the trace boundary; we never rewrite the answer)."""
    last_end = -1
    for term in _TERMINATORS:
        for mt in term.finditer(text):
            if mt.end() > last_end:
                last_end = mt.end()
    if last_end >= 0:
        tail = text[last_end:].strip()
        if tail:
            return tail
    return text.strip()


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    wall_seconds: float
    active_params: int
    active_param_seconds: float


def _count_params(model) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(model.parameters())))


class MLXModel:
    """Lazily loads an mlx model and generates chat completions with metering."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._model = None
        self._tok = None
        self._active_params: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.path)
            self._active_params = _count_params(self._model)

    @property
    def active_params(self) -> int:
        self._ensure_loaded()
        assert self._active_params is not None
        return self._active_params

    def generate(self, user_prompt: str, max_tokens: int = 128) -> GenResult:
        self._ensure_loaded()
        from mlx_lm import generate as mlx_generate
        msgs = [{"role": "user", "content": user_prompt}]
        prompt = self._tok.apply_chat_template(msgs, add_generation_prompt=True)
        prompt_tokens = len(prompt) if isinstance(prompt, list) else len(self._tok.encode(prompt))
        t0 = time.time()
        text = mlx_generate(self._model, self._tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        wall = time.time() - t0
        gen_tokens = max(1, len(self._tok.encode(text)))
        aps = self.active_params * wall
        return GenResult(
            text=extract_final_answer(text),
            prompt_tokens=int(prompt_tokens),
            gen_tokens=int(gen_tokens),
            wall_seconds=float(wall),
            active_params=int(self.active_params),
            active_param_seconds=float(aps),
        )

    def option_logprobs(self, user_prompt: str, option_letters: List[str]) -> "np.ndarray":
        """Log-likelihood MC scoring (the standard, truncation-proof method).

        Returns a normalized probability vector over option_letters = the model's
        next-token probability for each letter after the chat prompt. No text is
        generated, so verbose reasoning models cannot derail it. This replaces
        free-generation + regex parsing, which failed on chatty models.
        """
        import numpy as np
        import mlx.core as mx
        self._ensure_loaded()
        msgs = [{"role": "user", "content": user_prompt}]
        prompt = self._tok.apply_chat_template(msgs, add_generation_prompt=True)
        ids = mx.array(prompt)[None]
        logits = self._model(ids)            # (1, seqlen, vocab)
        last = logits[0, -1, :]
        # per-letter logit = logit of the letter's FIRST token id (single-token A..J)
        vals = []
        for L in option_letters:
            tid = self._tok.encode(L)
            vals.append(float(last[tid[-1]]))
        v = np.asarray(vals, dtype=float)
        v = v - v.max()                       # softmax over the option set
        p = np.exp(v)
        return p / p.sum()
