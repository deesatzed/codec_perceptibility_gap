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

# These OptiQ gemma-4 builds emit a harmony-style reasoning channel:
#   <|channel>thought\n...reasoning...<channel|>FINAL ANSWER
# The final answer follows the closing channel marker. We extract it so the
# cascade compares ANSWERS, not reasoning traces. This is output normalization,
# not answer modification — the model's own final segment is returned verbatim.
_CHANNEL_CLOSE = re.compile(r"<\s*channel\s*\|?\s*>")
_THOUGHT_OPEN = re.compile(r"<\|?\s*channel\s*\|?\s*>\s*thought", re.I)


def extract_final_answer(text: str) -> str:
    """Return the post-thinking final answer if the harmony channel is present,
    else the text unchanged. Always returns the model's own tokens verbatim."""
    if _THOUGHT_OPEN.search(text):
        parts = _CHANNEL_CLOSE.split(text)
        if len(parts) >= 2 and parts[-1].strip():
            return parts[-1].strip()
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
