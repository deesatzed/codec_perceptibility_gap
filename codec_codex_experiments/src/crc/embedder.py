"""Local sentence embedder (all-MiniLM-L6-v2, cached). Offline, no mock."""
from __future__ import annotations

import os

import numpy as np

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _MODEL) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(list(texts)), dtype=float)
