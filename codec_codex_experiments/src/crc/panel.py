"""Independent decoder families (local MLX) + answer cache (real generations,
written once). Distinct families = genuine independence for the disagreement
signal. Models verified materialized on disk (design Gate-0 context)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.dern_b.mlx_backend import MLXModel

# Distinct families on disk (different vendors => genuine independence).
DEFAULT_FAMILIES: List[Dict[str, str]] = [
    {"key": "gemma-google", "path": "/Volumes/WS4TB/models/mlx-community/gemma-4-e4b-it-OptiQ-4bit"},
    {"key": "qwen-alibaba", "path": "/Volumes/WS4TB/models/mlx-community/Qwen3.6-27B-OptiQ-4bit"},
    {"key": "granite-ibm", "path": "/Volumes/WS4TB/models/ddark-il/granite-4.1-3b-optiq"},
]


class DecoderPanel:
    def __init__(self, families: List[Dict[str, str]], cache_path: Optional[Path] = None,
                 max_tokens: int = 256) -> None:
        self.families = families
        self.max_tokens = max_tokens
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())
        self._models: Dict[str, MLXModel] = {}

    def _model(self, fam: Dict[str, str]) -> MLXModel:
        if fam["key"] not in self._models:
            self._models[fam["key"]] = MLXModel(fam["path"])
        return self._models[fam["key"]]

    def answer(self, prompt: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        dirty = False
        for fam in self.families:
            ck = f"{fam['key']}\x00{prompt}"
            if ck not in self._cache:
                self._cache[ck] = self._model(fam).generate(prompt, self.max_tokens).text
                dirty = True
            out[fam["key"]] = self._cache[ck]
        if dirty and self.cache_path:
            self.cache_path.write_text(json.dumps(self._cache, indent=2))
        return out
