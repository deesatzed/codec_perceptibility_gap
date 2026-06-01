"""Cross-family disagreement for one question = semantic spread of the family
answer-embeddings. Text analog of codec_contest.disagreement_score."""
from __future__ import annotations

import numpy as np


def answer_disagreement(family_vecs: np.ndarray) -> float:
    """family_vecs: (n_families, dim) embeddings of each family's answer to ONE
    question. Returns mean per-dim std across families (>=0; 0 = identical)."""
    v = np.asarray(family_vecs, dtype=float)
    if v.shape[0] < 2:
        return 0.0
    return float(v.std(axis=0).mean())
