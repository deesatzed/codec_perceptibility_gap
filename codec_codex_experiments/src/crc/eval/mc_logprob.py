"""Log-likelihood multiple-choice scoring + probability-distribution disagreement.

The standard way to evaluate LLMs on MC: read the model's probability over the
option letters (one forward pass) instead of generating + parsing free text.
Truncation-proof, parse-free, fast, and unaffected by verbose reasoning models.

Disagreement here = spread of the per-model option-probability distributions
across the panel (cleaner than embedding free text for MC).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def option_letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def model_choice(prob_vec: np.ndarray) -> int:
    """Argmax option index from a model's probability vector."""
    return int(np.argmax(prob_vec))


def panel_distributions(panel_probs: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack family probability vectors -> (n_families, n_options)."""
    return np.stack([np.asarray(panel_probs[k], dtype=float) for k in panel_probs])


def distribution_disagreement(panel_probs: Dict[str, np.ndarray]) -> float:
    """Cross-family disagreement = mean per-option std of the probability vectors.
    0 = all families assign identical option probabilities; higher = more spread.
    Direct probability-space analog of CodecGuard's disagreement_score."""
    P = panel_distributions(panel_probs)
    if P.shape[0] < 2:
        return 0.0
    return float(P.std(axis=0).mean())


def family_correct(prob_vec: np.ndarray, gold_index: int) -> bool:
    return model_choice(prob_vec) == gold_index


def ensemble_wrong(panel_probs: Dict[str, np.ndarray], gold_index: int) -> float:
    """Majority-vote wrongness: 1.0 if fewer than half the families pick gold."""
    n_correct = sum(1 for v in panel_probs.values() if family_correct(np.asarray(v), gold_index))
    return 0.0 if n_correct * 2 > len(panel_probs) else 1.0


def difficulty(panel_probs: Dict[str, np.ndarray], gold_index: int) -> float:
    """Fraction of families that got it wrong (the capability/difficulty prior)."""
    n_wrong = sum(1 for v in panel_probs.values() if not family_correct(np.asarray(v), gold_index))
    return n_wrong / max(1, len(panel_probs))
