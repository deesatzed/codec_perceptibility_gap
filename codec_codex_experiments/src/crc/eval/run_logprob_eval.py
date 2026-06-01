"""Decisive experiment: difficulty-disentangled evaluation of cross-family
disagreement on MMLU-Pro, using LOG-LIKELIHOOD MC scoring (truncation-proof,
parse-free, works for verbose reasoning models).

Per item: each family's option-probability vector -> disagreement (probability
spread), ensemble-wrong (majority), difficulty (fraction wrong). Then the harness
gates (prescreen, collinearity) + difficulty-controlled partial + naive-difficulty
baseline decide: does disagreement predict wrongness BEYOND difficulty?

Run in your terminal (loads 3 real families incl. 27B qwen):
    python -m src.crc.eval.run_logprob_eval --limit 120
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from src.dern_b.mlx_backend import MLXModel
from src.crc.panel import DEFAULT_FAMILIES
from src.crc.eval.benchmark_loader import load_items
from src.crc.eval.mc_logprob import (
    option_letters, distribution_disagreement, ensemble_wrong, difficulty, family_correct,
)
from src.crc.eval.harness import evaluate_signal


def collect_signals(models: Dict[str, MLXModel], items, cache: Dict[str, Any] | None = None):
    """Returns (disagree[], wrong[], difficulty[], per_family_correct counts)."""
    disagree, wrong, diff = [], [], []
    fam_correct = {k: 0 for k in models}
    for it in items:
        gold = ord(it.answer_letter) - ord("A")
        letters = option_letters(len(it.options))
        probs = {k: m.option_logprobs(it.question, letters) for k, m in models.items()}
        disagree.append(distribution_disagreement(probs))
        wrong.append(ensemble_wrong(probs, gold))
        diff.append(difficulty(probs, gold))
        for k, v in probs.items():
            if family_correct(np.asarray(v), gold):
                fam_correct[k] += 1
    return np.array(disagree), np.array(wrong), np.array(diff), fam_correct


def run(limit: int = 120, seed: int = 7) -> Dict[str, Any]:
    items = load_items("mmlu_pro", allow_network=False, limit=limit)
    models = {f["key"]: MLXModel(f["path"]) for f in DEFAULT_FAMILIES}
    disagree, wrong, diff, fam_correct = collect_signals(models, items)

    result = evaluate_signal(disagree.tolist(), diff.tolist(), wrong.tolist(),
                             n_families=len(models))
    return {
        "benchmark": "mmlu_pro",
        "n_items": len(items),
        "families": list(models),
        "per_family_correct": fam_correct,
        "ensemble_error_rate": round(float(np.mean(wrong)), 3),
        "scoring": "log_likelihood_option_probs",
        "disagreement_signal": "cross_family_probability_spread",
        "evaluation": result,
        "note": ("Log-likelihood MC scoring (no generation) -> truncation-proof, "
                 "works for verbose reasoning models. Verdict is gated: error-scarce "
                 "or collinear -> typed status, never a fabricated number."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", default="results/crc_logprob_eval.json")
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    res = run(limit=args.limit)
    from pathlib import Path
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
