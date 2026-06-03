"""Worked example: is a 'reliability signal' real, or just a difficulty proxy?

Two synthetic-but-honest cases run through the same Battery:
  A) a signal that genuinely beats difficulty  -> VERIFIED
  B) a signal that is just difficulty in disguise -> REFUSED

This mirrors the real finding from the program (cross-model disagreement on
MMLU-Pro was a difficulty proxy and got refused). Run:
    python -m src.refuse.examples.example_difficulty
"""
from __future__ import annotations

import numpy as np

from src.refuse import Battery, base_rate, beats_difficulty


def _case(real_signal: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 400
    label = (rng.random(n) < 0.45).astype(float)          # 45% events -> not scarce
    # difficulty is a WEAK predictor (a noisy hint), not a label leak:
    difficulty = 0.15 * label + rng.normal(0, 1.0, n)
    if real_signal:
        # signal adds genuine information ABOVE the weak difficulty hint
        signal = 0.9 * label + 0.1 * difficulty + rng.normal(0, 0.3, n)
    else:
        # signal is just the difficulty hint relabeled (no extra info)
        signal = difficulty + rng.normal(0, 0.01, n)
    battery = Battery([
        base_rate(label.tolist(), floor=0.30),
        beats_difficulty(signal.tolist(), difficulty.tolist(), label.tolist()),
    ])
    return battery.evaluate(value="reliability_score")


def main():
    for real in (True, False):
        r = _case(real_signal=real)
        tag = "real-signal" if real else "difficulty-proxy"
        print(f"[{tag:16s}] {r.status}", end="  ")
        if r.status == "REFUSED":
            print(f"-> refused: {r.reason}")
        else:
            ev = r.receipts[-1].evidence
            print(f"-> verified: AUC {ev['auc_signal']} beats baseline {ev['auc_baseline']}")


if __name__ == "__main__":
    main()
