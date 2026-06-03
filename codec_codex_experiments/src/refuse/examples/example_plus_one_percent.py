"""LANDING DEMO: the '+1% improvement' that refuse declines.

The single most common over-claim in ML: "our Model B beats Model A by ~1%."
A naive pipeline reports the mean gain as a win. `refuse` declines it when the
paired per-seed improvement sits inside run-to-run noise (cf. arXiv 2511.19794,
"When +1% Is Not Enough"), and only VERIFIES a gain whose bootstrap CI clears 0.

This is the product: not the statistics (those are standard), but the
refuse-by-default CONTRACT that withholds the "win" until it's earned.

    python -m src.refuse.examples.example_plus_one_percent
"""
from __future__ import annotations

import numpy as np

from src.refuse import Battery, improvement_beats_noise


def _accuracies(true_gain: float, noise: float, n_seeds: int = 10, seed: int = 0):
    """Per-seed accuracy of model A and model B (paired by seed). true_gain is the
    real underlying B-A difference; noise is the per-seed run-to-run spread."""
    rng = np.random.default_rng(seed)
    a = 0.80 + rng.normal(0, noise, n_seeds)
    b = a + true_gain + rng.normal(0, noise, n_seeds)   # paired, with its own noise
    return a, b


def _run(label, true_gain, noise):
    a, b = _accuracies(true_gain=true_gain, noise=noise)
    headline = (b.mean() - a.mean()) * 100
    result = Battery([improvement_beats_noise(a.tolist(), b.tolist())]).evaluate(
        value=f"Model B beats Model A by {headline:+.2f}%")
    print(f"[{label}]")
    print(f"   naive headline: 'B beats A by {headline:+.2f}%'  (a={a.mean()*100:.2f}%  b={b.mean()*100:.2f}%)")
    print(f"   refuse verdict: {result.status}")
    if result.status == "REFUSED":
        print(f"      -> {result.reason}")
    else:
        ev = result.receipts[-1].evidence
        print(f"      -> earned: mean gain {ev['mean_gain']:+.4f}, CI {ev['gain_ci95']} (lo > 0)")
    print()


def main():
    print("=== refuse: the '+1% is not enough' guardrail (cf. arXiv 2511.19794) ===\n")
    # Case 1: a ~1% headline 'win' that is actually inside the noise -> REFUSED
    _run("noisy +1% (the classic over-claim)", true_gain=0.010, noise=0.020)
    # Case 2: a genuine, large, low-noise improvement -> VERIFIED
    _run("real improvement", true_gain=0.060, noise=0.010)
    print("The product is the CONTRACT: a 'win' is withheld until its paired gain "
          "clears run-to-run noise. The stats are standard; the default-to-refuse is the point.")


if __name__ == "__main__":
    main()
