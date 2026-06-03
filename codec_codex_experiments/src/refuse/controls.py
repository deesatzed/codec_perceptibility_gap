"""The control battery. Each control is a factory returning a zero-arg callable
(a Control) that produces a CheckResult. All are standard statistics; the point
is composition + the refuse-by-default contract, not novelty.

Controls:
- base_rate:        refuse if the thing-to-detect is too rare (or too universal)
                    to validate a detector against (error/event scarcity).
- beats_difficulty: refuse unless a signal beats a difficulty/triviality baseline
                    with a bootstrap-CI on the AUC gap strictly above zero.
- permutation_null: refuse unless an observed correlation exceeds its permutation
                    null (real, not chance).
- collinearity:     refuse a 'controlled' comparison when the control variable is
                    near-identical to the label (the comparison is degenerate).
- coverage:         refuse a measurement whose sampling did not cover the work
                    window (e.g. energy sampled < X% of the measured interval).
- improvement_beats_noise: refuse to declare 'B beats A' unless the PAIRED
                    per-unit improvement's bootstrap CI is strictly above zero
                    (the '+1% is not enough' guardrail against over-claiming wins).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .contract import CheckResult, Control


def base_rate(events: List[float], floor: float = 0.30, ceil: float = 0.95,
              min_n: int = 30, name: str = "base_rate") -> Control:
    """events: 0/1 per item (1 = the event/error to be detected)."""
    def _check() -> CheckResult:
        e = np.asarray(events, dtype=float)
        n = len(e)
        if n < min_n or len(np.unique(e)) < 2:
            return CheckResult(name, False,
                               f"insufficient data: n={n} (<{min_n}) or single class",
                               {"n": n})
        rate = float(np.mean(e))
        if rate < floor or rate > ceil:
            return CheckResult(name, False,
                               f"event rate {rate:.3f} outside [{floor},{ceil}] — too "
                               "scarce/universal to validate a detector against",
                               {"rate": round(rate, 3), "n": n})
        return CheckResult(name, True, f"event rate {rate:.3f} in [{floor},{ceil}]",
                           {"rate": round(rate, 3), "n": n})
    return _check


def collinearity(control_var: List[float], label: List[float], bound: float = 0.95,
                 name: str = "collinearity") -> Control:
    """Refuse if |corr(control_var, label)| >= bound (degenerate comparison)."""
    def _check() -> CheckResult:
        x = np.asarray(control_var, float); y = np.asarray(label, float)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return CheckResult(name, False, "degenerate variance in control or label", {})
        c = abs(float(np.corrcoef(x, y)[0, 1]))
        if not np.isfinite(c) or c >= bound:
            return CheckResult(name, False,
                               f"control ~= label (|corr|={c:.3f} >= {bound}); "
                               "controlled comparison is degenerate",
                               {"corr": round(c, 3)})
        return CheckResult(name, True, f"control vs label |corr|={c:.3f} < {bound}",
                           {"corr": round(c, 3)})
    return _check


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    # rank-based AUC (no sklearn dependency)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0
    n_pos = int(pos.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def beats_difficulty(signal: List[float], difficulty: List[float], label: List[float],
                     n_boot: int = 300, seed: int = 17, name: str = "beats_difficulty") -> Control:
    """Refuse unless the signal predicts the label BETTER than a difficulty-only
    baseline, with a bootstrap CI on the AUC gap strictly above zero."""
    def _check() -> CheckResult:
        s = np.asarray(signal, float); d = np.asarray(difficulty, float); y = np.asarray(label, float)
        auc_s, auc_b = _auc(s, y), _auc(d, y)
        if not (np.isfinite(auc_s) and np.isfinite(auc_b)):
            return CheckResult(name, False, "AUC undefined (single-class label)", {})
        rng = np.random.default_rng(seed)
        gaps = []
        nn = len(y)
        for _ in range(n_boot):
            idx = rng.choice(nn, nn, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            gaps.append(_auc(s[idx], y[idx]) - _auc(d[idx], y[idx]))
        if not gaps:
            return CheckResult(name, False, "could not bootstrap (resamples single-class)", {})
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        ev = {"auc_signal": round(auc_s, 3), "auc_baseline": round(auc_b, 3),
              "gap_ci95": [round(float(lo), 3), round(float(hi), 3)]}
        if lo > 0.0:
            return CheckResult(name, True,
                               f"signal AUC {auc_s:.3f} beats difficulty {auc_b:.3f} "
                               f"(gap CI lo={lo:.3f} > 0)", ev)
        return CheckResult(name, False,
                           f"signal does NOT beat difficulty baseline (AUC {auc_s:.3f} vs "
                           f"{auc_b:.3f}, gap CI=[{lo:.3f},{hi:.3f}] includes/below 0)", ev)
    return _check


def permutation_null(x: List[float], y: List[float], n_perm: int = 500, seed: int = 7,
                     name: str = "permutation_null") -> Control:
    """Refuse unless corr(x,y) exceeds its permutation-null 95th percentile."""
    def _check() -> CheckResult:
        a = np.asarray(x, float); b = np.asarray(y, float)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return CheckResult(name, False, "degenerate variance", {})
        corr = float(np.corrcoef(a, b)[0, 1])
        rng = np.random.default_rng(seed)
        null = [float(np.corrcoef(a[rng.permutation(len(a))], b)[0, 1]) for _ in range(n_perm)]
        p95 = float(np.percentile(null, 95))
        ev = {"corr": round(corr, 3), "null_p95": round(p95, 3)}
        if corr > p95:
            return CheckResult(name, True, f"corr {corr:.3f} > null p95 {p95:.3f}", ev)
        return CheckResult(name, False, f"corr {corr:.3f} <= null p95 {p95:.3f} (not beyond chance)", ev)
    return _check


def improvement_beats_noise(scores_a: List[float], scores_b: List[float],
                            n_boot: int = 2000, seed: int = 11,
                            name: str = "improvement_beats_noise") -> Control:
    """Refuse the claim 'B beats A' unless the PAIRED per-unit improvement
    (b_i - a_i, e.g. per-seed or per-item) has a bootstrap CI strictly above 0.

    This is the '+1% is not enough' guardrail (cf. arXiv 2511.19794): a headline
    mean gain is not a win if it sits inside run-to-run noise. scores_a/scores_b
    must be PAIRED (same length, aligned units)."""
    def _check() -> CheckResult:
        a = np.asarray(scores_a, float); b = np.asarray(scores_b, float)
        if a.shape != b.shape or a.size < 2:
            return CheckResult(name, False, "scores must be paired, aligned, length>=2", {})
        delta = b - a
        mean_gain = float(delta.mean())
        rng = np.random.default_rng(seed)
        n = len(delta)
        boots = [float(delta[rng.choice(n, n, replace=True)].mean()) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ev = {"mean_gain": round(mean_gain, 4),
              "gain_ci95": [round(float(lo), 4), round(float(hi), 4)], "n_pairs": n}
        if lo > 0.0:
            return CheckResult(name, True,
                               f"B beats A: mean gain {mean_gain:+.4f}, CI lo={lo:.4f} > 0", ev)
        return CheckResult(name, False,
                           f"'B beats A' NOT supported: mean gain {mean_gain:+.4f} but CI "
                           f"[{lo:.4f},{hi:.4f}] includes 0 (inside run-to-run noise)", ev)
    return _check


def coverage(sampled_seconds: float, work_seconds: float, floor: float = 0.60,
             name: str = "coverage") -> Control:
    """Refuse a measurement whose sampling covered < floor of the work window."""
    def _check() -> CheckResult:
        if work_seconds <= 0:
            return CheckResult(name, False, "work_seconds <= 0", {})
        cov = sampled_seconds / work_seconds
        ev = {"coverage": round(cov, 3)}
        if cov >= floor:
            return CheckResult(name, True, f"coverage {cov:.2f} >= {floor}", ev)
        return CheckResult(name, False,
                           f"coverage {cov:.2f} < {floor}: measurement did not span the work window", ev)
    return _check
