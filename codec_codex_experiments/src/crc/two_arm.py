"""Two-arm CRC experiment: does cross-family disagreement signal RELIABILITY, or
is it just detecting memorization / re-testing capability? (The higher-concept
question.)

Arm 1 (memorized): plain factual questions the models likely recall.
Arm 2 (perturbed): same facts wrapped in a counterfactual premise, so the
  memorized answer is provably WRONG and the correct answer follows only from
  reading the premise. A model that parrots the memorized answer is provably not
  reasoning on the input.

Verdict logic:
- disagreement predicts wrongness on Arm 2 -> tracks genuine reasoning failure,
  not just recall/difficulty: REAL signal.
- disagreement blind on Arm 2 (models agree on the parroted memorized answer and
  are all wrong) -> hollow agreement / correlated error: signal is a memorization
  or capability proxy, NOT reliability.

Difficulty control: also report whether disagreement adds information BEYOND a
per-question difficulty prior (mean family error rate). If disagreement only
mirrors difficulty, it is not a distinct concept.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.crc.run_crc import phase1_calibrate, _disagreement_for
from src.crc.correctness import is_correct


def _per_question_signals(panel, embedder, questions: List[str], key: Dict[str, str]):
    """Return (disagree[], ensemble_wrong[], difficulty[]) for a question set.
    difficulty = fraction of families wrong on that question (the capability prior)."""
    disagree, wrong, difficulty = [], [], []
    for q in questions:
        answers, fam_keys, dis = _disagreement_for(panel, embedder, q)
        fam_wrong = [0.0 if is_correct(answers[k], key[q]) else 1.0 for k in fam_keys]
        disagree.append(dis)
        difficulty.append(float(np.mean(fam_wrong)))
        n_correct = sum(1 for w in fam_wrong if w == 0.0)
        wrong.append(0.0 if n_correct * 2 > len(fam_keys) else 1.0)
    return np.array(disagree), np.array(wrong), np.array(difficulty)


def disagreement_beyond_difficulty(disagree: np.ndarray, wrong: np.ndarray,
                                   difficulty: np.ndarray) -> Dict[str, Any]:
    """Does disagreement predict wrongness BEYOND the difficulty prior?

    Partial correlation of disagreement with wrongness, controlling for difficulty.
    If ~0, disagreement carries no information difficulty doesn't already have
    (i.e. it's a capability proxy, not a distinct concept).
    """
    def _resid(y, x):
        if np.std(x) < 1e-9:
            return y - np.mean(y)
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)
    rd = _resid(disagree, difficulty)
    rw = _resid(wrong, difficulty)
    if np.std(rd) < 1e-9 or np.std(rw) < 1e-9:
        partial = float("nan")
    else:
        partial = float(np.corrcoef(rd, rw)[0, 1])
    raw = float(np.corrcoef(disagree, wrong)[0, 1]) if (np.std(disagree) > 1e-9 and np.std(wrong) > 1e-9) else float("nan")
    return {
        "raw_corr_disagree_wrong": round(raw, 3) if raw == raw else None,
        "partial_corr_controlling_difficulty": round(partial, 3) if partial == partial else None,
        "interpretation": (
            "partial ~ 0 => disagreement is a difficulty/capability proxy (no higher "
            "concept); partial clearly > 0 => disagreement carries reliability "
            "information beyond difficulty."
        ),
    }


def run_two_arm(panel, embedder, arm1_key: Dict[str, str], arm2_key: Dict[str, str]) -> Dict[str, Any]:
    """Calibrate + difficulty-control on both arms; emit the higher-concept verdict."""
    out: Dict[str, Any] = {}
    for name, key in [("arm1_memorized", arm1_key), ("arm2_perturbed", arm2_key)]:
        qs = list(key)
        cal = phase1_calibrate(panel, embedder, qs, key)
        d, w, diff = _per_question_signals(panel, embedder, qs, key)
        ctrl = disagreement_beyond_difficulty(d, w, diff)
        out[name] = {
            "calibration_verdict": cal["verdict"],
            "corr": cal.get("corr"),
            "threshold": cal.get("threshold"),
            "precision": cal.get("precision"),
            "recall": cal.get("recall"),
            "hollow_agreement_warning": cal.get("hollow_agreement_warning"),
            "ensemble_wrong_rate": round(float(np.mean(w)), 3),
            "mean_disagreement": round(float(np.mean(d)), 4),
            "difficulty_control": ctrl,
        }

    # higher-concept verdict
    a2 = out["arm2_perturbed"]
    a2_valid = a2["calibration_verdict"] == "VALID"
    a2_partial = a2["difficulty_control"]["partial_corr_controlling_difficulty"]
    beyond_difficulty = bool(a2_partial is not None and a2_partial > 0.2)
    if a2_valid and beyond_difficulty:
        verdict = "REAL_SIGNAL"
        msg = ("Disagreement predicts wrongness on perturbed (reasoning-required) "
               "questions AND beyond a difficulty prior -> a genuine reliability "
               "signal, not just a capability/memorization proxy.")
    elif not a2_valid and a2["ensemble_wrong_rate"] > 0.5 and a2["mean_disagreement"] < out["arm1_memorized"]["mean_disagreement"]:
        verdict = "HOLLOW_AGREEMENT"
        msg = ("On perturbed questions the families agree (low disagreement) yet are "
               "mostly WRONG (parroting memorized answers): hollow agreement / "
               "correlated error. Disagreement is blind to shared-blindspot failure "
               "-> a memorization/capability proxy here, not reliability.")
    else:
        verdict = "INCONCLUSIVE"
        msg = ("Mixed/insufficient signal; see per-arm numbers. Not enough evidence "
               "that disagreement is a reliability signal beyond difficulty.")
    out["higher_concept_verdict"] = {"verdict": verdict, "explanation": msg,
                                     "arm2_beyond_difficulty": beyond_difficulty}
    return out
