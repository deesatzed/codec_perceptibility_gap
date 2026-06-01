"""CRC runner: Phase 1 (calibrate on known answers) and Phase 2 (report on
unlabeled questions). Phase 2 issues a calibrated verdict only if Phase 1 earned
trust; otherwise every card is 'uncalibrated'. Out-of-calibration-range questions
are flagged. See docs/plans/2026-05-31-reliability-report-card-design.md."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.crc.disagreement import answer_disagreement
from src.crc.correctness import is_correct
from src.crc.calibrate import calibrate
from src.crc.guardrail import pairwise_error_correlation


def _disagreement_for(panel, embedder, prompt: str):
    answers = panel.answer(prompt)            # {family_key: text}
    keys = list(answers)
    vecs = embedder.encode([answers[k] for k in keys])
    return answers, keys, float(answer_disagreement(vecs))


def phase1_calibrate(panel, embedder, questions: List[str], answer_key: Dict[str, str]) -> Dict[str, Any]:
    """Run known-answer questions through the panel; prove (or refuse) that
    disagreement predicts wrongness."""
    disagree: List[float] = []
    ensemble_wrong: List[float] = []
    per_family_err: Dict[str, List[float]] = {}
    for q in questions:
        answers, fam_keys, dis = _disagreement_for(panel, embedder, q)
        disagree.append(dis)
        key = answer_key[q]
        fam_correct = {k: is_correct(answers[k], key) for k in fam_keys}
        for k in fam_keys:
            per_family_err.setdefault(k, []).append(0.0 if fam_correct[k] else 1.0)
        # ensemble = majority vote correct? wrong if < half the families are correct.
        n_correct = sum(1 for v in fam_correct.values() if v)
        ensemble_wrong.append(0.0 if n_correct * 2 > len(fam_keys) else 1.0)

    fam_err_matrix = np.array([per_family_err[k] for k in per_family_err])
    pec = pairwise_error_correlation(fam_err_matrix) if fam_err_matrix.shape[0] >= 2 else float("nan")
    pec_val = 0.0 if (isinstance(pec, float) and np.isnan(pec)) else float(pec)

    cal = calibrate(np.array(disagree), np.array(ensemble_wrong), pairwise_err_corr=pec_val)
    cal["n_questions"] = len(questions)
    cal["families"] = list(per_family_err)
    return cal


def phase2_report(panel, embedder, questions: List[str], calibration: Dict[str, Any]) -> Dict[str, Any]:
    """Score unlabeled questions; apply the calibrated threshold if VALID, else
    every card is 'uncalibrated'. Flag out-of-calibration-range questions."""
    valid = calibration.get("verdict") == "VALID"
    thr = calibration.get("threshold")
    crange = calibration.get("calibration_range")
    cards: List[Dict[str, Any]] = []
    for q in questions:
        answers, fam_keys, dis = _disagreement_for(panel, embedder, q)
        if not valid:
            verdict = "uncalibrated"
        else:
            out_of_range = bool(crange and (dis < crange[0] or dis > crange[1]))
            if out_of_range:
                verdict = "out-of-calibration"
            elif dis >= thr:
                verdict = "distrust"
            else:
                verdict = "trust"
        cards.append({"question": q, "disagreement": round(dis, 4),
                      "verdict": verdict, "answers": answers})
    cards.sort(key=lambda c: c["disagreement"], reverse=True)   # worst-first
    return {
        "verdict_basis": calibration.get("verdict"),
        "calibrated": valid,
        "hollow_agreement_warning": calibration.get("hollow_agreement_warning"),
        "cards": cards,
        "manifest": {
            "families": calibration.get("families"),
            "calibration_precision": calibration.get("precision"),
            "calibration_recall": calibration.get("recall"),
            "pairwise_error_corr": calibration.get("pairwise_error_corr"),
            "note": ("Verdicts are only as good as the calibration benchmark's "
                     "similarity to these questions; out-of-calibration cards are flagged."),
        },
    }
