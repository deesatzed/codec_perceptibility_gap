"""Skill-score audit of agent action-confidence (the one experiment that survived
four adversarial checks). Honest framing: this is a Brier-skill-score / selective-
prediction audit (cite BSS + Geifman/El-Yaniv lineage), NOT a new safety primitive.

Question: does an agent's ACTION-confidence predict task success BEYOND a
difficulty-only baseline? Reuses the difficulty-baseline harness
(beats_difficulty_baseline) with bootstrap-CI separation.

Signals tested (per task, real, no mock):
  - confidence  = max tool-choice probability of the primary model
  - neg_entropy = 1 - normalized entropy of the choice distribution
  - disagreement = L2 distance between two model families' choice distributions
Success = the model's argmax tool == the verifiably-correct tool.

Run locally:  python -m src.crc.agent.run_skill_audit
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from src.dern_b.mlx_backend import MLXModel
from src.crc.agent.tasks import TASKS, TOOL_LETTERS, render, difficulty_features
from src.crc.eval.baseline import beats_difficulty_baseline

PRIMARY = "/Volumes/WS4TB/models/ddark-il/granite-4.1-3b-optiq"
SECONDARY = "/Volumes/WS4TB/models/mlx-community/gemma-4-e4b-it-OptiQ-4bit"


def _norm_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def collect(primary: MLXModel, secondary: MLXModel) -> Dict[str, Any]:
    confidence, neg_entropy, disagreement, wrong, diff_rows = [], [], [], [], []
    per_task = []
    for t in TASKS:
        prompt = render(t)
        p1 = primary.option_logprobs(prompt, TOOL_LETTERS)
        p2 = secondary.option_logprobs(prompt, TOOL_LETTERS)
        choice = TOOL_LETTERS[int(np.argmax(p1))]
        is_wrong = 0.0 if choice == t.correct else 1.0
        confidence.append(float(p1.max()))
        neg_entropy.append(1.0 - _norm_entropy(p1))
        disagreement.append(float(np.linalg.norm(p1 - p2)))
        wrong.append(is_wrong)
        diff_rows.append(difficulty_features(t))
        per_task.append({"task": t.prompt_task[:50], "correct": t.correct,
                         "chose": choice, "wrong": is_wrong, "is_trap": t.is_trap,
                         "confidence": round(float(p1.max()), 3)})
    return {"confidence": np.array(confidence), "neg_entropy": np.array(neg_entropy),
            "disagreement": np.array(disagreement), "wrong": np.array(wrong),
            "difficulty_rows": diff_rows, "per_task": per_task}


def _difficulty_scalar(rows: List[dict], wrong: np.ndarray) -> np.ndarray:
    """A difficulty-only predictor of wrongness, fit from pre-action features
    (the skill-score reference forecast). Logistic fit; if degenerate, fall back
    to a single most-correlated feature."""
    import numpy as np
    keys = list(rows[0].keys())
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    y = np.asarray(wrong, float)
    try:
        from sklearn.linear_model import LogisticRegression
        if len(np.unique(y)) < 2:
            return np.full(len(y), float(y.mean()))
        clf = LogisticRegression(max_iter=200).fit(X, y)
        return clf.predict_proba(X)[:, 1]
    except Exception:
        # fallback: standardized sum (still a difficulty-only score)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        return Xs.sum(1)


def run() -> Dict[str, Any]:
    primary = MLXModel(PRIMARY)
    secondary = MLXModel(SECONDARY)
    d = collect(primary, secondary)
    wrong = d["wrong"]
    err_rate = float(np.mean(wrong))

    # Honesty gate (reuse Gate 1): if the agent errs too rarely, there is nothing
    # for any signal to predict and every AUC/CI is noise. Refuse, do not report
    # skill numbers. This is the exact discipline the rest of the program enforces;
    # the first run lacked it and produced uninformative CIs ([-0.48, 0.67]).
    from src.crc.eval.prescreen import prescreen_error_rate
    pre = prescreen_error_rate(wrong.tolist())
    if not pre["gate_pass"]:
        return {
            "verdict": pre["status"],
            "n_tasks": len(TASKS),
            "error_rate": round(err_rate, 3),
            "n_traps": int(sum(1 for t in TASKS if t.is_trap)),
            "reason": ("agent error rate too low (or n too small) to audit skill; "
                       "signals cannot be validated against a baseline with so few errors. "
                       "Need harder/error-inducing tasks before any verdict."),
            "per_task": d["per_task"],
        }

    difficulty = _difficulty_scalar(d["difficulty_rows"], wrong)

    signals = {}
    for name in ("confidence", "neg_entropy", "disagreement"):
        s = d[name]
        # higher signal should mean MORE reliable; we predict WRONGNESS, so for
        # confidence/neg_entropy (high=reliable) we invert to predict wrong.
        pred = (-s if name in ("confidence", "neg_entropy") else s)
        signals[name] = beats_difficulty_baseline(pred.tolist(), difficulty.tolist(), wrong.tolist())

    return {
        "n_tasks": len(TASKS),
        "error_rate": round(err_rate, 3),
        "n_traps": int(sum(1 for t in TASKS if t.is_trap)),
        "framing": "Brier-skill-score / selective-prediction audit of agent action-confidence; NOT a new primitive",
        "signals_vs_difficulty_baseline": signals,
        "per_task": d["per_task"],
        "verdict_note": ("For each signal, wins=True means action-confidence beats the "
                         "difficulty-only baseline (bootstrap-CI of AUC gap > 0) -> it has "
                         "SKILL. wins=False -> it is a difficulty proxy (the MC finding "
                         "transfers to agent action)."),
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    res = run()
    from pathlib import Path
    Path("results/agent_skill_audit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=2))
