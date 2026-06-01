# CRC (Calibrated Reliability Card) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a batch tool that ranks an AI model's answers by reliability using disagreement between independent model families — but only after proving on known-answer data that disagreement predicts wrongness.

**Architecture:** A new `src/crc/` subpackage. Independent decoder families (gemma/qwen/granite, local MLX) answer each question; a local embedder (`all-MiniLM-L6-v2`, cached) turns answers into vectors; cross-family disagreement = semantic spread of those vectors. Phase 1 calibrates the disagreement→wrongness relationship on a known-answer benchmark (reusing CodecGuard's LOCO + permutation-null + bootstrap-CI); Phase 2 reports verdicts on unlabeled questions only if calibration earned trust. The correlated-error ("hollow agreement") caveat is a first-class output. Every failure resolves to an honest status, never a fabricated verdict.

**Tech Stack:** Python 3.13, `sentence-transformers` (MiniLM, installed + cached), `mlx-lm` (decoder families, local), numpy, pytest. Reuses `src/codec_contest.py` (`disagreement_score`, `bootstrap_corr_ci`) and `src/dern_b/mlx_backend.py` (`MLXModel`). No new heavy deps.

**Working dir (all commands):** `/Volumes/WS4TB/codec-exper/codec_codex_experiments` ; `source .venv/bin/activate` first.
**Test commands:** DERN/CRC fast suite `pytest tests/ -q -m "not heavy"` (must stay green — 84 tests today); CRC only `pytest tests/crc -q`; heavy `pytest tests/crc -q -m heavy`.

**Exit criterion (staged, from design §5):**
- Gate 0: real local embedder works (CONFIRMED: MiniLM cached, encodes offline, Paris~"capital is Paris" 0.776 > Paris~Lyon 0.497).
- Gate 1: 100% of pure-logic honesty-gate tests pass.
- Gate 2: Phase 1 runs on a real small benchmark and HONESTLY reports whether the signal works (calibrated threshold + measured P/R, OR `UNVALIDATED`). A fabricated number is never acceptable.

---

## Conventions
- Source `src/crc/*.py` (+ `__init__.py`); tests `tests/crc/test_*.py` (+ `__init__.py`). Imports `from src.crc... import ...`, `from src.codec_contest import ...`, `from src.dern_b.mlx_backend import ...`.
- Heavy tests (load real decoder families) marked `@pytest.mark.heavy` (already registered in `pytest.ini`). Embedder tests are NOT heavy (MiniLM is tiny/fast).
- Real data only. The answer cache stores genuine generations; it is a cache, not a mock.
- All randomness via `np.random.default_rng(seed)`.
- Set `HF_HUB_OFFLINE=1` in the embedder loader so it never hits the network.
- Keep CRC self-contained; do not modify existing modules (the 84 tests stay green).

---

### Task 0: Scaffold + Gate 0 embedder

**Files:** Create `src/crc/__init__.py`, `tests/crc/__init__.py`, `src/crc/embedder.py`; Test `tests/crc/test_embedder.py`

**Step 1: failing test**
```python
# tests/crc/test_embedder.py
import numpy as np
from src.crc.embedder import Embedder

def test_embedder_semantic_spread_real():
    e = Embedder()
    v = e.encode(["Paris", "The capital is Paris.", "Lyon"])
    assert v.shape[0] == 3 and v.ndim == 2
    def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # same meaning closer than different meaning -> the metric is real
    assert cos(v[0], v[1]) > cos(v[0], v[2])
```

**Step 2: run, expect fail** `pytest tests/crc/test_embedder.py -v` → ModuleNotFound.

**Step 3: implement**
```python
# src/crc/__init__.py
"""CRC — Calibrated Reliability Card. Reliability-by-disagreement, validated
before use. See docs/plans/2026-05-31-reliability-report-card-design.md."""
```
```python
# tests/crc/__init__.py   (empty)
```
```python
# src/crc/embedder.py
"""Local sentence embedder (all-MiniLM-L6-v2, cached). Offline, no mock."""
from __future__ import annotations
import os
import numpy as np

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _MODEL) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(list(texts)), dtype=float)
```

**Step 4: run, expect pass.** **Step 5: commit** `feat(crc): scaffold + real MiniLM embedder (Gate 0)`.

---

### Task 1: Disagreement scorer (text analog of CodecGuard)

**Files:** Create `src/crc/disagreement.py`; Test `tests/crc/test_disagreement.py`

**Step 1: failing test**
```python
# tests/crc/test_disagreement.py
import numpy as np
from src.crc.disagreement import answer_disagreement

def test_identical_answers_zero_disagreement():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # 3 identical family vecs
    assert answer_disagreement(v) == 0.0

def test_scattered_answers_higher_than_close():
    close = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.05]])
    far   = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    assert answer_disagreement(far) > answer_disagreement(close)
```

**Step 3: implement** (mirrors `codec_contest.disagreement_score`: per-dimension std across families, mean)
```python
# src/crc/disagreement.py
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
```

**Steps 2/4/5:** fail → pass → commit `feat(crc): cross-family disagreement scorer`.

---

### Task 2: Correctness scorer (Phase 1 ground-truth match)

**Files:** Create `src/crc/correctness.py`; Test `tests/crc/test_correctness.py`

**Step 1: failing test**
```python
# tests/crc/test_correctness.py
from src.crc.correctness import is_correct

def test_exact_match_case_insensitive():
    assert is_correct("Paris", "paris") is True
    assert is_correct("Lyon", "Paris") is False

def test_substring_answer_counts_correct():
    # model wraps the answer in a sentence
    assert is_correct("The capital is Paris.", "Paris") is True
```

**Step 3: implement** (exact / normalized-substring; embedding-sim is a later extension, YAGNI now)
```python
# src/crc/correctness.py
"""Phase-1 correctness vs an answer key. Crisp answers: normalized exact or
substring match. (Embedding-similarity matching deferred until needed.)"""
from __future__ import annotations
import re

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def is_correct(answer: str, key: str) -> bool:
    a, k = _norm(answer), _norm(key)
    if not k:
        return False
    return k == a or k in a.split() or k in a
```

**Steps 2/4/5:** fail → pass → commit `feat(crc): ground-truth correctness scorer`.

---

### Task 3: Calibrator + honesty gates (the heart — pure logic)

**Files:** Create `src/crc/calibrate.py`; Test `tests/crc/test_calibrate.py`

Reuses `bootstrap_corr_ci`; implements permutation null + LOCO + pairwise-error-corr locally (small, mirrors codec_contest).

**Step 1: failing tests (the honesty gates — these are the flaw-catchers)**
```python
# tests/crc/test_calibrate.py
import numpy as np
from src.crc.calibrate import calibrate, MIN_N

def _arrays(n, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    wrong = rng.integers(0, 2, n).astype(float)
    if signal:          # disagreement tracks wrongness
        disagree = wrong * 0.5 + rng.normal(0, 0.05, n) + 0.1
    else:               # disagreement is pure noise wrt wrongness
        disagree = rng.normal(0.3, 0.1, n)
    return disagree, wrong

def test_refuses_below_min_n():
    d, w = _arrays(MIN_N - 1)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "UNVALIDATED"
    assert out["threshold"] is None

def test_refuses_when_signal_is_noise():
    d, w = _arrays(200, signal=False)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "UNVALIDATED"   # must not clear permutation null

def test_validates_and_thresholds_when_signal_real():
    d, w = _arrays(200, signal=True)
    out = calibrate(d, w, pairwise_err_corr=0.1)
    assert out["verdict"] == "VALID"
    assert out["threshold"] is not None
    assert 0.0 <= out["precision"] <= 1.0 and 0.0 <= out["recall"] <= 1.0

def test_correlated_error_guardrail_warns():
    d, w = _arrays(200, signal=True)
    out = calibrate(d, w, pairwise_err_corr=0.7)   # families fail together
    assert out["hollow_agreement_warning"] is True
```

**Step 3: implement**
```python
# src/crc/calibrate.py
"""Phase-1 calibrator. Proves disagreement predicts wrongness BEFORE a threshold
is issued: correlation + bootstrap CI + permutation null + LOCO. Surfaces the
correlated-error (hollow-agreement) caveat. Refuses (UNVALIDATED) if unproven."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from src.codec_contest import bootstrap_corr_ci

MIN_N = 30                 # variance-gate: corr on a handful of points is noise
HOLLOW_AGREEMENT = 0.5     # pairwise family-error corr above this => agreement suspect

def _perm_null_p95(d: np.ndarray, w: np.ndarray, n_perm: int = 500, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        vals.append(float(np.corrcoef(d[rng.permutation(len(d))], w)[0, 1]))
    return float(np.percentile(vals, 95))

def _best_threshold(d: np.ndarray, w: np.ndarray):
    # sweep candidate thresholds; pick the one maximizing F1 of (d>=t) vs wrong
    best = (None, -1.0, 0.0, 0.0)
    for t in np.quantile(d, np.linspace(0.1, 0.9, 17)):
        flag = d >= t
        tp = float((flag & (w > 0)).sum()); fp = float((flag & (w == 0)).sum())
        fn = float((~flag & (w > 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (float(t), f1, prec, rec)
    return best

def calibrate(disagree: np.ndarray, wrong: np.ndarray, pairwise_err_corr: float) -> Dict[str, Any]:
    d = np.asarray(disagree, float); w = np.asarray(wrong, float)
    if len(d) < MIN_N or len(np.unique(w)) < 2:
        return {"verdict": "UNVALIDATED", "reason": f"need >= {MIN_N} pts and both classes",
                "threshold": None, "hollow_agreement_warning": bool(pairwise_err_corr >= HOLLOW_AGREEMENT)}
    corr = float(np.corrcoef(d, w)[0, 1])
    ci = bootstrap_corr_ci(d, w, n_boot=300)
    null_p95 = _perm_null_p95(d, w)
    clears_null = corr > null_p95 and ci["ci95"][0] > 0
    if not clears_null:
        return {"verdict": "UNVALIDATED", "reason": "disagreement does not predict wrongness",
                "corr": round(corr, 3), "null_p95": round(null_p95, 3), "ci95": ci["ci95"],
                "threshold": None, "hollow_agreement_warning": bool(pairwise_err_corr >= HOLLOW_AGREEMENT)}
    t, f1, prec, rec = _best_threshold(d, w)
    return {"verdict": "VALID", "corr": round(corr, 3), "null_p95": round(null_p95, 3),
            "ci95": ci["ci95"], "threshold": round(t, 4), "f1": round(f1, 3),
            "precision": round(prec, 3), "recall": round(rec, 3),
            "calibration_range": [round(float(d.min()), 4), round(float(d.max()), 4)],
            "pairwise_error_corr": round(float(pairwise_err_corr), 3),
            "hollow_agreement_warning": bool(pairwise_err_corr >= HOLLOW_AGREEMENT)}
```

**Steps 2/4/5:** fail → pass → commit `feat(crc): calibrator with permutation-null + hollow-agreement gates`.

---

### Task 4: Pairwise family-error correlation (the guardrail metric)

**Files:** Create `src/crc/guardrail.py`; Test `tests/crc/test_guardrail.py`

**Step 1: failing test**
```python
# tests/crc/test_guardrail.py
import numpy as np
from src.crc.guardrail import pairwise_error_correlation

def test_independent_failures_low_corr():
    rng = np.random.default_rng(0)
    errs = rng.integers(0, 2, (3, 200)).astype(float)  # 3 families fail independently
    assert pairwise_error_correlation(errs) < 0.3

def test_together_failures_high_corr():
    base = (np.random.default_rng(1).integers(0, 2, 200)).astype(float)
    errs = np.stack([base, base, base])                # fail together
    assert pairwise_error_correlation(errs) > 0.9
```

**Step 3: implement** (mirrors `codec_contest.mean_pairwise_codec_error_correlation`)
```python
# src/crc/guardrail.py
"""Correlated-error guardrail: mean pairwise correlation of per-question
right/wrong across family pairs. High => families fail together => agreement is
hollow. Mirrors codec_contest.mean_pairwise_codec_error_correlation."""
from __future__ import annotations
import numpy as np

def pairwise_error_correlation(family_errors: np.ndarray) -> float:
    """family_errors: (n_families, n_questions) of 0/1 wrong flags."""
    e = np.asarray(family_errors, float)
    if e.shape[0] < 2:
        return float("nan")
    cm = np.corrcoef(e)
    mask = ~np.eye(cm.shape[0], dtype=bool)
    return float(np.nanmean(cm[mask]))
```

**Steps 2/4/5:** fail → pass → commit `feat(crc): pairwise family-error correlation guardrail`.

---

### Task 5: Decoder panel + answer cache (real models, heavy)

**Files:** Create `src/crc/panel.py`; Test `tests/crc/test_panel.py`

Panel wraps N `MLXModel` instances (from `dern_b.mlx_backend`) pointed at on-disk families; caches answers to a JSON file keyed by (model_key, prompt). The model paths are listed as constants.

**Step 1: failing test**
```python
# tests/crc/test_panel.py
import pytest
from src.crc.panel import DecoderPanel, DEFAULT_FAMILIES

def test_family_constants_point_at_distinct_families():
    keys = [f["key"] for f in DEFAULT_FAMILIES]
    assert len(set(keys)) == len(keys) >= 2     # >=2 distinct families

@pytest.mark.heavy
def test_panel_answers_and_caches(tmp_path):
    panel = DecoderPanel(DEFAULT_FAMILIES[:2], cache_path=tmp_path / "c.json")
    a1 = panel.answer("What is the capital of France? One word.")
    assert len(a1) == 2 and all(isinstance(x, str) and x for x in a1.values())
    # second call must hit cache (no re-generation): same object content, fast
    a2 = panel.answer("What is the capital of France? One word.")
    assert a1 == a2
```

**Step 3: implement**
```python
# src/crc/panel.py
"""Independent decoder families (local MLX) + answer cache (real generations,
written once). Distinct families = genuine independence for the disagreement
signal. Materialized models verified on disk (design Gate 0 context)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
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
```
(Note: granite path verified on disk at 2.1G; if `Qwen3.6-27B` 17G is too heavy for a quick run, tests may use `DEFAULT_FAMILIES[:2]` with gemma+granite. Model selection is the user's per policy; these are the materialized defaults.)

**Steps 2/4/5:** logic test passes immediately; heavy test gated. Commit `feat(crc): decoder panel + real-generation answer cache`.

---

### Task 6: End-to-end Phase 1 + Phase 2 runner + report

**Files:** Create `src/crc/run_crc.py`; Test `tests/crc/test_run_crc.py`

Ties it together: embed answers → disagreement per question → (Phase 1) correctness + calibrate + guardrail → (Phase 2) verdicts via calibrated threshold + out-of-range flag.

**Step 1: failing test (logic, with a tiny in-memory fake panel that returns fixed strings — substitutes ONLY model I/O, not CRC logic; flag for no-mock note)**
```python
# tests/crc/test_run_crc.py
import numpy as np
from src.crc.run_crc import phase1_calibrate, phase2_report

class _FakePanel:
    # substitutes ONLY model I/O (returns canned answers). CRC logic under test is real.
    def __init__(self, table): self.table = table
    def answer(self, prompt): return self.table[prompt]

def test_phase1_returns_verdict_and_phase2_respects_it():
    # 40 questions; 2 families that AGREE on 'easy' (correct) and DISAGREE on 'hard' (wrong)
    from src.crc.embedder import Embedder
    emb = Embedder()
    qs, key, table = [], {}, {}
    for i in range(40):
        if i % 2 == 0:
            q = f"easy{i}"; table[q] = {"a": "Paris", "b": "Paris"}; key[q] = "Paris"
        else:
            q = f"hard{i}"; table[q] = {"a": "Paris", "b": "Lyon"};  key[q] = "Paris"
        qs.append(q)
    cal = phase1_calibrate(_FakePanel(table), emb, qs, key)
    assert cal["verdict"] in {"VALID", "UNVALIDATED"}
    # Phase 2 on unlabeled: if UNVALIDATED, no card may claim a calibrated verdict
    rep = phase2_report(_FakePanel(table), emb, ["easy0", "hard1"], cal)
    for card in rep["cards"]:
        if cal["verdict"] != "VALID":
            assert card["verdict"] == "uncalibrated"
```

**Step 3: implement** `phase1_calibrate` (embed→disagreement→correctness→`calibrate`+`pairwise_error_correlation`) and `phase2_report` (embed→disagreement→threshold verdict trust/caution/distrust, or `uncalibrated`; flag out-of-`calibration_range`). Emit honesty manifest. (Full code follows the design §3; build it to satisfy the test.)

**Steps 2/4/5:** fail → pass → commit `feat(crc): phase1 calibrate + phase2 report runner`.

---

### Task 7: Heavy end-to-end on real models + results note

**Files:** Test `tests/crc/test_end_to_end_real.py`; Create `results/crc_results.md` (after run)

**Step 1: heavy test**
```python
# tests/crc/test_end_to_end_real.py
import pytest
from src.crc.panel import DecoderPanel, DEFAULT_FAMILIES
from src.crc.embedder import Embedder
from src.crc.run_crc import phase1_calibrate

@pytest.mark.heavy
def test_real_calibration_is_honest(tmp_path):
    # A small real known-answer benchmark (factual one-word Qs committed as data).
    from src.crc.bench import TINY_FACTUAL   # (questions, answer key)
    panel = DecoderPanel(DEFAULT_FAMILIES[:2], cache_path=tmp_path / "c.json")
    cal = phase1_calibrate(panel, Embedder(),
                           list(TINY_FACTUAL), TINY_FACTUAL)
    # Honest: either it proves the signal (VALID + threshold + P/R) or refuses.
    assert cal["verdict"] in {"VALID", "UNVALIDATED"}
    if cal["verdict"] == "VALID":
        assert cal["threshold"] is not None and "precision" in cal
    assert "hollow_agreement_warning" in cal
```
Add `src/crc/bench.py` with a committed `TINY_FACTUAL` dict (real questions + keys, ~40 items, no network).

**Step 2: run it yourself** (heavy): `pytest tests/crc/test_end_to_end_real.py -q -m heavy` then run a driver to print the calibration.

**Step 3: write `results/crc_results.md`** with the honest outcome — the measured corr/null/threshold/P/R and the pairwise-error-corr, OR the `UNVALIDATED` finding (a real negative about cross-family disagreement on this benchmark). Mirror the honest tone of `results/dern_stage_b_results.md`. State does/doesn't.

**Step 5: commit + push** `feat(crc): real-model calibration result + honest results note`.

---

### Task 8: Full-suite green

`pytest tests/ -q -m "not heavy"` → the 84 existing + new CRC logic tests pass; heavy CRC tests pass when run with `-m heavy`. Confirm no existing test broke. Commit any fixups.

---

## Done criteria (recap)
Gate 0 (embedder) confirmed. Gate 1 = all pure-logic honesty gates green (refuse-below-N, refuse-on-noise, hollow-agreement-warns, no-verdict-without-calibration). Gate 2 = real Phase-1 calibration runs and honestly reports VALID+P/R or UNVALIDATED. The pre-committed honest negative (cross-family disagreement insufficient) is a publishable finding, not a failure to hide.

## Notes for the implementer
- Reuse `codec_contest.bootstrap_corr_ci`; do not reinvent. Mirror (don't import-couple) the permutation-null/LOCO/pairwise-corr patterns into `src/crc/` so CRC is self-contained.
- Never weaken an honesty-gate assertion to make it pass; a FAIL there means the signal didn't earn trust — report it.
- The fake/canned panels in logic tests substitute ONLY model I/O, not CRC logic (same boundary as the dern_b stubs); the heavy tests carry the real-model truth.
- All randomness via `np.random.default_rng(seed)`.
