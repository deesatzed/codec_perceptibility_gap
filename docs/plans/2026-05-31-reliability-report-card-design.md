# CRC — Calibrated Reliability Card

**Design document.** 2026-05-31. Status: approved, pre-implementation.
Author dialogue: Wayne A. Satz + Claude (brainstorming skill).
Lineage: applies the validated CodecGuard result (multi-codec disagreement
predicts error; *Perceptibility Gap* program, this repo) to a pragmatic AI
reliability/audit application — re-aimed from the energy-routing track after a
step-back review judged reliability more distinctive and more validated.

---

## 0. Origin and intent

After building DERN (energy-routing) and hardening its measurement through a
four-agent ultrathink, a step-back review concluded the **energy angle had
drifted from the original mission and was a crowded field**, while the
**reliability/audit use of CodecGuard** is both more distinctive and already
validated (it survived even the chaotic Hénon family). We re-aim the pragmatic
work at reliability.

**Target (locked):** a **batch reliability report card** for an AI model. Given
a model + a set of questions, run each through several **independent decoder
families**, measure how much their answers **disagree**, and produce a ranked
report of which answers are probably unreliable — **but only after proving, on
known-answer data, that disagreement actually predicts wrongness.** The
correlated-error caveat (families wrong *together* → hollow agreement) is a
first-class output, not hidden.

**Why this, not energy:** it answers "can I trust what this AI told me?" without
needing the right answer in advance; it's a less-crowded niche than model-routing;
it reuses the validated CodecGuard machinery; and it's downstream of the
perceptibility-gap theory (measuring when a translation loses something).

**The honest claim:** CRC never issues a reliability verdict on unlabeled data
unless Phase 1 *measured* that the signal works on comparable known-answer data.
If the signal doesn't earn trust (correlation below the permutation null, or LOCO
non-positive), CRC reports `UNVALIDATED` and refuses a calibrated threshold — the
same honest-failure discipline as the DERN energy gates.

### Locked decisions (from the design dialogue)
- **Decoders = different model families** (gemma-4/Google, qwen3/Alibaba,
  granite/IBM) — genuine independence, so agreement is meaningful and
  "confidently-wrong-together" is rarer. Uses the diverse MLX models on disk.
- **Disagreement = semantic spread of answer embeddings** — the text analog of
  CodecGuard's `disagreement_score` (std of predictions). Treats "Paris" and "The
  capital is Paris" as agreement, "Lyon" as disagreement.
- **Validate-then-deploy** — calibrate on a known-answer benchmark (prove the
  signal), then run unlabeled (the deployable value).
- **Approach A (Calibrated Reliability Report).** Approach C (multi-metric
  bake-off) deferred — only if the single metric proves insufficient (YAGNI).
- **Embedder (Gate 0, resolved):** `sentence-transformers/all-MiniLM-L6-v2` is
  fully cached locally (real `model.safetensors`); needs only
  `pip install sentence-transformers`. Backup: nomic-embed GGUF on disk. No mock.

---

## 1. Concept

**CRC — Calibrated Reliability Card.** Two phases, one tool:
- **Phase 1 — Calibrate (needs answer key):** run a known-answer benchmark through
  N independent families; per question compute cross-family disagreement
  (embedding spread) and actual correctness; measure whether disagreement predicts
  wrongness; output a calibrated threshold with measured precision/recall — or
  `UNVALIDATED`.
- **Phase 2 — Report (no answer key):** run unlabeled questions, score
  disagreement, rank worst-first, apply the calibrated threshold; per-question
  cards + honesty manifest.

CRC issues a calibrated verdict only when the signal earned trust in Phase 1.

---

## 2. Architecture

```
 INPUT: questions (+ answer key in Phase 1 only)
   |
   v
 (1) DECODER PANEL — N independent model families (gemma/qwen/granite)
     each answers every question; answers CACHED to disk (real generations, written once)
   |
   v
 (2) EMBEDDER — all-MiniLM-L6-v2 (local, cached); embeds each answer -> vector
   |
   v
 (3) DISAGREEMENT SCORER — per question: semantic spread of the N answer-embeddings
     (mirrors codec_contest.disagreement_score)
   |
   +--------------- PHASE 1 (answer key) ----------------+
   v                                                     |
 (4a) CORRECTNESS SCORER       (4b) CALIBRATOR+GUARDRAIL  |
   answer vs key                - corr(disagree, wrong)   |
   (exact / embedding-sim)      - LOCO + permutation null |
                                - pairwise family-error   |
                                  corr (correlated-error) |
                                -> threshold + P/R, or    |
                                   UNVALIDATED             |
   +-----------------------------------------------------+
   |
   v
 (5) REPORT (Phase 2): rank by disagreement; apply calibrated threshold;
     per-question cards; honesty manifest (families, P/R, guardrail, caveats)
```

**Reused (validated):** `dern_b.mlx_backend.MLXModel`; `codec_contest`
`disagreement_score`, LOCO, permutation-null, `bootstrap_corr_ci`,
`mean_pairwise_codec_error_correlation`; honest-failure + ledger discipline.
**New:** embedder step, text correctness scoring, calibrate->deploy flow, report.

New subpackage: `src/crc/`. Answer cache keyed by (model, prompt) — genuine
generations stored once, not a mock.

---

## 3. Data flow, calibration, guardrail

### 3.1 Phase 1 calibration
Per question q with known answer a*: get N family answers, embed, compute
`disagree[q]` (mean-dim std of vectors) and `correct[q]` (per family), and
`ensemble_wrong[q]` (majority answer != a*). Then:
- correlation + bootstrap CI: is disagreement higher on wrong answers?
- LOCO: predict each held-out family's error from the others' disagreement
  (removes self-reference).
- permutation null: real correlation must clear the 95th-percentile null.
- threshold sweep -> operating point with measured precision/recall.

**Gate:** correlation fails null OR LOCO non-positive -> `UNVALIDATED`, no
threshold; Phase 2 cards stamped "uncalibrated."

### 3.2 Correlated-error guardrail (the built-in honesty)
`pairwise_error_corr` = mean correlation of per-question wrong/right across family
pairs. Low -> families fail independently -> agreement meaningful. High ->
families fail together -> agreement hollow -> CRC prints a warning and
down-weights confidence. This is `codec_contest.mean_pairwise_codec_error_correlation`
(reported 0.57-0.66 in the research), pointed at LLM families. Always reported.

### 3.3 Phase 2 report
Same pipeline minus correctness; verdict = trust|caution|distrust via the
calibrated threshold; rank worst-first; manifest with families, calibration P/R,
guardrail status, and the calibration-domain caveat.

### 3.4 Calibration-transfer trap (named + guarded)
A threshold calibrated on trivia may not hold on medical questions. Phase 2: (a)
record the calibration domain, (b) flag questions whose disagreement falls
OUTSIDE the calibrated range as "out-of-calibration — verdict unreliable." The CRC
analog of the energy probe's coverage guard: don't extrapolate the signal.

---

## 4. Error handling and failure modes

Principle: every failure -> honest "can't tell you," never a fabricated verdict.

| Failure | Fail-safe |
|---|---|
| Calibration corr fails permutation null / LOCO non-positive | `UNVALIDATED`, no threshold |
| High pairwise-error correlation | guardrail warning, confidence down-weighted, agreement labeled "may be hollow" |
| Decoder family errors/won't load | drop it, report panel size used; panel < 2 -> abort |
| Embedder unavailable | fall back to judge-equivalence metric, or abort with clear message; never fabricate vectors |
| Model emits empty/garbled output | recorded as real degenerate answer -> maximal disagreement, flagged (a broken model IS a signal) |
| Phase-2 question out-of-calibration | marked "out-of-calibration — verdict unreliable" |
| Answer-key match ambiguous (open-ended) | correctness scorer reports match confidence; low-confidence excluded from calibration, not guessed |
| Tiny benchmark (too few questions) | calibration refuses below min N (corr on a handful of points is noise) |

**Contract:** a calibrated verdict is issued only when panel >=2 working families,
embedder produced real vectors, benchmark >= min N, corr cleared null + LOCO, and
the question is in-calibration-range. Else: honest status string.

---

## 5. Testing

No mock, real models for heavy tests, honest failures never tuned away.

**Pure-logic (every commit):** disagreement-metric math vs CodecGuard;
refuse-below-min-N; refuse-when-null-not-cleared (the flaw-catcher equivalent —
uncorrelated input must yield UNVALIDATED); correlated-error-guardrail-fires;
out-of-calibration-flagged; panel-below-two-aborts; no-verdict-without-calibration.

**Heavy (real models, marked `heavy`):** embedder-produces-real-vectors (Gate 0);
panel-answers-real-questions (+cache); end-to-end-calibration-real (returns a
calibrated threshold with measured P/R OR honest UNVALIDATED — both valid, a
fabricated number is not); phase2-report-real (no card claims an unearned verdict).

**Acceptance gates (staged):**
- Gate 0: a real local embedder works (MiniLM cached) — confirmed; no mock.
- Gate 1: 100% of pure-logic honesty gates pass.
- Gate 2: Phase 1 runs on a real small benchmark and honestly reports whether the
  signal works.

**Pre-committed honest outcomes:** if cross-family disagreement does NOT predict
wrongness on the benchmark, that is a publishable negative ("naive cross-family
disagreement is insufficient; measured evidence here") pointing toward Approach-C
multi-metric work or the correlated-error research — reported, not hidden.

**Does / doesn't:** proves (or honestly fails to prove) that cross-family
disagreement predicts LLM wrongness on a real benchmark, with the correlated-error
caveat measured. Does NOT establish a production hallucination detector,
generalize across domains (calibration-transfer guarded), or claim the signal
works where Phase 1 didn't measure it.

---

## 6. Next step

Implementation plan via writing-plans, decomposing into TDD tasks starting with
Gate 0 (wire the real MiniLM embedder, no mock), then the calibration honesty
gates (pure logic), then the heavy real-model end-to-end calibration. Reuse
`codec_contest` and `dern_b.mlx_backend`; keep CRC self-contained in `src/crc/`
so the existing 84 tests stay green.
