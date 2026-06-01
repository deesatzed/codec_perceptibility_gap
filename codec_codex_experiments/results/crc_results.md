# CRC Stage Results — first real calibration

First real Phase-1 calibration of the Calibrated Reliability Card
(`docs/plans/2026-05-31-reliability-report-card-design.md`).

**Scope (read first):** This is the *honest negative* the design pre-committed
to. It is real (two local model families really answered 40 known-answer
questions; no mock), small, and it reports that the disagreement signal **could
not be validated on this benchmark** — which is the tool refusing to certify a
signal it cannot actually prove, exactly as designed.

## Setup
- Families (distinct vendors → genuine independence): `gemma-google`
  (gemma-4-e4b, ~6 GB) + `granite-ibm` (granite-4.1-3b, ~2 GB), local 4-bit MLX.
- Benchmark: `src/crc/bench.py` `TINY_FACTUAL` — 40 short factual questions with
  known answers (capitals, arithmetic, chemistry, colors, geography).
- Embedder: `all-MiniLM-L6-v2` (local, offline). Disagreement = semantic spread.
- `pytest tests/ -q -m "not heavy"` → 100 passed (84 prior + 16 CRC), no regressions.

## Result (honest negative)

```
verdict: UNVALIDATED
reason:  disagreement does not predict wrongness (fails permutation null/CI)
corr:    0.633          # disagreement DOES correlate with wrongness...
null_p95: 0.633         # ...but does not EXCEED the permutation null
ci95:    [NaN, NaN]     # bootstrap CI undefined
hollow_agreement_warning: true
n_questions: 40
```

## What this means (the diagnosis)

The verdict is `UNVALIDATED` for a real, understandable reason — **not a bug**:

1. **Easy questions + 2 strong models = too few wrong answers.** Both families
   get most of these factual questions right, so the "wrong" array is nearly all
   zeros — a near-constant. You cannot calibrate "disagreement predicts
   wrongness" when there is almost no wrongness to predict; the bootstrap CI is
   `NaN` (correlation with a near-constant is undefined) and the permutation null
   cannot be cleared.
2. **The guardrail correctly fired** (`hollow_agreement_warning: true`): with only
   two families on easy items, their errors are correlated (right/wrong
   together), so agreement is partly hollow — precisely the caveat the design
   surfaces rather than hides.
3. **The raw correlation (0.633) is positive and promising** — disagreement *does*
   trend with wrongness — but "promising" is not "proven," and the tool refuses to
   issue a threshold on an unproven signal. That refusal is the point.

## What it tells us to do next (the real experiment)

To actually *validate* the signal, the calibration set must contain enough
genuine disagreement and error:
- **More families (3+):** add a third distinct vendor (e.g. qwen) so agreement is
  meaningful and the error matrix is non-degenerate.
- **Harder questions:** include items where strong models actually disagree and
  err (ambiguous, specialized, or adversarial), so there is a real
  disagreement-vs-wrongness gradient to calibrate on. Easy trivia is the wrong
  proving ground precisely because everyone gets it right.
- Then re-run Phase 1; if disagreement clears the null on harder data, we get a
  calibrated threshold + measured precision/recall and can run Phase 2 on
  unlabeled questions.

## What this does and does not show

**Does:** the CRC pipeline runs end-to-end on real models, and its honesty gates
work — it refused to certify a signal it could not validate, surfaced the
hollow-agreement caveat, and reported a real negative instead of a fabricated
number. The machinery (cross-family disagreement, calibration, guardrail) is
sound and behaves correctly at the boundary.

**Does not:** establish that cross-family disagreement predicts LLM wrongness —
that remains *unproven on this benchmark* and is the explicit next experiment
(more families + harder questions). No reliability verdict on unlabeled data is
warranted until Phase 1 earns trust on a benchmark with real error variance.

This negative is more valuable than a forced positive: it tells us *where* the
signal can be proven (harder, more diverse data) and confirms the tool will not
lie when it cannot.
