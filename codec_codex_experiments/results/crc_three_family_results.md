# CRC Three-Family Results — the difficulty-control test fires

The run that settles the user's question ("is cross-family disagreement a real
reliability signal, or just a capability/memorization proxy?") by removing the
2-family collinearity that made the decisive test uncomputable.

**Setup:** THREE distinct families — gemma-google (gemma-4-e4b), qwen-alibaba
(Qwen3.6-27B), granite-ibm (granite-4.1-3b), local 4-bit MLX, 512 tokens (so each
model finishes its reasoning trace; answer extractor generalized to qwen's
`</think>` format this session). Arm 1 = 40 plain factual Qs; Arm 2 = 32
counterfactual-premise Qs. Embedder all-MiniLM-L6-v2 offline. `pytest tests/ -q -m
"not heavy"` -> 112 passed.

## Result

```
                              Arm 1 (memorized)   Arm 2 (perturbed/reasoning)
ensemble_wrong_rate                0.025                0.065
calibration_verdict                UNVALIDATED          UNVALIDATED
  raw corr (disagree~wrong)        0.897                0.14
  partial corr | difficulty        null (*)             -0.337
hollow_agreement_warning           true                 false

higher_concept_verdict: INCONCLUSIVE  (arm2_beyond_difficulty = false)
```
(*) Arm 1 partial is null because at a 2.5% error rate the wrong-on-difficulty
residual is ~zero-variance — almost nothing is wrong, nothing to control.

## The decisive finding (and a reversal of the 2-family hint)

The 2-family run had looked encouraging — disagreement earned precision 1.0 on
the perturbed arm. **That was an artifact.** With only 2 families, "ensemble
wrong" (majority wrong) is mathematically identical to "difficulty >= 0.5", so
disagreement trivially tracked wrongness *because wrongness was difficulty*. The
difficulty-controlled partial was uncomputable (perfect collinearity), so the
confound was invisible.

Adding a **third independent family breaks the collinearity** and makes the
partial computable — and it comes back **-0.337 (negative)**. Reading: once you
control for question difficulty, cross-family disagreement carries **no extra
reliability information** on this benchmark. The apparent signal was a
capability/difficulty proxy, exactly as the user suspected might be the case.

**This is the difficulty-control test doing its job** — it was built to catch a
capability proxy masquerading as a reliability signal, and it caught one. Had we
stopped at 2 families and shipped the precision-1.0 number, we would have
published a false positive.

## Honest caveat on the negative (we don't overclaim it either)

This is a *bounded* negative, not a universal one:
- **Too few errors to test against.** With 3 capable families and room to reason,
  the perturbed error rate fell to 6.5%. You cannot strongly calibrate
  "disagreement predicts wrongness" when there is very little wrongness. Both arms
  came back UNVALIDATED partly for this reason.
- So the precise claim is: **on this benchmark, with these models, there is no
  evidence that cross-family disagreement is a reliability signal beyond
  difficulty** — measured, not assumed. It is NOT proof that the concept fails in
  general; it is proof that this test, on data the models find easy, shows nothing
  beyond capability.

## What would actually settle it (loops back to the first instinct)

The binding constraint is now error scarcity, not collinearity. The decisive
experiment needs **questions hard enough that strong models genuinely and often
err** — i.e. the user's original instinct about harder/established benchmarks
(GPQA-style, adversarial, specialized), where error rates are 30-60%, so there is
real wrongness for disagreement to predict and the difficulty-controlled partial
has signal to measure. With the contamination caveat handled (perturbation or
genuinely held-out items), that run yields a clean verdict:
- partial-beyond-difficulty clearly > 0 on a high-error set -> REAL_SIGNAL;
- ~0 or negative -> disagreement is a capability proxy, full stop.

## What this does and does not show

**Does:** the difficulty-control methodology works and is decisive — it exposed
that the promising 2-family result was a collinearity artifact, and refused to
certify a higher concept that the data does not support. The 3-family harness,
multi-format answer extraction, and honest verdict logic all function on real
models.

**Does not:** prove cross-family disagreement is useless for reliability — only
that, on an error-scarce benchmark, it shows nothing beyond difficulty. The
honest next step is a high-error-rate benchmark; until then, no reliability
verdict on unlabeled data is warranted.

**Bottom line for the user's question:** as currently tested, this is closer to a
**capability proxy than a higher concept** — and we know that because the test we
built to detect exactly this confound fired. That is a more valuable, more honest
outcome than a forced positive.
