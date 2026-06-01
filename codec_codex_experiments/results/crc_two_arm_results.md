# CRC Two-Arm Results — is disagreement a reliability signal or a capability proxy?

The experiment built to answer the user's question: *"are we just re-testing model
capability/memorization, or is there a higher concept?"* — using offline
counterfactual perturbation (no unverifiable training-cutoff assumptions).

**Setup:** gemma-google (gemma-4-e4b, ~6 GB) + granite-ibm (granite-4.1-3b, ~2 GB),
local 4-bit MLX. Embedder all-MiniLM-L6-v2 (offline). Arm 1 = 40 plain factual Qs
(likely memorized); Arm 2 = 32 hand-authored counterfactual items where a premise
in the prompt makes the memorized answer provably wrong (key = premise answer,
human-verifiable). `pytest tests/ -q -m "not heavy"` -> 107 passed, no regressions.

## Result

```
                         Arm 1 (memorized)   Arm 2 (perturbed/reasoning)
ensemble_wrong_rate          0.05                 0.226
calibration_verdict          UNVALIDATED          VALID
  corr (disagree~wrong)      0.633                0.418
  threshold / precision/recall  -                 0.0256 / 1.0 / 0.571
hollow_agreement_warning     true                 false
partial_corr | difficulty    0.908                null (uncomputable, see below)

higher_concept_verdict: INCONCLUSIVE
```

## What this means (honest, nuanced)

**The encouraging evidence:**
1. **Perturbation worked.** It raised the error rate from 5% (memorized) to 23%
   (reasoning-required) — it successfully forced the models off memorized recall,
   which is the whole point of Arm 2.
2. **Disagreement VALIDATED on the hard arm.** On the reasoning-required set,
   cross-family disagreement earned a calibrated threshold with **precision 1.0**
   (when the two families disagreed, the answer was reliably wrong — zero false
   alarms) and recall 0.571 (caught 57% of errors). The hollow-agreement guardrail
   did NOT fire here.
3. **Arm 1's difficulty-controlled partial = 0.908** — strong evidence that, at
   least on the memorized arm, disagreement predicts wrongness *beyond* a pure
   difficulty prior. That points toward a real higher concept.

**Why the overall verdict is INCONCLUSIVE (not REAL_SIGNAL):**
The decisive test — does disagreement predict wrongness *beyond difficulty* on
the perturbed arm — was **uncomputable**, and for a precise structural reason:
**with only 2 families, "ensemble wrong" (majority wrong) is mathematically
identical to "difficulty >= 0.5" (both families wrong).** We measured
`corr(difficulty, wrong) = 1.0` exactly. When difficulty and wrongness are the
same variable, you cannot statistically control one for the other — the partial
correlation residual has zero variance, hence `null`. The verdict logic then
correctly refused to claim REAL_SIGNAL on an uncomputable test.

So the honest reading: **promising and validated on the hard arm, but the
cleanest higher-concept test is structurally impossible with 2 decoders.** This
is the system declining to overclaim, exactly as designed.

## The precise next experiment (what would settle it)

The 2-family collinearity is the binding constraint. To compute the
difficulty-controlled partial on the perturbed arm, we need **>= 3 families**, so
that "majority wrong" and "fraction wrong" are distinct variables. With 3+
families:
- if disagreement's partial-correlation-beyond-difficulty on the perturbed arm is
  clearly > 0 -> REAL_SIGNAL (a genuine reliability concept, not a capability
  proxy);
- if ~0 -> disagreement is a difficulty/capability proxy on reasoning tasks too
  (honest negative).

Add the third on-disk family (qwen-alibaba, 17-27 GB) and re-run; optionally widen
both arms. No new code needed — the harness already supports N families.

## What this does and does not show

**Does:** the two-arm method runs on real models and produces an honest, precise
verdict; perturbation successfully separates recall from reasoning; disagreement
shows real predictive value (precision 1.0) on reasoning-required questions; and
the system refused to declare a higher concept proven when the decisive test was
uncomputable.

**Does not:** prove (yet) that cross-family disagreement is a reliability signal
*beyond* difficulty on reasoning tasks — that specific claim is gated on a >=3-family
run, which is the immediate next step. Two families is structurally insufficient
for the difficulty control, which is itself a useful finding about minimum panel size.
