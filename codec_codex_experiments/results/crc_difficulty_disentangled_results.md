# CRC Difficulty-Disentangled Evaluation — the decisive result

The experiment that answers the question cleanly, with every prior confound
removed: *is cross-family disagreement a reliability signal beyond model
capability/difficulty?* Run on hard, error-rich data, with >=3 independent
families, using log-likelihood scoring (no parsing/truncation).

**Setup:** 3 families — gemma-google (gemma-4-e4b), qwen-alibaba (Qwen3.6-27B),
granite-ibm (granite-4.1-3b), local 4-bit MLX. Benchmark: **MMLU-Pro** (120 items,
the established hard MC benchmark). Scoring: **log-likelihood over option tokens**
(standard LLM MC method; one forward pass; truncation-proof; unaffected by verbose
reasoning models — this fixed the 3 prior answer-format failures). Disagreement =
cross-family spread of option-probability distributions. `pytest tests/crc/eval -q`
-> 27 tests pass.

## Result

```
ensemble_error_rate:  0.783        # error-rich: gate 1 PASSED (was the prior blocker)
families:             3 distinct   # collinearity gate 2 PASSED
verdict:              DIFFICULTY_PROXY

raw_corr (disagree~wrong):            0.415   # disagreement IS positively correlated
partial_corr | difficulty:            0.239   # ...some residual after controlling difficulty
naive-difficulty baseline:
  auc_signal (disagreement):          0.829
  auc_baseline (difficulty alone):    1.000   # difficulty predicts wrongness PERFECTLY
  gap (signal - baseline) CI95:       [-0.272, -0.084]   # disagreement is WORSE, significantly
  wins:                               false
```

## What this means (the clean, decisive answer)

**On hard, error-rich data, cross-family disagreement is a capability/difficulty
proxy — not a higher reliability concept.** And this time it is established without
any of the prior escape hatches:
- NOT an error-scarcity artifact (78.3% error — abundant wrongness to predict).
- NOT a 2-family collinearity artifact (3 independent families; collinearity gate passed).
- NOT an answer-parsing/truncation artifact (log-likelihood scoring, no generation).

The decisive number is the naive-difficulty baseline: **question difficulty alone
predicts ensemble-wrongness at AUC 1.0; disagreement predicts it at 0.829 — strictly
worse, with the gap CI entirely below zero.** Disagreement carries no reliability
information beyond the difficulty prior; here it carries less. The small positive
partial (0.239) is not enough to clear the baseline — hence `DIFFICULTY_PROXY`, not
`REAL_SIGNAL`.

(Why difficulty AUC = 1.0: with 3 families, majority-vote wrongness is near-fully
determined by the fraction of families wrong = difficulty. Not perfectly collinear
— the gate passed at <0.95 — but difficulty is a near-complete predictor of the
ensemble label. This is honest and expected, and it is exactly why a signal must be
tested *against* difficulty, not just *for* correlation with wrongness.)

## Why this is a real contribution (a rigorous negative the field needs)

This **corrects the leading 2026 result.** arXiv 2603.25450 (Mar 2026,
"Cross-Model Disagreement as a Label-Free Correctness Signal") claims cross-model
disagreement is a useful correctness signal — at this exact model scale — but
**does not run a difficulty-controlled baseline.** Two other 2025-2026 papers
(2509.19372, 2508.08285) independently show hallucination/uncertainty detectors
are confounded by difficulty/artifacts and recommend exactly the naive-baseline
control we implemented. We ran that control on an established hard benchmark and
the disagreement signal does not survive it.

The contribution is the **difficulty-disentangled evaluation protocol** + the
honest negative: a label-free signal must beat a difficulty baseline (bootstrap-CI
of the AUC gap above zero), not merely correlate with wrongness. Under that bar,
cross-family disagreement fails on MMLU-Pro.

## What this does and does not show

**Does:** on a hard, error-rich, established benchmark, with >=3 independent
families and the difficulty control the field omits, cross-family disagreement is
a difficulty proxy (AUC 0.829 vs difficulty 1.0; gap CI [-0.27,-0.08]). The
evaluation harness, gates, and log-likelihood scoring all function on real models;
every prior confound (error-scarcity, collinearity, parsing) is removed.

**Does not:** prove disagreement is useless for ALL purposes — it may still serve
as a cheap abstention/routing TRIGGER (you don't need to beat difficulty to gate an
action), and a different signal (self-consistency/semantic-entropy) or a different
task type could differ. It does show disagreement is not a *reliability metric
superior to difficulty* on MC reasoning tasks.

## Bottom line

The full arc — energy (DERN), reliability (CRC), and now the difficulty-disentangled
evaluation — converges on one durable asset: **a measurement discipline that
repeatedly catches its own (and the field's) false positives.** Here it turned a
plausible, citable claim ("disagreement predicts correctness") into a controlled,
honest negative on real models. That negative, with its protocol, is the publishable
result — and it is more trustworthy than the positive it corrects precisely because
it survived the controls the positive skipped.
