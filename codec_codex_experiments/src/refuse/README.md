# refuse — a result type that withholds a number until it's earned

**The contract is the point.** `refuse` gives you one return type for any
quantitative claim: **`Verified(value, receipts)`** or
**`Refused(reason, evidence)`**. The default is to *refuse* — your pipeline does
not get to report a "win," a "detection," or a "saving" until it survives a battery
of adversarial controls, and when it can't, you get a typed reason instead of a
plausible-looking number.

> Not "guardrails" (that word means LLM-output safety). This gates **statistical
> claims** — "B beats A by 1%", "this detector works", "we saved X%" — and refuses
> the ones that don't hold up.

Think `try/except` for over-claiming: wrap the measurement, get back a value with
receipts or a refusal with evidence.

## Honest framing (read this first)

This is **rigor packaged for reuse, not a novel technique.** Every control is
standard statistics — base rate, a difficulty/triviality baseline, a permutation
null, bootstrap-CI separation, sampling coverage, collinearity. The only thing
`refuse` adds is *composition* + a *refuse-by-default contract* + worked examples.
If you want novelty, this isn't it. If you want a cheap way to stop yourself (or a
pipeline) from reporting a number that doesn't hold up, it's useful.

It was distilled from a research program (energy / reliability / agent / carbon
audits) where this exact discipline repeatedly caught overclaims — **including its
own** (it retracted a cherry-picked carbon result; see `carbon_audit/RESULTS.md`).

## Use

```python
from src.refuse import Battery, base_rate, beats_difficulty

battery = Battery([
    base_rate(events=wrong_flags, floor=0.30),                  # enough to detect?
    beats_difficulty(signal, difficulty, label),                # beats the obvious predictor?
])
result = battery.evaluate(value="my_reliability_score")

if result.status == "VERIFIED":
    use(result.value)            # plus result.receipts (why it passed)
else:
    print(result.reason)         # typed refusal + result.evidence (all checks)
```

## Controls

| Control | Refuses when | Standard name |
|---|---|---|
| `base_rate(events, floor, ceil)` | the event to detect is too rare/universal to validate against | base-rate / prevalence |
| `beats_difficulty(signal, difficulty, label)` | the signal does not beat a difficulty-only baseline (bootstrap-CI of the AUC gap not > 0) | skill score / specification check |
| `permutation_null(x, y)` | a correlation doesn't exceed its permutation null | permutation test |
| `collinearity(control_var, label)` | the "control" is near-identical to the label (degenerate comparison) | collinearity / VIF |
| `coverage(sampled_s, work_s, floor)` | a measurement didn't sample enough of the work window | measurement coverage |
| `improvement_beats_noise(scores_a, scores_b)` | a "B beats A" gain's paired bootstrap CI includes 0 (inside run-to-run noise) | "+1% is not enough" / paired bootstrap |

## Worked examples (runnable, real)

```bash
python -m src.refuse.examples.example_plus_one_percent  # START HERE: a +1% 'win' REFUSED, a real gain VERIFIED
python -m src.refuse.examples.example_difficulty        # real signal VERIFIED, difficulty-proxy REFUSED
python -m src.refuse.examples.example_carbon            # a control-dependent carbon claim REFUSED
```

- `example_plus_one_percent` — **the landing demo.** "Model B beats A by ~1%" is
  refused when the paired gain sits inside run-to-run noise; a genuine +5.5% gain
  is verified. The over-claim everyone has shipped, caught by default.

- `example_difficulty` — shows a genuine signal passing and a "difficulty in
  disguise" signal being refused (mirrors the program's MMLU-Pro finding).
- `example_carbon` — uses the **real** Kariba (VCS902) forest-loss sweep measured
  from public Hansen satellite data: the additionality ratio swings 0.15→0.97 by
  control choice, so the claim is **refused** as control-dependent. Illustrative of
  the method on real data, not a verdict on the project.

## What it is / isn't

- **Is:** a small, tested, dependency-light (`numpy`) way to gate any measurement
  behind composable controls and refuse rather than fabricate.
- **Isn't:** novel science, a complete causal-inference suite, or a substitute for
  domain expertise. It encodes a few well-known controls and the discipline of
  withholding a number until they pass.
