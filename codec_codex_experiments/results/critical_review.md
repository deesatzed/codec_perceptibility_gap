# Critical Methodological Review — Codec Codex Experiments

Author: Codex review pass. Scope: `src/proof_ladder.py`, `src/codec_contest.py`, `src/magnetic_pendulum.py`, and the demo + confirmatory_local results. Every claim below is backed by either a code line reference or a re-run diagnostic (probe outputs are reproduced inline). This is an adversarial review: the goal is to find what is wrong or overclaimed, then propose mitigations. Findings are ranked by severity.

---

## Summary of findings

| # | Finding | Severity | Evidence | Mitigation status |
|---|---|---|---|---|
| F1 | Magnetic "expanded proxy" channel contains a noisy copy of the label/divergence target → **label leakage by construction** | **Critical** | Label-channel-only accuracy = 0.872 at m=5 vs noise ceiling 0.88; native-only = 0.877 | Concrete patch proposed |
| F2 | Magnetic slope gate is self-contradictory: requires leaked channel to beat an already-near-ceiling native baseline by 0.10 | High | Gate cond2 `mean(exp) > mean(nat)+0.10` = 0.935 > 1.02 → impossible | Redefine gate |
| F3 | CodecGuard "true error" measured against the **ensemble of the same codecs** whose disagreement is the predictor → semi-circular positive control | High | corr 0.78 against own ensemble; but a clean common-mode synthetic gives −0.03, so not pure tautology — heteroscedasticity is real | Add held-out oracle |
| F4 | S5 "sham" control shuffles train and test with **independent** permutations → destroys row alignment in both, so it tests "noise vs signal," not "shuffled-but-matched" | Medium | code lines 315–321; sham ≈ native at scale | Replace control |
| F5 | Single ground-truth simulator: every stage uses the **same** linear oscillator generator + same feature library → claims do not generalize beyond this one source family | Medium | `make_dataset(2, ...)` everywhere | Add source diversity |
| F6 | `n_boot` fixed seed (99 / 123) and tiny seed counts (1 demo, 5 confirmatory) → CIs are within-sample, not across-data-generating-seed | Medium | `boot_gap` rng=99; `bootstrap_corr_ci` seed=123 | Vary seeds, report across-seed SD |
| F7 | Report wording divergence: auto-report says magnetic "Slope gate: FAIL" using one gate, while module's `slope_gate_passed` is a *different* 3-condition gate | Low | report.py:168 vs magnetic_pendulum.py:254 | Unify gate source |
| F8 | nRMSE averages per-target RMSE/SD across heterogeneous targets (springs, g, c) with very different scales/identifiability → a single scalar hides per-target failure | Low | `nrmse` proof_ladder.py:169 | Report per-target |

---

## F1 — Magnetic expanded channel is label leakage (CRITICAL)

**What the code does.** `run_magnetic_one` builds the "expanded proxy" feature matrix as:

```python
# magnetic_pendulum.py:195-196
expanded_tr = np.concatenate([native_tr, train_label_channel, train_div_channel], axis=1)
expanded_te = np.concatenate([native_te, test_label_channel, test_div_channel], axis=1)
```

`train_label_channel` / `test_label_channel` are **one-hot encodings of the true basin label with a fraction `flip_prob` of labels flipped** (`_noisy_label_channel`, lines 90–97). `train_div_channel` is **the true divergence target plus Gaussian noise**. So the "expanded" feature set literally contains a noisy copy of both prediction targets.

**Why this invalidates the headline interpretation.** The report (and my own first-pass summary) read "expanded proxy ≥ native on basin accuracy" as evidence that an *additive-access channel recovers physical structure the native observer misses*. It does not. It measures that the model was handed the answer with 12% label noise. Probe (confirmatory config, m=5):

```
accuracy from LEAKED LABEL CHANNEL ALONE (no physics features): 0.872
accuracy from native preview features alone:                    0.877
theoretical ceiling from a 12%-flipped label (1 - flip):        0.880
```

The leaked channel alone reaches the noise ceiling (0.872 ≈ 0.88) using **zero** physics. The "expanded" result is therefore an upper bound set by `flip_prob`, not a measurement of access. The same applies to the divergence-nRMSE advantage (≈0.28 vs ≈0.83): the divergence target is in the features.

**This is the single most important correction to my prior report.** My earlier statement that "the expanded channel materially out-predicts native on divergence — the clearest positive signal" was wrong: that advantage is leakage, not signal.

**Mitigation (proposed, not yet applied — needs your approval to change experiment specs):**
- The "expanded" channel must be built from *physical observables the native channel omits* (e.g. the full preview trajectory, cross-coordinate interaction moments, longer preview window, magnet-relative geometry) — **never** the label or divergence target itself.
- If the intent is to model "an oracle side-channel with bounded noise" as a *positive control*, then it must be **labeled as a leakage positive-control**, reported separately, and its accuracy interpreted as "noise ceiling check," not "access recovery." Keep it, but rename it `oracle_leak_positive_control` and add a genuine `expanded_physics` channel for the real comparison.

---

## F2 — Magnetic slope gate is self-contradictory (High)

**What the gate requires** (`magnetic_pendulum.py:254-258`):

```python
gate = bool(
    min(expanded_acc) > max(1/m) + 0.20          # cond1
    and np.mean(expanded_acc) > np.mean(native_acc) + 0.10   # cond2
    and expanded_acc[0] >= expanded_acc[-1] - 0.03           # cond3
)
```

Decomposed on the confirmatory run:
```
cond1 min(exp)=0.902 > chance+0.20=0.533   -> True
cond2 mean(exp)=0.935 > mean(nat)+0.10=1.02 -> False   (impossible: RHS > 1.0)
cond3 exp[0]=0.968 >= exp[-1]-0.03=0.872     -> True
```

cond2 demands `mean(native) + 0.10`. Native preview accuracy is already ~0.92 (it is a good observer), so the requirement is ~1.02 — **above the 1.0 ceiling**. The gate can only pass if the native channel is *bad* and the leaked channel is *strong*. With F1 fixed (no leaked label) the gate becomes even harder to satisfy honestly. The gate is mis-specified relative to the system it scores.

**Mitigation:** redefine cond2 to test *graceful degradation of the honest expanded-physics channel relative to chance*, not a fixed +0.10 margin over an already-strong native baseline. E.g. `min(expanded_physics_acc) > chance + margin AND expanded_physics_acc non-collapsing across m`. Drop the native-margin condition or replace it with a paired-bootstrap significance test of (expanded_physics − native).

---

## F3 — CodecGuard true-error is semi-circular (High, but partially exonerated)

**What the code does** (`codec_contest.py:86-89`): the "true error" each item is scored against is the error of the **ensemble mean of the same four codecs**, and the predictor is the **cross-codec std** of those same codecs:

```python
ensemble = preds.mean(axis=0)
true_err = standardized_l2(ensemble, tte)   # error vs real targets tte
disagree = disagreement_score(preds, tte)   # std across codecs
corr = corrcoef(disagree, true_err)
```

The concern: std and |mean − truth| can be mechanically linked. **However**, `tte` here *is* the real simulator target, not the ensemble — I misread this on first pass; the ensemble error is measured against ground truth, which is legitimate. To test whether the 0.78 correlation is an artifact of shared-variance, I ran a synthetic with pure common-mode error:

```
Synthetic (all codecs share a common-mode error term):
   corr(disagreement, ensemble_error) = -0.027
```

So disagreement does **not** trivially track error when errors are common-mode. The observed 0.78 reflects genuine **heteroscedasticity**: hard items have both higher inter-codec spread *and* higher true error. That is a real and useful property — **the finding survives**. The residual problem is weaker: this is a *positive-control* on a single simulator where "hard" is well-defined; it does not establish that disagreement works as a deployment reliability signal. The high pairwise codec-error correlation (0.55) remains a legitimate overconfidence warning.

**Mitigation:** (a) keep the synthetic common-mode check as a regression guard so the metric can't silently become tautological; (b) add a **leave-one-codec-out** variant — predict held-out codec error from the disagreement of the *other* codecs — to remove the self-reference entirely; (c) report disagreement→error as a calibration curve with a permutation null, not only a single correlation.

---

## F4 — S5 sham control destroys alignment in both splits (Medium)

`proof_ladder.py:315-321`: the sham shuffles the coupling features with `rng.permutation` **independently for train and test**. This makes the sham channel pure noise in both splits. The intended control for "is the coupling signal real or could any added columns help?" is usually a *label-preserving* permutation (shuffle within train only, or a phase-randomized surrogate that preserves marginal distribution but breaks the cross-correlation). As written, `expanded ≈ sham` at scale (both ≈0.544) simply because both reduce to "native + uninformative columns." This makes the S5 PASS/FAIL hinge on whether coupling beats native *at all*, which F-diagnostic shows it barely does (coupling features are 68–79% linearly recoverable from native; |corr with z| ≤ 0.18).

**Mitigation:** replace the independent-shuffle sham with a **block/phase-randomized surrogate** of the coupling time series (preserves per-feature distribution, destroys only the cross-oscillator coupling), and add a **partial-information** test: regress native→z, then test whether coupling features predict the *native residual* (the unique-contribution test). This separates "coupling has no information" from "coupling information is redundant with native."

---

## F5 — Single source family limits external validity (Medium)

Every proof-ladder stage and both codec experiments draw from the **same** linear, driven, damped coupled-oscillator simulator with `n_osc=2` (`make_dataset(2, ...)` is hardcoded in S0–S6, S8, and both codec functions). S7 varies `n_osc` but stays linear. The magnetic module is the only nonlinear source, and it is compromised by F1. Consequence: PASS results demonstrate the *measurement logic executes and behaves sensibly on one generator* — they are not evidence about codec behavior across source families, which is what the framing ("source-family extension") implies.

**Mitigation:** parameterize the generator behind a `source` config key (`linear_oscillator`, `nonlinear_oscillator`, `magnetic`) and run the ladder across ≥2 families. Report each gate per source. Keep defaults small for the M4.

---

## F6 — Bootstrap and seed discipline (Medium)

- `boot_gap` uses a fixed `rng = default_rng(99)` and `bootstrap_corr_ci` uses `seed=123`. These give *reproducible* CIs but they are **resampling CIs over test rows of a single dataset**, not CIs over the data-generating process. The demo runs **1 seed**; confirmatory runs **5**. Across-seed variance (the scientifically relevant uncertainty) is never reported as an interval.
- The summary functions average point metrics across seeds but do not report across-seed SD for the headline numbers.

**Mitigation:** report mean ± across-seed SD for every headline metric; raise default seed count to ≥8 in confirmatory (cheap on this hardware — full confirmatory run was ~9 s); keep per-row bootstrap as a secondary within-dataset interval, clearly labeled.

---

## F7 — Two different magnetic "slope gates" (Low)

`report.py:168` prints `Slope gate: PASS/FAIL` from `mag["slope_gate_passed"]` (the 3-condition gate in the module). My earlier report paraphrased the gate note loosely. There is no contradiction in value (both FAIL), but the prose in `report_auto.md` describes the gate intent imprecisely. Low impact; fix for clarity.

**Mitigation:** have `report.py` print the three sub-conditions with their boolean outcomes so the FAIL is self-explaining.

---

## F8 — Scalar nRMSE hides per-target behavior (Low)

`nrmse` (proof_ladder.py:169) averages per-target normalized RMSE across the 4 theta targets (springs×2, g, c). These targets differ in identifiability (damping `c` and coupling `g` are harder to recover from short trajectories than spring constants). A single scalar can show "PASS" while one target is essentially unlearned.

**Mitigation:** emit the per-target nRMSE vector alongside the mean for S0/S5/S6, so a gate can't pass on the back of one easy target.

---

## What survives the review (genuinely sound)

- **S0–S4, S6:** the core proof-ladder gates (sanity, distinction-dependence, throughput-invariance, wrong-lever, corruption-decay, shared-placement) are correctly constructed and pass robustly. No leakage, sensible controls.
- **CodecBench recovery curve:** the grounded-vs-uninformative gap (0.49) is honest — the uninformative channel is corrupted-input, not target-leaked, and the recovery metric is bounded against a real mean-predictor baseline.
- **CodecGuard heteroscedasticity finding (F3):** survives the common-mode null; disagreement genuinely tracks error on this generator, with an appropriate correlated-error warning.
- **S5/S7 failures:** correctly reported as honest negatives/boundary conditions in the prior report; this review only refines *why* (redundancy + weak control for S5; linear-source structure for S7).

---

## Net effect on the prior report's claims

| Prior claim | Revised verdict |
|---|---|
| "Expanded proxy materially out-predicts native on divergence — clearest positive signal" | **Retracted.** That advantage is label/target leakage (F1). |
| "Expanded proxy ≥ native on basin accuracy, consistent with additive-access" | **Retracted as stated.** It reflects a leaked label at the noise ceiling, not access. |
| CodecGuard disagreement→error correlation is a useful triage signal | **Upheld** (survives common-mode null), with the existing overconfidence caveat. |
| CodecBench graceful-degradation vs collapse | **Upheld.** |
| S5 medium/scale-dependent failure; S7 expected boundary | **Upheld and sharpened.** |
| Proof-ladder S0–S4/S6/S8 sound | **Upheld.** |
