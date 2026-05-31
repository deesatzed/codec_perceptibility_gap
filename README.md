# Can we measure when a machine's view of the world stops fitting into ours?

*A research log: building the Perceptibility Gap simulation backbone, auditing it until it broke, fixing what broke, and testing whether the findings survive a change of source.*

---

## The question

Most definitions of "superintelligence" are about *problem-solving*: a system that out-reasons us across domains. This program studies a different, more structural worry. A machine may form **distinctions** — internal contrasts that matter for how it acts — that don't compress cleanly into any channel a human can perceive. When that happens, an explanation isn't a window into the system; it's a lossy projection. Fluent, confident, and possibly wrong. We call that failure mode **false transparency**.

You can't test that thesis directly without a superintelligence on hand. So this repository tests the *minimal measurable bridge* the thesis needs: can you build a rate-distortion frontier for machine representations of a dynamical system, separate **how many distinctions** a codec resolves from **how much bandwidth** it has, and show that the distinction count — not the bandwidth — is what governs whether structure is recoverable?

Everything here is **simulation and software**. It makes the measurement logic executable. It does **not** confirm any human-subject hypothesis. We are religious about that line.

The companion academic manuscript is `perceptibility_gap_paper_v1_4.tex` / `.docx` / `.md` (a preregistered two-study program). This page is the engineering story behind its developmental section.

---

## What we built

A one-command bundle (`python -m src.run_all`) that runs four things on a local machine (these results: Apple M4 Pro, Python 3.13, numpy + scikit-learn, no GPU, no network):

1. **The proof ladder (S0–S8)** — nine gated checks. Does direct access beat an uninformative baseline (S0)? Do more distinctions lower error at fixed bandwidth (S1)? Is error invariant to bandwidth once intake saturates (S2)? Does the "wrong lever" (more bandwidth, fewer distinctions) underperform the right one (S3)? And so on through corruption-decay, additive access, shared placement, a dimensionality slope, and a multi-codec audit signal.
2. **CodecGuard** — treats disagreement between independent codecs as a reliability signal. Does disagreement actually predict error?
3. **CodecBench** — compresses a structured channel through fewer and fewer distinctions and measures how much recoverable structure survives.
4. **A magnetic-pendulum stress test** — a chaotic, multistable source where we score only well-posed targets (basin label, finite-time divergence) inside the Lyapunov window, never long-horizon exact paths.

It ran. Eight of nine gates passed on the first real run. It would have been easy to stop there and write it up.

---

## Then we tried to break it

Before believing any of it, we ran an adversarial self-audit (`codec_codex_experiments/results/critical_review.md`). It found eight issues. One was serious enough to invalidate a headline result:

> **The magnetic "expanded" channel was cheating.** It built its input features by concatenating a *noisy copy of the answer* — the basin label and the divergence target — into the feature matrix. A model given only that leaked channel, with zero physics, scored 0.87 basin accuracy against a noise ceiling of 0.88. The earlier "expanded beats native" result wasn't measuring better access to physical structure. It was measuring that we'd handed the model the label.

That's label leakage, and it's the kind of bug that quietly inflates a paper. We retracted the claim.

Two other concerns were real but subtler:
- The additive-access control (S5) used a sham that destroyed information rather than a distribution-matched surrogate — so it tested "signal vs noise" instead of "signal vs redundant."
- CodecGuard's disagreement-vs-error correlation was measured against the ensemble's own error, which risks being a mathematical tautology.

---

## What we fixed (and what the fixes revealed)

The mitigations (`results/mitigation_results.md`) improved **integrity, not headline numbers** — which is exactly what an honest fix looks like:

- **Leakage removed.** The leaked channel is retained but explicitly renamed `oracle_leak_positive_control` (a ceiling check, never evidence). A new honest `expanded_physics` channel uses only real omitted observables — velocity profile, radial dynamics, a cross-coordinate interaction moment. Under the honest channel, expanded physics does **not** beat the native preview; the full trajectory (`direct`) wins. The opposite of the leaked conclusion.
- **CodecGuard de-tautologized.** Added a leave-one-codec-out estimate (predict a held-out codec's error from the *other* codecs' disagreement) and a permutation null. The signal **survives**: correlation 0.75, LOCO 0.62, null 95th-percentile only 0.07, cleared on every seed. A synthetic common-mode control confirmed it isn't an artifact. The correlated-error warning (pairwise 0.57) stays — agreement among correlated codecs can be confidently wrong.
- **S5 control hardened.** A phase-randomized surrogate plus a unique-contribution test (does coupling predict the native channel's residual?). On the linear family the answer is no — residual nRMSE 1.003, no better than the mean. An honest negative, sharper than before.

---

## The structural gap, and closing it

Even after the fixes, one limitation remained: **every gate ran on a single source family** (linear coupled oscillators). A PASS there shows the logic is coherent on one generator — it doesn't show the logic *generalizes*.

So we built a source-family registry behind a single config switch (`src/sources.py`, `src/cross_family.py`), added a **Duffing-type nonlinear oscillator** (cubic stiffness), froze the linear numbers in a regression test so the refactor couldn't silently change them, and added a per-family no-leakage test so the leakage bug above can never come back. Then we ran the whole ladder across both families and built a **gate matrix**.

### The result (8 seeds, three families across two dynamical classes)

| Gate | Claim | linear | nonlinear | Hénon (chaotic) | Verdict |
|---|---|:---:|:---:|:---:|---|
| S0 | sanity | ✅ | ✅ | ❌ | family-robust |
| S1 | distinction-dependence | ✅ | ✅ | ✅ | family-robust |
| S2 | throughput-invariance | ✅ | ✅ | ✅ | family-robust |
| S3 | wrong-lever | ✅ | ✅ | ✅ | family-robust |
| S4 | corruption-decay | ✅ | ✅ | ✅ | family-robust |
| S5 | additive-access | ❌ | ✅ | ✅ | family-robust |
| S6 | shared-placement | ✅ | ✅ | ❌ | family-robust |
| S7 | dimensionality-slope | ❌ | ❌ | ❌ | **fails on all** |
| S8 | multi-codec-audit | ✅ | ✅ | ❌ | family-robust |

**Eight of nine gates are family-robust** (PASS on ≥2 families). The core measurement logic holds across linear, nonlinear, *and* discrete-time chaotic sources. Only S7 fails everywhere — and it's the gate we already knew was mis-designed.

---

## Closing the harder gap: three external criticisms

A structured external methods review (`codec_codex_experiments/critic1.md`) raised three sharper points. We mitigated all three (`results/critic1_mitigation_results.md`), keeping the same rule: no threshold tuned to force a pass.

**1. "The linear ladder is brittle — one middle gate fails and the whole back end is in jeopardy."** Fixed by **parallel verification tracks**: the same gates, regrouped into three independent lanes. The information-theoretic lane (S0–S3) now PASSes 4/4 *on its own*, no longer marginalized by the S7 quirk sitting in the separate complexity lane.

**2. "Your machine ceiling uses toy estimators; a transformer could move it and break your margins."** Fixed by a **bridging validation** that runs the proposed confirmatory MLP against Ridge on the *exposed developmental sets only* — never the reserved holdout. The MLP *does* move the ceiling (up to −25%), confirming the concern was real; at the confirmatory data budget the throughput-invariance margins still hold. The rule is now data-backed: re-derive the margin from the stronger estimator if it ever flips.

**3. "Two coupled-oscillator families are still 'two lakes' — both have momentum and periodic attractors."** Fixed by adding a **discrete-time chaotic family** (a coupled lattice of Hénon maps) with no continuous momentum and no periodic attractor. The headline test — *does the CodecGuard audit signal survive without oscillator physics?* — comes back **yes**: weaker on the chaotic family (naive correlation 0.20 vs ~0.75) but still clearing the permutation null on every seed, with a positive leave-one-codec-out correlation (0.24) and a worst-error catch rate (0.36 > 0.25 baseline).

And the honest limit it exposed: on the chaotic family the *parameter-recovery* gates fail, because chaos parameters are barely recoverable from coarse orbit statistics — sensitive dependence, not a bug. So we can audit *relative reliability* in an alien dynamical class, but not necessarily read off its generating parameters. We report that as the finding it is.

> **The rule we never broke:** no gate threshold was ever changed to make a family pass. A FAIL is a published negative. (S5's earlier linear-family FAIL still stands; it just became family-robust once two of three families pass it.)

---

## What this does and doesn't show

**Does:** the measurement apparatus is executable, internally coherent, survives an adversarial audit, and generalizes across three source families spanning two dynamical classes (oscillator and discrete-time chaotic) for 8 of 9 gates. The CodecGuard audit signal is real, not a tautology, and survives even on a non-oscillatory chaotic source. The additive-access effect is real but family-dependent.

**Doesn't:** establish human learning, LLM deployment reliability, or any horizon-access claim. The magnetic test is a nonlinear *stress test*, not proof of horizon access. Multi-codec contrast is a candidate *audit layer*, not a solution to alignment — and correlated errors remain its primary failure mode.

---

## Run it yourself

```bash
cd codec_codex_experiments
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                                    # 17 tests
python -m src.run_all --config configs/demo.json --out results/demo            # fast, single family
python -m src.run_all --config configs/confirmatory_local.json --out results/critic1_mitigation   # 8 seeds, 3 families + gate matrix + bridging validation
```

Then read `results/critic1_mitigation/report.md` and `results/critic1_mitigation/results.json`.

---

## Map of the repository

| Path | What's there |
|---|---|
| `codec_codex_experiments/` | the runnable package (code, configs, tests, results) |
| `codec_codex_experiments/src/` | `proof_ladder.py`, `codec_contest.py`, `magnetic_pendulum.py`, `sources.py`, `cross_family.py`, `bridging_validation.py`, `report.py`, `run_all.py` |
| `codec_codex_experiments/critic1.md` | the external methods critique |
| `codec_codex_experiments/results/critical_review.md` | the 8-finding adversarial audit |
| `codec_codex_experiments/results/mitigation_results.md` | before/after evidence for the F-series fixes |
| `codec_codex_experiments/results/cross_family_results.md` | the v1.4 cross-family writeup |
| `codec_codex_experiments/results/critic1_mitigation_results.md` | the v1.5 critique mitigations (tracks, bridging, Hénon) |
| `codec_codex_experiments/results/critic1_mitigation/` | canonical run output (results.json, report.md, figures) |
| `codec_codex_experiments/docs/PLAN_F5_cross_family.md` | the plan that produced the cross-family work |
| `perceptibility_gap_paper_v1_5.md` / `.tex` / `.docx` / `.pdf` | the academic manuscript (preregistered two-study program) |

Earlier manuscript versions (`v1_2b`, `v1_3`, `v1_4`) are kept for provenance. `v1_5` is current.

*These are developmental simulation results. Read the manuscript for the full claim ladder and the preregistration discipline.*
