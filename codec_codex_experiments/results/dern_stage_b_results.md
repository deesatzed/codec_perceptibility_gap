# DERN Stage-B Results — real model, measured cost

Stage B of DERN (`docs/plans/2026-05-31-dern-stage-b-design-and-plan.md`): the
distinction-governed cascade ported from synthetic agents to **real local
models**, producing a **measured** efficiency result.

**Scope statement (read first):** This is a **bounded-loss (Lane 2)** cascade
measured on a **small fixed prompt set (9 prompts)** with **two specific local
models**. It is real and measured, not a mock — but it is a single-machine,
small-sample developmental result, not a benchmark claim. Lane 1 (exact,
byte-identical) is **deferred** (needs white-box token-acceptance control).
**Joules are not measured** here (powermetrics needs sudo, which the harness will
not run unattended or fake); energy is reported via measured proxies.

## Setup (user-selected models, per policy)

| Role | Model (local 4-bit MLX, on disk) | Runtime |
|---|---|---|
| cheap | `mlx-community/gemma-4-e4b-it-OptiQ-4bit` (~6 GB) | mlx-lm 0.31.3 |
| reference (authority) | `mlx-community/gemma-4-31B-it-OptiQ-4bit` (~21 GB) | mlx-lm 0.31.3 |

Hardware: Apple M4 Pro, 64 GB. Both models are the same `gemma4` family (clean
cascade). The cheap answer is served only if it agrees with the reference within
ε on token-F1 (`epsilon = 0.5`); otherwise the reference is served. The reference
model's answer is the authority/baseline.

**Run:** `python -m src.dern_b.run_stage_b` (max_tokens=512, ε=0.5, seed=0).
**Tests:** `pytest tests/ -q -m "not heavy"` -> 68 passed (54 pre-existing + 14
DERN-B); 3 heavy real-model tests pass when run with `-m heavy`.

## Measured result (9 prompts)

| Metric | Value | Tag |
|---|---|---|
| cheap acceptance rate | **0.889** (8/9 served by cheap, verified) | — |
| escalation rate | 0.111 (1/9 escalated to reference) | — |
| mean token savings / request | **25.4 tokens** | measured |
| mean latency savings / request | **17.3 s** | measured (wall-clock) |
| mean active-param-seconds savings / request | **~1.19e11** | measured (compute proxy) |
| served worse than reference | **0** | safety invariant held |
| joules (cascade) | **measured** (see Energy section) | `measured` via powermetrics, validated |

**Reading (honest):**
- On this prompt set, the cheap 6 GB model's answers agreed with the 31B
  reference within ε on 8 of 9 prompts, so DERN served the cheap model and
  **skipped the slow reference**, saving a measured ~17 s of wall-clock and ~25
  tokens per request — at the cost of one escalation where the cheap answer
  disagreed and the reference was served instead.
- **The safety floor held absolutely:** zero requests were served an output worse
  than the reference. Every served answer was either verified-within-ε of the
  reference or the reference itself.
- The large active-param-seconds figure reflects the ~5x parameter gap (e4b vs
  31B) combined with the reference's much longer reasoning time — it is a
  measured compute proxy, not joules.

## Energy (measured via powermetrics, validated)

Measured in a user-run terminal (sudo can't prompt in the assistant's shell).
The probe was audited and fixed across three rounds against real output — see
`results/dern_stage_b_energy_method.md` for the 8 flaws found and fixed (parser
phantom samples, ~7x window inflation, pipe-truncation, no idle baseline,
load-vs-inference attribution). Final reading is coverage-validated.

| Quantity | Value |
|---|---|
| avg power, active (cascade) | 18.19 W (total system) |
| avg power, idle baseline | 0.66 W |
| sample coverage | 89% of the 39.7 s work window (177 samples; >60% guard) |
| batch energy, idle-subtracted | 695.9 J (6 routes, models pre-warmed) |
| **per-route mean, idle-subtracted** | **~116 J** |

**Honest scope:** total-system power (not model-only); idle-subtracted is the
better proxy. Models were pre-warmed so this is *inference* energy, not the
one-time 21 GB load. This is NOT yet a savings claim vs always-reference — that
needs an always-31B batch measured under the same probe (clean next step). It
establishes that the harness produces a *trustworthy measured-joules* number for
the cascade, with the validity checks (coverage, idle separation, window
alignment) that the earlier flawed readings (~15.5 J, ~393 J) lacked.

## Important caveats (do not overread)

1. **Audit cost is real and currently amortized by replay, not per-request.** To
   *verify* a cheap answer the harness runs the reference too — so a *first,
   audited* encounter of a prompt region does not save compute. Savings come from
   the experience graph: once a region is trusted, future hits replay the cheap
   route and the reference is sampled only at `audit_prob`. The reported
   per-request savings reflect the served-vs-reference comparison; a production
   deployment must keep `audit_prob` low enough that amortized audit cost stays
   below the savings (the net-of-overhead accounting is built in, but this small
   run does not stress it).
2. **Bounded-loss, not lossless.** ε=0.5 token-F1 is a loose agreement bar; a
   tighter ε lowers acceptance and savings. The number is an operating point, not
   a free lunch. Exact-lossless Lane 1 is deferred.
3. **Small sample, two models, one machine.** 9 prompts is a smoke test, not a
   benchmark. Different model pairs / tasks / ε will give different numbers.
4. **Reasoning-model formatting.** These OptiQ builds emit a harmony reasoning
   channel (`<|channel>thought...<channel|>ANSWER`); the harness extracts the
   model's own final answer verbatim (`extract_final_answer`). Early runs showed
   0% acceptance purely from token truncation + unparsed channel — diagnosed and
   fixed honestly (more tokens + answer extraction), not by tuning ε.

## What this does and does not show

**Does:** the DERN control logic transfers to real models and produces a
**measured** efficiency win (latency + tokens + compute proxy) at a **hard safety
floor** (never worse than the reference), with an honest manifest (no faked
joules). The verifier-as-governor and experience-graph-replay mechanisms work on
real model outputs.

**Does not:** establish a benchmark-grade savings number, a measured-joules
energy claim, exact-lossless operation, or that this beats existing cascade
methods. Those need: a larger task suite, the sudo-gated energy probe run by the
user, Lane 1, and a head-to-head vs FrugalGPT/RouteLLM/LYNX.

## Optional next steps (gated on user)

- **Measured joules:** prime sudo in the session (`! sudo -v`) then re-run; the
  energy probe will then tag joules `measured` instead of `unavailable`.
- **Larger/again:** expand the prompt set, sweep ε to plot the
  acceptance-vs-savings curve (the CodecBench recovery-curve analog).
- **Prior-art:** re-verify (MARLIN 2605.13496, LYNX 2512.05325, etc.) before any
  external claim; distinguish on per-input routing + co-learned stop +
  verifier-as-governor + energy-indexed experience graph.
