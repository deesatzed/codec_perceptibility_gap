# DERN Stage B — Design + Implementation Plan (real model, measured cost)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port DERN from synthetic agents to a real local model and produce an
**honest, measured** efficiency result: a distinction-governed cascade that
serves a cheap model when its output is verified good-enough and escalates to a
reference model otherwise, reporting measured tokens + latency + active-parameter-
seconds saved at bounded quality loss, net of router cost.

**Architecture:** A new `src/dern_b/` subpackage reusing the Stage-A control
spine (experience graph, trust breaker, online controller, ledger, two-lane
verifier concept) but with **real-model components**: an mlx-lm backend, a
distinction probe over the prompt, and a **bounded-loss cascade** (Lane 2)
audited against the reference model. The runtime is local on Apple Silicon.

**Tech Stack:** Python 3.13, **mlx-lm 0.31.3 + mlx 0.31.2** (installed), the
existing `src/dern/` modules (graph, breaker, controller reused as-is), pytest.
No new heavy deps beyond mlx-lm. No mocks — tests load the **real** local models.

---

## Locked decisions (from the design dialogue)

- **Runtime:** local **mlx-lm** on Apple M4 Pro / 64 GB (clean energy attribution).
- **Models (user-selected, same family, both local 4-bit MLX, both materialized):**
  - cheap: `/Volumes/WS4TB/models/mlx-community/gemma-4-e4b-it-OptiQ-4bit` (~6 GB)
  - reference: `/Volumes/WS4TB/models/mlx-community/gemma-4-31B-it-OptiQ-4bit` (~21 GB)
  - Verified: cheap model loads in ~2.8 s and generates real text. No mock.
- **Lanes:** **bounded-loss cascade (Lane 2 only)**, audited against the
  reference model as the authority at tolerance ε. **Lane 1 (exact speculative,
  byte-identical) requires white-box token-acceptance control and is deferred**
  to a documented local-follow-up; it is NOT claimed for Stage B.
- **Headline metric (honesty boundary — critical):**
  - **MEASURED, no privilege:** tokens (prompt + generated), wall-clock latency,
    and **active-parameter-seconds** (model active-param count × decode seconds),
    a real compute proxy.
  - **TRUE JOULES:** `powermetrics` requires `sudo` (a password) and cannot run
    unattended. Energy in joules is therefore an **opt-in** measurement the user
    runs with `sudo`; when unavailable it is tagged `unavailable`, and an
    estimate may be tagged `derived` (active-param-seconds × a published J/param-s
    constant) but is **NEVER** tagged `measured`.
  - Every cost dimension carries a `measured | telemetry-derived | derived |
    simulated | unavailable` tag — the honesty manifest.
- **Quality:** bounded loss means the cheap output is accepted only if it agrees
  with the reference output within ε on a task-appropriate metric (exact-match /
  token-F1 for short factual tasks). Accuracy is measured against the reference
  model's full-compute answer as the authority (the cascade's reference is the
  baseline, exactly as in Stage A).

## Prior-art posture (carried from the 2026-05-31 re-check)

Stage B is a **bounded-loss cascade with a learned stop + experience-graph
memory + measured cost ledger**. Closest neighbors to distinguish in any writeup:
FrugalGPT/RouteLLM (fixed router, $-only), LYNX (learned stop, no routing/memory),
Energy-Aware Routing to LRMs 2601.00823 (oracle stop, fixed router), MARLIN
2605.13496 (datacenter scheduling, not per-input; cooling from fixed COP). The
defensible core remains Legs 2+3 (verifier-as-governor + energy-indexed
experience graph); re-verify before any external claim.

---

## Conventions (same discipline as Stage A)

- Source: `src/dern_b/*.py` (+ `__init__.py`). Tests: `tests/dern_b/test_*.py`
  (+ `__init__.py`). Imports `from src.dern_b... import ...`.
- Reuse `src/dern/graph.py`, `src/dern/breaker.py`, `src/dern/controller.py`
  unchanged (DRY). Do not modify Stage-A files.
- Real models only. Tests that load the 31B reference are marked
  `@pytest.mark.heavy` and may be skipped in fast runs, but a real cheap-model
  test always runs (no mock).
- All randomness via `np.random.default_rng(seed)`.
- Run DERN-B suite: `pytest tests/dern_b -q`. Full own suite: `pytest tests/ -q`
  (must stay green; never break the 54 existing tests).
- Working dir: `/Volumes/WS4TB/codec-exper/codec_codex_experiments`; `source .venv/bin/activate`.

---

### Task 1: mlx-lm backend wrapper (real model, measured timing/tokens)

**Files:** Create `src/dern_b/__init__.py`, `src/dern_b/mlx_backend.py`; Test `tests/dern_b/__init__.py`, `tests/dern_b/test_mlx_backend.py`

A thin wrapper that loads an mlx model once and runs a chat generation,
returning text + measured (prompt_tokens, gen_tokens, wall_seconds). Active
params per model are read from config (or set explicitly).

Test (real cheap model, no mock):
```python
import pytest
from src.dern_b.mlx_backend import MLXModel, CHEAP_PATH

def test_cheap_model_generates_and_meters():
    m = MLXModel(CHEAP_PATH)
    r = m.generate("Reply with exactly the single word: OK", max_tokens=8)
    assert isinstance(r.text, str) and len(r.text) > 0
    assert r.gen_tokens >= 1 and r.prompt_tokens >= 1
    assert r.wall_seconds > 0.0
    assert r.active_param_seconds > 0.0
```
Implementation: `MLXModel.generate` calls `mlx_lm.generate`, times it, counts
tokens via the tokenizer, computes `active_param_seconds = active_params * wall_seconds`.
`CHEAP_PATH`/`REF_PATH` constants point at the two local model dirs. Commit.

### Task 2: Distinction probe over a prompt

**Files:** Create `src/dern_b/probe_text.py`; Test `tests/dern_b/test_probe_text.py`

A cheap, deterministic, non-semantic key for a prompt: features like token
length bucket, presence of digits/code markers, question-type cues — quantized
to a small int tuple. No model call (must be cheap). Test: deterministic, tuple
of ints, separates short-factual from long-reasoning prompts into different keys.
Commit.

### Task 3: Bounded-loss verifier against the reference model

**Files:** Create `src/dern_b/verifier_b.py`; Test `tests/dern_b/test_verifier_b.py`

`verify_against_reference(cheap_text, ref_text, epsilon, metric="token_f1")` ->
Verdict(passed, delta, proof="bounded"). The reference model's answer is the
authority; the cheap answer passes iff agreement ≥ (1 − ε). PURE w.r.t.
controller (no controller arg — same structural guarantee asserted by test).
Test with synthetic strings (no model needed) for the metric logic, plus one
heavy test using both real models on a trivial prompt. Commit.

### Task 4: Measured cost vector + honesty tags

**Files:** Create `src/dern_b/cost_b.py`; Test `tests/dern_b/test_cost_b.py`

`cost_record(gen_result, posture, overhead)` -> dict with `tokens`,
`wall_seconds`, `active_param_seconds` tagged **`measured`**; `joules` tagged
`unavailable` (or `derived` if a J/param-s constant is supplied, never
`measured`); `_tags` manifest. `net_savings_tokens` / `net_savings_aps`
(active-param-seconds) charge router overhead to the chosen side. Test asserts
measured dims are tagged `measured`, joules is NEVER `measured`. Commit.

### Task 5: powermetrics energy probe (opt-in, sudo-gated, honest)

**Files:** Create `src/dern_b/energy_probe.py`; Test `tests/dern_b/test_energy_probe.py`

`measure_energy(fn)` runs `fn` while sampling `powermetrics` IF a non-interactive
sudo is available; returns (result, joules_or_None, source_tag). When sudo is
unavailable it returns `(result, None, "unavailable")` — it MUST NOT fabricate a
number. Test asserts: with no sudo, source_tag == "unavailable" and joules is
None (the honest path); the measured path is documented for the user to run
manually with `! sudo -v` first. Commit.

### Task 6: DERN-B cascade runtime

**Files:** Create `src/dern_b/runtime_b.py`; Test `tests/dern_b/test_runtime_b.py`

Ties it together, reusing `src/dern/graph.py`, `breaker.py`, `controller.py`:
probe(prompt) -> key -> graph lookup -> controller proposes {cheap, reference}
-> run cheap; if controller proposed cheap, AUDIT against reference at ε (Lane 2);
pass -> serve cheap + record measured savings; fail -> serve reference (never
serve unverified-worse-than-reference) + evict/lock. Reward minted only from the
verdict + measured cost. Breaker forces reference on instability. Heavy test
(both real models) on a tiny prompt set asserts: every served output is either
cheap-and-verified or the reference; ledger has measured token/latency dims; no
joules tagged measured. Commit.

### Task 7: Stage-B driver + measured report

**Files:** Create `src/dern_b/run_stage_b.py`; Test `tests/dern_b/test_run_stage_b.py`

Runs the cascade over a small REAL prompt set (a fixed local list of short
factual + reasoning prompts, committed as data — no network), prints a report:
mean measured token savings, mean latency savings, mean active-param-seconds
savings, accuracy-vs-reference (agreement rate), cheap-acceptance rate, escalation
rate, all net of overhead; honesty manifest (which dims measured). Heavy/manual
run documented. Test asserts the report has measured dims and zero
served-worse-than-reference. Commit.

### Task 8: Full-suite green + Stage-B results note

Run `pytest tests/ -q` (own suite stays green; heavy tests may be deselected with
`-m "not heavy"` for speed but must pass when run). Write
`results/dern_stage_b_results.md` with the measured report and the explicit
honesty manifest (tokens/latency/aps measured; joules opt-in/unavailable;
quality bounded vs reference). State the prior-art posture. Commit + push.

---

## Stage-B acceptance gate

- Every served output is cheap-and-verified-within-ε OR the reference model's
  output (no served-worse-than-reference); asserted in tests.
- On the real prompt set, the cascade achieves measured token + latency +
  active-param-seconds savings > 0 net of overhead at the chosen ε, with the
  cheap-acceptance rate and agreement-vs-reference reported honestly (if savings
  ≤ 0 at a safe ε, that is reported as an honest negative, not tuned away).
- Joules are reported ONLY if the user ran the sudo-gated energy probe; otherwise
  tagged unavailable/derived, never measured.
- The 54 existing tests stay green.

## Deferred (documented, not claimed)
- Lane 1 exact speculative decoding (needs white-box token acceptance).
- True measured joules as a headline (needs the user's sudo run).
- Broadened thermal/fan/DVFS/carbon envelope as MEASURED (thermal/fan readable
  via powermetrics under sudo; carbon/cooling remain simulated on a laptop).
