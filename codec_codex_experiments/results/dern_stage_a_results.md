# DERN Stage-A Results

Stage A of the DERN design (`docs/plans/2026-05-31-dern-ai-to-ai-energy-routing-design.md`),
implemented per `docs/plans/2026-05-31-dern-stage-a-implementation.md`.

**Scope statement (read first):** Stage A validates the runtime **logic only**,
on the repo's synthetic dynamical agents, with a **simulated** cost vector.
**No energy claim is made here.** Stage B (real open-weight model, Lane-1
speculative decoding, *measured* NVML/RAPL/powermetrics energy net of router
cost) is gated on this passing and is a separate build.

**Run:** `python -m src.dern.run_stage_a` (120 requests/family).
**Tests:** `pytest tests/ -q` -> 54 passed (17 pre-existing + 37 new DERN tests).
The pre-existing suite is byte-for-byte intact; DERN is self-contained in `src/dern/`.

---

## Stage-A acceptance gate — met

| Criterion | Result |
|---|---|
| Safety-invariant tests pass (100%) | yes (`tests/dern/test_safety_invariants.py`, 4/4) |
| Failure-injection tests pass (100%) | yes (`tests/dern/test_failure_injection.py`, 4/4) |
| Controller converges below baseline on >=1 workload | yes (all three families, positive net savings) |
| Zero unverified serves | yes (all families) |
| Zero Lane-1 exactness violations | yes (all families) |
| All cost dimensions tagged `simulated` (no phantom "measured") | yes |

## Report (simulated cost; no energy claim)

| family | requests | mean_net_savings | unverified_serves | lane1_exactness_violations |
|---|---|---|---|---|
| linear_oscillator | 120 | 2.95 | 0 | 0 |
| nonlinear_oscillator | 120 | 2.95 | 0 | 0 |
| henon_map | 120 | 8.55 | 0 | 0 |

**Reading (honest):**
- Net savings are **simulated cost units**, not joules. They show the *routing
  logic* finds and reuses cheaper-but-verified configurations — nothing about
  real energy.
- The Hénon (chaotic) family shows the *largest* simulated savings, which is
  the correct and informative behavior: its distinctions are largely
  unrecoverable (sensitive dependence), so under a loose epsilon the cheap
  config is verified "no worse than full" and the controller routes to it —
  when expensive compute buys no additional distinctions, spending it is waste.
  This is the `throughput != distinctions` principle (proof ladder S1/S6)
  operating as a runtime control law.
- **The safety spine held absolutely on every family:** nothing unverified was
  ever served, and no config claiming exact-lossless ever deviated from full
  compute. The aggressive online controller never produced a worse-than-baseline
  output, by construction.

## What this does and does not show

**Does:** the DERN runtime logic is executable and internally coherent — the
unmodifiable dual-lane verifier gates every serve, the experience graph earns
and evicts trust on proof, the trust breaker bounds instability, and the
always-learning controller's reward is minted only from the verifier's verdict
(reward-hacking is structurally blocked). Every error path collapses to full,
verified compute.

**Does not:** make any energy claim, run on a real model, or measure joules.
Those are Stage B, explicitly gated on this. The "beyond-token / non-human
codec" property here is narrow and real: the routing key and the experience
graph are non-semantic; the levers remain known techniques.

## Prior-art honesty (carried from the design)

Defensible novelty is the *combination* (confirmed absent in the May-2026
prior-art pass): one always-learning controller co-learning tier + stop-signal,
governed by an unmodifiable dual-lane verifier, backed by an energy-indexed
experience graph, optimizing a broad cost vector. The closest unified prior work
(arXiv 2601.08113, Jan 2026) is a *fixed* LP/MILP/MPC stack whose compute knob is
scheduling/clock, not per-input routing, and omits carbon and race-to-idle.
Re-verify prior art before any external claim, especially before Stage B's
measured-energy results are published.
