# DERN — Distinction-governed Envelope Routing Network

**Design document.** 2026-05-31. Status: approved, pre-implementation.
Author dialogue: Wayne A. Satz + Claude (brainstorming skill).
Lineage: applies the *Perceptibility Gap / CodecGuard / CodecBench* methodology
(this repo) and the *CIO-II* escalate-only-on-unresolved-distinction architecture
to a pragmatic, energy-governing AI inference application.

---

## 0. Origin and intent

The task: turn the methodology and theory into **pragmatic, breakthrough,
non-human-codec applications** that move toward a superintelligence framework —
not academic proofs. Through iterative refinement the target converged from
"AI-to-AI communication" to something sharper and buildable:

**An always-learning controller that governs how much compute and what
operational posture each inference request gets, driven by a non-human
distinction-state signal, to achieve identical-or-better output at lower
energy — across a broad cost envelope (compute + thermal/fan + DVFS +
race-to-idle + carbon).**

The non-human-codec property is scoped honestly: with current tools we govern
*how much* token-machinery and *how much machine* runs; we do not escape the
token stack. What is genuinely extra-human and real:
- the **routing decision** uses a non-semantic distinction-state signal, not a
  human-trained gate or token probabilities;
- the **experiential memory** stores distinction-state regions keyed to proven
  cost, not human text.

"Post-token substrate" is the research frontier the architecture is built to
grow into (the synthetic-agent lane keeps that path real), not a first-build claim.

### Theory connection (already validated in this repo)
- **throughput ≠ distinctions** (proof ladder S1–S3) → spend compute only where
  it buys distinctions; raw compute without distinction-gain is waste.
- **CodecGuard disagreement → error** (S8, survived even on the chaotic Hénon
  family) → when independent experts agree, more compute cannot change the
  answer; stop. This becomes the runtime stop-condition.

### CIO-II's role
Design ancestor, **not** proof of transfer. It contributes the architecture
(tiered escalate-only-on-unresolved-distinction, fail-closed, trust circuit
breaker, per-decision ledger, no-mock real-data testing) and the governance
discipline. It does not prove the pattern transfers to LLMs — Stage B measures that.

---

## 1. Concept and honest claims

**What it is:** an energy-governing inference router whose heart is an
**online-learning controller** that decides, per input: which compute config
(speculative draft length / experts / depth / model tier), which operational
posture (clock / fan / idle / defer), and **when to stop** — learning both its
choices and its stop-signal at runtime.

**What is novel (and what is not — stated plainly):**
- *Not novel:* the levers. Speculative decoding, mixture-of-depths, early-exit,
  cascade inference, dynamic MoE pruning, RL DVFS, RL cooling, carbon-aware
  scheduling all exist and are optimized. DERN invents no inference primitive.
- *The contribution* (confirmed absent in prior art, May 2026): the
  **combination** —
  1. online co-learning of **tier AND stop-signal** in one continually-updated
     controller (prior online routers learn tier only, with a single decision and
     no learned stop; stop-signals elsewhere are fixed trained gates);
  2. governance by an **unmodifiable verifier** running **dual lanes** (exact-
     lossless + bounded-loss with sampled full-compute spot-checks) — the
     spot-check primitive exists (DiFR) but for fraud/trust, never as a referee
     authorizing+auditing a bounded-loss fast lane;
  3. an **energy-indexed experience graph** as memory (semantic caches reuse
     *answers*; trainable graph memory reuses agent *strategies*; nobody keys a
     memory by distinction-region → cheapest proven-safe config for routing);
  4. a **broad cost vector** (compute energy + thermal/fan + DVFS +
     race/pace-to-idle + carbon) under **one** learning policy — these live in two
     non-communicating research worlds today (inference-efficiency optimizes
     FLOPs and ignores fans/clocks/carbon; systems-energy treats the workload as
     fixed and never changes how much model compute an input gets).

**Two honesty flags (carry into every external claim):**
- **Closest prior art: arXiv 2601.08113 (Jan 2026), "Coordinated Cooling and
  Compute Management for AI Datacenters."** Must be cited and distinguished on
  three axes: (i) it is a *fixed* LP/MILP/MPC stack with only learned
  *predictors*, not an always-learning controller; (ii) its compute knob is
  *scheduling/clock selection*, not *per-input inference routing*; (iii) no
  carbon, no race-to-idle.
- **"RL for energy control" is not itself novel** (DeepMind ran a real cooling
  plant on RL). Novelty is the **scope of the action space under one learning
  policy** (routing ⊕ stop-signal ⊕ DVFS ⊕ fan ⊕ idle ⊕ carbon), refereed by an
  unmodifiable verifier.

**Two lanes (the honest "lossless" framing):**
- **Lane 1 — exact-lossless:** speculative-style verify; the target model checks
  every kept token via rejection sampling → output is provably identical to full
  compute. Energy drops only when cheap drafts are accepted. Strict-safe mode;
  this is where literal zero-loss holds.
- **Lane 2 — bounded-loss:** early-exit, expert-pruning, experiential replay.
  Output may differ; a measured ≤ε quality guarantee is calibrated against full
  compute on a sampled audit stream, with a tunable risk knob. More savings,
  honestly labeled bounded — never lossless.

**Headline outcome (human-undeniable, non-semantic):**
- Lane 1 → *identical output, measured energy down* (net of router cost).
- Lane 2 → *≤ε measured quality loss, larger measured energy down* (net of router cost).

**Validation, sequenced (no-mock discipline):**
- **Stage A** — real, local, on the repo's synthetic dynamical agents (ground-
  truth distinctions). Proves the *logic, stop-signal safety under the verifier,
  graph earn/evict, breaker, honest accounting*. **No energy claim.**
- **Stage B** — one open-weight model, real task, real hardware. Lane 1 first;
  **measured** energy (NVML/RAPL/powermetrics) + fan/thermal telemetry, net of
  router cost. The real headline; cooling-plant/carbon dimensions labeled simulated.

---

## 2. Architecture

```
 INPUT (task / request / trajectory)
   │
   ▼
 (1) DISTINCTION PROBE — cheap read of "what's unresolved"; emits a non-semantic
     distinction-state KEY (no human-text form); cost metered as overhead_J
   │
   ▼
 (2) EXPERIENCE GRAPH (non-human codec / memory)
     key = distinction-state region
     edge → (config, PROVEN cost-vector distribution, proof-of-safety, trust)
   │
   ▼
 (3) CONTROLLER (always-learning) — proposes compute config + operational posture
     + STOP proposal; learns choices AND stop-signal online; bounded by trust-breaker
   │
   ▼
 (4) VERIFIER (UNMODIFIABLE — the safety spine; outside the controller's reach)
     Lane 1 exact:   speculative verify → byte-identical or reject→full
     Lane 2 bounded: with prob p_audit run FULL alongside → δ≤ε PASS, else FAIL+evict
   │
   ▼
 (5) ENVELOPE ACTUATOR + LEDGER
     applies compute config AND posture (clock/fan/idle/defer)
     meters BROAD cost vector incl. the controller's OWN cost
     writes per-decision ledger → feeds graph + controller
```

**Cost vector on every edge / record:**
`v = (compute_J, thermal_slope, dvfs_state, idle_strategy, carbon_window,
controller_overhead_J)`, each dimension tagged `measured | telemetry-derived |
simulated`. "Cheapest proven-safe config" is a multi-objective lookup under the
current envelope constraints. The fan insight lives here: an edge learns "this
distinction-region runs short & cool → low-fan/low-clock posture proven safe."

### Stage-A component map (real, local, no-mock, no energy claim)
| Component | Stage-A concrete form | Validates |
|---|---|---|
| (1) Probe | cheap trajectory statistic → distinction key (reuse `proof_ladder` channels) | key separates easy/hard |
| (2) Graph | in-memory graph: region → (config, proven recovery, **simulated** cost-vector) | replay + eviction |
| (3) Controller | online bandit/RL over {few↔many experts, shallow↔deep, stop-now} + learned stop threshold | converges, stable under breaker |
| (4) Verifier | Lane 1: exact full-compute compare on synthetic targets; Lane 2: sampled full-compute spot-check at ε | catches over-eager stops; eviction fires |
| (5) Actuator+ledger | applies config; **simulated** cost-vector incl. controller overhead; per-decision ledger | net-positive accounting; auditability |

---

## 3. Data flow and the learning loop

One request: PROBE→key → GRAPH lookup near key (trusted edge ⇒ candidate config;
else controller cold proposal) → CONTROLLER proposes config+posture+stop →
VERIFIER (Lane 1 exact accept/reject; Lane 2 sampled δ vs ε) → ACTUATOR applies
*verified* outcome + meters cost vector → LEDGER record → GRAPH update +
CONTROLLER update.

**Core invariant:** the controller's reward and the graph's trust are computed
**only** from the verifier's verdict and the metered cost — never from the
controller's own stop proposal. The optimized signal is always grounded in
something the optimizer cannot write.

**Graph write (earn trust):** edge created/refreshed only after PASS.
Lane-1 PASS → full-strength trust, `proof="exact"`. Lane-2 PASS (δ≤ε) →
`p_audit`-discounted trust accumulating over repeated proven hits (Beta-style
success count); cost stored as a *distribution* (remembers variance).

**Graph evict (revoke trust):** Lane-2 FAIL on a trusted region → immediate
eviction + region locked to "audit-forced" until re-earned; staleness decay so
regions keep re-proving; constraint shift → partial eviction of posture
dimensions only (compute-safety and posture-safety tracked separately).

**Controller update (always-learning, contained):** reward =
`−normalized_cost_vector(v)` on PASS, **hard penalty on Lane-2 FAIL**, Lane-1
reject costs reward ∝ wasted `overhead_J`. Stop-signal learning is safe-by-
construction in Lane 1 (exact verify) and audit-bounded in Lane 2. Exploration
is gated by the verifier (worst case = wasted joules, never quality). "Better"
is always measured against a **frozen full-compute reference** (reproducibility).

**Trust-breaker (stability spine, from CIO-II):** triggers on Lane-2 FAIL-rate
over threshold, controller oscillation (action-entropy spike), cost worsening vs
reference, or eviction spike → adaptation **pauses** (controller frozen at
last-known-good) → routing falls back to full compute + full audit → resumes
after cool-down with recalibrated thresholds. Converts online-learning
instability into self-detect-and-revert-to-baseline.

---

## 4. Error handling and failure modes

**Governing principle:** *every failure path resolves to full, verified,
known-good compute — never to a degraded guess.* The worst case is always the
baseline, never worse.

**Verifier:** Lane-1 unavailable → fall through to full (no unverified draft
emitted). Lane-2 spot-check errors → treat as FAIL (fail-closed), evict.
Verifier too slow → output stands but **trust write withheld** (never mint
unconfirmed trust). Controller→verifier write → structurally impossible (module
boundary + asserted test).

**Probe:** mis-key → caught downstream (Lane-1 reject / Lane-2 FAIL); costs
joules, never quality. Key collision → edge variance widens, trust suppressed,
evicted on repeated FAIL. Probe too expensive → priced out by net-positive rule.

**Graph:** cold start → conservative mode (full compute + high audit), launches
*as* baseline, only ever gets cheaper. Poisoning → edges minted only after PASS;
later FAIL → eviction + lockout; burst → breaker trips. Forgetting/bloat →
trust decay + bounded LRU-by-trust; proven Lane-1 edges sticky. Stale envelope →
partial eviction of posture dimensions.

**Controller:** divergence/oscillation → breaker. Reward hacking → cannot
succeed (reward only from verifier verdicts; Lane-2 FAIL penalized; exploited
route evicted). Slow convergence → not a safety failure; reported as "no net
savings on this workload."

**Cross-cutting (prior-art-flagged):**
- *Telemetry gaps/sampling distortion (Stage B):* energy measured over windows
  beating ~15ms NVML sampling; PMT-style unified timestamps; any unmeasurable
  dimension tagged `simulated`, **never** reported as measured. No-mock applied
  to the metric.
- *Unmeasurable dimensions on a laptop:* cooling-plant + carbon tagged
  `simulated`, barred from headline claims; may drive research exploration only.
- *Envelope-constraint conflict:* hard ordering — safety/correctness > latency
  SLA > thermal limit > energy/carbon. Controller optimizes only within the
  feasible region; conflicts resolve deterministically, logged.
- *Adversarial breaker-trip (DoS):* degrades to baseline cost = what a non-DERN
  system pays; attack ceiling is benign.

**Safety contract:** every error path terminates at full, verified, known-good
compute. Savings are an earned optimization on top of a baseline the system can
always fall back to.

---

## 5. Testing and acceptance gates

No mocks, real data, failures as honest negatives, no threshold tuned to pass.

### Stage-A suite (logic correctness; simulated cost)
**Functional:** `test_probe_separates_difficulty`, `test_graph_replay_lowers_cost`,
`test_controller_converges`, `test_lane1_exact_identity`, `test_lane2_epsilon_bound`.
**Safety invariants:** `test_controller_cannot_write_verifier`,
`test_early_stop_caught_by_verifier`, `test_reward_only_from_verdict`,
`test_frozen_reference_fixed`.
**Failure-injection:** `test_verifier_down_falls_to_full`,
`test_drift_evicts_and_locks`, `test_poison_bounded_blast_radius`,
`test_oscillation_trips_breaker`, `test_unmeasurable_dim_tagged_simulated`,
`test_constraint_ordering_deterministic`.

**Stage-A acceptance gate (blocks Stage B):** 100% of safety-invariant +
failure-injection tests pass (non-negotiable); controller converges below
baseline on ≥1 synthetic workload with zero Lane-1 exactness violations and zero
undetected Lane-2 ε-breaches; ledger accounting net-honest (overhead included).
Sub-100% on the safety set → action plan or explicit human waiver; never silent.

### Stage-B measurement protocol (real model, measured energy)
Per-dimension status on commodity HW: compute energy **measured**
(NVML/RAPL/powermetrics, PMT-style unified timestamps, windowed); wall-clock
**measured**; fan/thermal **telemetry-derived**; DVFS **measured**; cooling-plant
**simulated**; carbon **simulated/external-API**. Every record tags its dimensions.

```
net_savings   = baseline_cost(full)
              − (DERN_output_cost + controller_overhead + probe_overhead + verifier_overhead)
quality_delta = accuracy(DERN) − accuracy(baseline)    # same real task set
```
Claims stated only on **measured** dimensions, net of DERN's own cost.

**Stage-B acceptance gates:**
- *Lane 1 (strict win):* outputs byte-identical to full compute AND measured net
  compute-energy reduction > 0 after router cost. If it fails, no lossless claim
  ships — report the negative.
- *Lane 2:* measured quality_delta ≥ −ε AND larger net savings than Lane 1 AND
  live Lane-2 FAIL rate within bound. Reported as "≤ε measured loss, X% more
  savings," never lossless.
- *Broadened-envelope:* on a workload experience predicts cool, the learned
  low-fan/low-clock posture yields measured fan/DVFS energy reduction with no SLA
  or thermal-limit breach (measured/telemetry dims only; carbon/cooling simulated
  + labeled).
- *Prior-art differentiation:* re-run the prior-art check before any external
  claim; explicitly distinguish arXiv 2601.08113 on the three axes. If the gap
  has closed, report it and narrow the claim.

### Proof artifacts (CIO-II-style)
Per-decision ledger; savings report (net measured savings, quality delta,
router-overhead line, Lane-1 vs Lane-2 split); safety report (Lane-1 exactness
violations [target 0], Lane-2 FAIL rate vs bound, breaker trips, evictions);
honesty manifest (measured vs simulated dimensions on this hardware).

### Pre-committed honest failure outcomes (no goalpost-moving)
- Lane 1 no net savings after router cost → exact lane doesn't beat baseline on
  this model/task; report, don't tune.
- Controller won't converge → "no learned advantage on this workload," not hidden.
- Bounded lane can't hold ε → Lane 2 disabled for that workload; only exact lane ships.
- Whole thing nets negative → publishable negative result about online-learned
  envelope routing on current tools.

---

## 6. Prior-art summary (May 2026 check)

Nearest neighbors per ingredient — none combine them, and the two defining
pieces are individually absent:
- Online tier routers under $ budget: PILOT (EMNLP'25), ParetoBandit (2026),
  GreenServ (energy-aware MAB) — tier only, single decision, no learned stop, no
  verifier, no energy-indexed memory.
- Exact-lossless lane: speculative decoding family (EAGLE-3, Medusa, HSD,
  lookahead) — exact via rejection sampling, but verifier is internal, fixed.
- Spot-check primitive: DiFR / Token-DiFR — for fraud/trust, not governing a
  bounded-loss router.
- Dynamic depth/experts: Mixture-of-Depths, Mixture-of-Recursions, DiEP — fixed
  trained gates, bounded-loss, FLOP-only.
- Experience memory: semantic caches (reuse answers), trainable graph memory
  (reuse agent strategies) — neither energy-indexed for routing.
- Systems energy: RL DVFS (MetaDVFS, FiDRL), DeepMind/Meta RL cooling,
  carbon-aware scheduling (GREEN NSDI'25, CarbonFlex) — workload treated as
  fixed; never changes per-input compute.
- Closest unified threat: **arXiv 2601.08113** — fixed LP/MILP/MPC + learned
  predictors; compute knob = scheduling/clock, not per-input routing; no carbon,
  no race-to-idle.

**Defensible novelty:** one always-learning controller whose compute action is
per-input inference routing (tier + learned stop) jointly with a broad
thermal/DVFS/carbon envelope, refereed by an unmodifiable dual-lane verifier,
backed by an energy-indexed experience graph. Re-verify before any filing/publication.

### 6.1 Prior-art re-check delta (2026-05-31)

A fresh re-check found work newer than the original survey that **narrows** the
claim. The combination of all four legs still has no single prior match, but:

- **RETIRED FRAMING:** "no learned joint compute+cooling controller exists" is now
  **false**. **MARLIN (arXiv 2605.13496, 2026-05-13)** is a multi-agent RL
  controller for sustainable LLM inference (TTFT + carbon + water + energy). It
  must be cited and distinguished: it controls *geo-distributed request
  scheduling* (which datacenter), not *per-input* tier/routing; its cooling is
  *derived from a fixed COP*, not co-learned; it has no stop signal, no verifier,
  no experience graph; it updates per 15-min epoch.
- **Leg 1 (co-learn tier + stop) — holds, narrowed.** **LYNX (arXiv 2512.05325)**
  learns an online confidence-controlled *stop* (early-exit); other 2026 work
  learns *routing*. None co-learns *both* in one online controller. Must argue
  explicitly against LYNX + a learned router.
- **Leg 2 (verifier-as-governor) — holds, strongest.** No 2026 work uses a
  verifier as the safety governor of a routing/stop policy; speculative decoding
  remains framed as acceleration, DiFR as fraud detection.
- **Leg 3 (energy-indexed experience graph) — holds, strongest.** Only KV-cache /
  expert-activation caching found; no memory of *proven-safe energy configs keyed
  by input region*.
- **Leg 4 (broad envelope under one learned policy) — holds, narrowed by MARLIN.**
  Differentiators narrow to: per-input granularity, race-to-idle, and cooling/DVFS
  *co-learned in the same online policy*.

**Mandatory new citations:** MARLIN (2605.13496), LYNX (2512.05325),
Energy-Aware Routing to LRMs (2601.00823); neighbors: R2-Reasoner (WWW'26),
Cost-Aware Model Orchestration (2512.01099), ZIP-RC (2512.01457).

**Re-anchored claim:** the inseparable combination of **per-input routing +
co-learned stop + unmodifiable dual-lane verifier + energy-indexed experience
graph**, with Legs 2 and 3 as the wide-open core differentiators. Re-verify
again immediately before any external/measured-energy claim.

---

## 7. Next step

Stage A is the first build. The implementation plan (via the writing-plans
skill) decomposes Stage A into ordered, independently-verifiable tasks against
the synthetic-agent testbed already in `codec_codex_experiments/`, with the
Stage-A acceptance gate as the exit criterion before any real-model / energy work.
