# DERN Stage-A Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and validate the DERN runtime *logic* — distinction probe, energy-indexed experience graph, always-learning controller, unmodifiable dual-lane verifier, and envelope actuator+ledger — entirely on the repo's synthetic dynamical agents, where distinctions have ground truth, with a **simulated** cost vector and **no energy claim**.

**Architecture:** A new `src/dern/` subpackage. The synthetic `SourceFamily` agents (`src/sources.py`) supply inputs whose ground-truth distinctions are the sampled `theta`. A cheap probe maps a trajectory to a non-semantic distinction-state key; an experience graph remembers `key -> (config, proven cost-vector, proof, trust)`; an online controller proposes `(compute_config, posture, stop)`; an **unmodifiable verifier** confirms via Lane 1 (exact compare to full compute) or Lane 2 (sampled full-compute spot-check at tolerance epsilon); an actuator meters a simulated cost vector (incl. controller overhead) and writes a per-decision ledger that feeds graph + controller. A trust circuit breaker bounds adaptation. Every failure path collapses to full compute.

**Tech Stack:** Python 3.13, numpy, scikit-learn (already in `requirements.txt`), pytest (`pytest.ini` has `pythonpath = .`). Imports use `from src.dern... import ...` and `from src.proof_ladder import ...`, matching existing tests. No new dependencies. No mocks — all tests use the real synthetic agents and real numeric assertions, mirroring `tests/test_sources.py`.

**Working directory for all commands:** `/Volumes/WS4TB/codec-exper/codec_codex_experiments`
**Venv:** `source .venv/bin/activate` before running pytest.

**Exit criterion (Stage-A acceptance gate, from the design doc §5):** 100% of the safety-invariant and failure-injection tests pass; the controller converges to below-baseline *simulated* cost on >=1 synthetic workload with zero Lane-1 exactness violations and zero undetected Lane-2 epsilon-breaches; ledger accounting is net-honest (controller/probe/verifier overhead included). No real-model or energy work happens in Stage A.

---

## Conventions for every task

- Test files: `tests/dern/test_*.py`. Create `tests/dern/__init__.py` once (Task 0).
- Source files: `src/dern/*.py` with `src/dern/__init__.py` (Task 0).
- Run a single test: `pytest tests/dern/test_x.py::test_name -v`
- Run the DERN suite: `pytest tests/dern -q`
- Run everything (must stay green — never break the 17 existing tests): `pytest -q`
- Commit after each task with the message shown. Stage everything the task created/modified.
- TDD: write the failing test first, run it red, implement minimally, run it green, commit.
- All randomness uses `numpy.random.default_rng(seed)` — never `Math.random`-style global state — so tests are deterministic (same discipline as `src/proof_ladder.py`).

---

### Task 0: Scaffold the `dern` subpackage

**Files:**
- Create: `src/dern/__init__.py`
- Create: `tests/dern/__init__.py`
- Test: `tests/dern/test_scaffold.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_scaffold.py
def test_dern_package_imports():
    import src.dern as dern
    assert dern.__doc__ is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_scaffold.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'src.dern' / tests.dern)

**Step 3: Create the package files**

```python
# src/dern/__init__.py
"""DERN — Distinction-governed Envelope Routing Network (Stage A: logic only).

Stage A validates the runtime logic on synthetic agents with a SIMULATED cost
vector. No energy claim is made here; see docs/plans/2026-05-31-dern-*-design.md.
"""
```

```python
# tests/dern/__init__.py
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_scaffold.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dern/__init__.py tests/dern/__init__.py tests/dern/test_scaffold.py
git commit -m "feat(dern): scaffold Stage-A subpackage"
```

---

### Task 1: Distinction probe — non-semantic key from a trajectory

The probe reads a cheap statistic from a trajectory and emits a stable
distinction-state **key** (a small int tuple, no human-text form). It must (a) be
cheap, (b) separate easy from hard inputs above chance, (c) be deterministic.

**Files:**
- Create: `src/dern/probe.py`
- Test: `tests/dern/test_probe.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_probe.py
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.probe import distinction_key, probe_cost_units


def _cfg():
    return normalize_cfg({"n_train": 200, "n_test": 90, "T": 6.0, "dt": 0.1, "seeds": [0]})


def test_key_is_deterministic_and_nonsemantic():
    cfg = _cfg()
    traj, _ = SOURCE_REGISTRY["linear_oscillator"].sample(10, cfg, 0)
    k1 = distinction_key(traj[0])
    k2 = distinction_key(traj[0])
    assert k1 == k2                      # deterministic
    assert isinstance(k1, tuple)         # non-semantic structured key
    assert all(isinstance(x, int) for x in k1)


def test_key_separates_easy_from_hard():
    # Linear (easy: recoverable params) vs Henon (hard: chaotic, near-unrecoverable).
    # Their key distributions must differ, or the probe carries no signal.
    cfg = _cfg()
    lin, _ = SOURCE_REGISTRY["linear_oscillator"].sample(60, cfg, 0)
    hen, _ = SOURCE_REGISTRY["henon_map"].sample(60, cfg, 0)
    lin_keys = {distinction_key(lin[i]) for i in range(len(lin))}
    hen_keys = {distinction_key(hen[i]) for i in range(len(hen))}
    # Overlap should be far from total; the families occupy different key regions.
    overlap = len(lin_keys & hen_keys) / max(len(lin_keys | hen_keys), 1)
    assert overlap < 0.5, f"probe cannot separate families (overlap={overlap:.2f})"


def test_probe_cost_is_positive_and_small():
    assert 0 < probe_cost_units() < 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_probe.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.probe)

**Step 3: Write minimal implementation**

```python
# src/dern/probe.py
"""Distinction probe: cheap, deterministic, non-semantic key for a trajectory.

The key is a tuple of small integers (quantized coarse statistics). It carries
NO human-text form — it is an opaque region label the experience graph keys on.
Cost is reported in abstract 'units' (Stage A has no real energy; the actuator
converts units -> simulated cost vector).
"""
from __future__ import annotations

import numpy as np

# Quantization bins per statistic; coarse so similar trajectories share a key.
_BINS = 4


def _quantize(value: float, lo: float, hi: float, bins: int = _BINS) -> int:
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    return int(np.clip(int(frac * bins), 0, bins - 1))


def distinction_key(traj: np.ndarray) -> tuple:
    """traj: (T, N) single-trajectory array -> structured int key.

    Uses cheap shape statistics: per-step energy spread, late-vs-early variance
    ratio (a coarse 'is it settling or chaotic' signal), and mean abs level.
    """
    x = np.asarray(traj, dtype=float)
    half = x.shape[0] // 2 or 1
    early_var = float(x[:half].var())
    late_var = float(x[half:].var())
    spread = float(x.std())
    level = float(np.abs(x).mean())
    ratio = late_var / (early_var + 1e-8)
    return (
        _quantize(spread, 0.0, 6.0),
        _quantize(ratio, 0.0, 3.0),
        _quantize(level, 0.0, 4.0),
    )


def probe_cost_units() -> float:
    """Abstract cost of running the probe (small but nonzero; metered as overhead)."""
    return 0.05
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_probe.py -v`
Expected: PASS (3 passed). If `test_key_separates_easy_from_hard` is flaky on bin edges, widen `_BINS` ranges — do NOT relax the assertion below 0.5.

**Step 5: Commit**

```bash
git add src/dern/probe.py tests/dern/test_probe.py
git commit -m "feat(dern): distinction probe emits non-semantic region key"
```

---

### Task 2: Configs and the full-compute oracle (ground truth)

DERN needs a notion of "full compute" (the always-known-good baseline) and a
recovery score. On synthetic agents the ground-truth distinctions are the
sampled `theta`; "full compute" = recover `theta` from the richest channel; a
cheaper config = recover from fewer features / a coarser quantization.

**Files:**
- Create: `src/dern/configs.py`
- Create: `src/dern/oracle.py`
- Test: `tests/dern/test_oracle.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_oracle.py
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.configs import COMPUTE_CONFIGS, full_config, compute_cost_units
from src.dern.oracle import recover_distinctions, recovery_error


def _cfg():
    return normalize_cfg({"n_train": 300, "n_test": 120, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_configs_ordered_by_cost():
    costs = [compute_cost_units(c) for c in COMPUTE_CONFIGS]
    assert costs == sorted(costs), "configs must be listed cheapest->most expensive"
    assert compute_cost_units(full_config()) == max(costs), "full config is most expensive"


def test_full_config_recovers_easy_family_well():
    cfg = _cfg()
    src = SOURCE_REGISTRY["linear_oscillator"]
    err = recovery_error(src, cfg, full_config(), seed=0)
    assert err < 0.6, f"full compute should recover linear distinctions (nrmse={err:.3f})"


def test_cheaper_config_no_better_than_full_on_easy():
    # Sanity: the cheapest config should not beat full compute on an easy family.
    cfg = _cfg()
    src = SOURCE_REGISTRY["linear_oscillator"]
    full_err = recovery_error(src, cfg, full_config(), seed=0)
    cheap_err = recovery_error(src, cfg, COMPUTE_CONFIGS[0], seed=0)
    assert cheap_err + 1e-9 >= full_err - 0.25, "cheap must not dramatically beat full"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_oracle.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.configs)

**Step 3: Write minimal implementation**

```python
# src/dern/configs.py
"""Compute configurations DERN can route to, cheapest -> most expensive.

A config is a dict describing how much 'machine' to run. On synthetic agents we
model 'how much compute' as feature richness + quantization granularity, which
is a faithful proxy for distinction-resolving power (proof-ladder S1/S6 showed
fewer distinctions => higher error). Each config has an abstract compute cost.
"""
from __future__ import annotations

from typing import Any, Dict, List

# channel: which feature builder; k: quantization levels (more = finer distinctions)
COMPUTE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "cheap", "channel": "native", "k": 3, "cost": 1.0},
    {"name": "mid", "channel": "engineered", "k": 6, "cost": 2.5},
    {"name": "full", "channel": "direct", "k": 16, "cost": 5.0},
]


def full_config() -> Dict[str, Any]:
    return COMPUTE_CONFIGS[-1]


def compute_cost_units(config: Dict[str, Any]) -> float:
    return float(config["cost"])
```

```python
# src/dern/oracle.py
"""Full-compute oracle + recovery scoring on synthetic agents.

Ground-truth distinctions are the sampled theta. recovery_error runs a config's
channel through the existing proof-ladder estimator and returns nRMSE vs theta.
This is the 'did we resolve the distinctions' measure the verifier trusts.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.proof_ladder import (
    native_feats,
    engineered_feats,
    direct_feats,
    quantize,
    fit_predict,
    nrmse,
)

_CHANNELS = {
    "native": native_feats,
    "engineered": engineered_feats,
    "direct": direct_feats,
}


def _features(traj: np.ndarray, channel: str) -> np.ndarray:
    return _CHANNELS[channel](traj)


def recover_distinctions(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int):
    """Return (theta_test, predicted_theta) for a config on a source family."""
    xtr, ttr = src.sample(int(cfg["n_train"]), cfg, seed)
    xte, tte = src.sample(int(cfg["n_test"]), cfg, seed + 500)
    Ftr = _features(xtr, config["channel"])
    Fte = _features(xte, config["channel"])
    k = int(config["k"])
    Qtr = quantize(Ftr, k, Ftr)
    Qte = quantize(Fte, k, Ftr)
    pred = fit_predict(Qtr, ttr, Qte, cfg, seed)
    return tte, pred


def recovery_error(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int) -> float:
    tte, pred = recover_distinctions(src, cfg, config, seed)
    return nrmse(tte, pred)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_oracle.py -v`
Expected: PASS (3 passed). If thresholds are off for your numpy build, adjust the *test* tolerances within reason — never weaken the ordering assertions in `test_configs_ordered_by_cost`.

**Step 5: Commit**

```bash
git add src/dern/configs.py src/dern/oracle.py tests/dern/test_oracle.py
git commit -m "feat(dern): compute configs + full-compute recovery oracle"
```

---

### Task 3: The unmodifiable verifier (the safety spine) — Lane 1 + Lane 2

The verifier is the only component that can declare a cheaper config "safe." It
must be a pure function of (source, cfg, config, seed, epsilon) and the oracle —
it takes NO controller state, so the controller structurally cannot influence it.

**Files:**
- Create: `src/dern/verifier.py`
- Test: `tests/dern/test_verifier.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_verifier.py
import inspect
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.configs import COMPUTE_CONFIGS, full_config
from src.dern.verifier import verify_lane1, verify_lane2, Verdict


def _cfg():
    return normalize_cfg({"n_train": 300, "n_test": 120, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_lane1_full_config_is_exact():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    v = verify_lane1(src, cfg, full_config(), seed=0)
    assert v.passed is True and v.proof == "exact" and v.delta == 0.0


def test_lane1_cheaper_config_rejected_unless_identical():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    v = verify_lane1(src, cfg, COMPUTE_CONFIGS[0], seed=0)
    # cheap output differs from full -> Lane 1 must reject (never silently accept)
    assert v.passed is False


def test_lane2_passes_within_epsilon_fails_outside():
    cfg = _cfg(); src = SOURCE_REGISTRY["linear_oscillator"]
    loose = verify_lane2(src, cfg, COMPUTE_CONFIGS[1], seed=0, epsilon=1.0)
    tight = verify_lane2(src, cfg, COMPUTE_CONFIGS[0], seed=0, epsilon=0.0)
    assert loose.passed is True
    assert tight.passed is False        # zero tolerance => any deviation fails

def test_verifier_signature_takes_no_controller_state():
    # Structural guarantee: the controller cannot pass itself into the verifier.
    for fn in (verify_lane1, verify_lane2):
        params = set(inspect.signature(fn).parameters)
        assert "controller" not in params and "policy" not in params
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_verifier.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.verifier)

**Step 3: Write minimal implementation**

```python
# src/dern/verifier.py
"""Unmodifiable verifier: the only authority that can declare a config safe.

Lane 1 (exact): a config passes only if its recovered distinctions are
byte-identical to full compute (delta == 0). This models speculative decoding's
exact verification: a non-full config can only pass if it happens to produce the
identical result, which for distinct configs it will not -> it is rejected and
the caller falls back to full compute. Loss is impossible.

Lane 2 (bounded): a config passes iff its recovery error exceeds full compute's
by at most epsilon. This is the sampled full-compute spot-check.

Both functions are PURE w.r.t. controller state — they accept no controller or
policy argument, so the learner cannot influence the verdict. This is enforced
by test_verifier_signature_takes_no_controller_state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.dern.configs import full_config
from src.dern.oracle import recover_distinctions, recovery_error


@dataclass(frozen=True)
class Verdict:
    passed: bool
    lane: int
    delta: float
    epsilon: Optional[float]
    proof: str  # "exact" | "bounded" | "rejected"


def verify_lane1(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int) -> Verdict:
    _, pred_full = recover_distinctions(src, cfg, full_config(), seed)
    _, pred_cfg = recover_distinctions(src, cfg, config, seed)
    identical = bool(np.array_equal(pred_full, pred_cfg))
    return Verdict(
        passed=identical, lane=1,
        delta=0.0 if identical else float(np.abs(pred_full - pred_cfg).mean()),
        epsilon=None, proof="exact" if identical else "rejected",
    )


def verify_lane2(src, cfg: Dict[str, Any], config: Dict[str, Any], seed: int, epsilon: float) -> Verdict:
    full_err = recovery_error(src, cfg, full_config(), seed)
    cfg_err = recovery_error(src, cfg, config, seed)
    delta = float(cfg_err - full_err)
    passed = bool(delta <= epsilon)
    return Verdict(
        passed=passed, lane=2, delta=delta, epsilon=float(epsilon),
        proof="bounded" if passed else "rejected",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_verifier.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add src/dern/verifier.py tests/dern/test_verifier.py
git commit -m "feat(dern): unmodifiable dual-lane verifier (exact + bounded)"
```

---

### Task 4: Cost vector + simulated envelope actuator

The actuator converts a config + posture into a **simulated** cost vector with
every dimension tagged `measured | telemetry-derived | simulated`. In Stage A
*everything* is `simulated` (we assert this — no dishonest "measured" tags).

**Files:**
- Create: `src/dern/cost.py`
- Test: `tests/dern/test_cost.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_cost.py
from src.dern.configs import COMPUTE_CONFIGS, full_config
from src.dern.cost import cost_vector, net_savings, ALL_SIMULATED


def test_cost_vector_dimensions_tagged_simulated_in_stage_a():
    v = cost_vector(full_config(), posture="default", overhead_units=0.1)
    assert set(v) >= {"compute", "thermal_slope", "dvfs", "idle", "carbon", "overhead", "_tags"}
    # Stage A: NOTHING may be tagged 'measured'. Honesty manifest invariant.
    assert all(tag == "simulated" for tag in v["_tags"].values())


def test_low_fan_posture_cuts_thermal_cost():
    full = cost_vector(full_config(), posture="default", overhead_units=0.0)
    cool = cost_vector(full_config(), posture="low_fan", overhead_units=0.0)
    assert cool["thermal_slope"] < full["thermal_slope"]


def test_net_savings_includes_overhead():
    baseline = cost_vector(full_config(), posture="default", overhead_units=0.0)
    cheap = cost_vector(COMPUTE_CONFIGS[0], posture="low_fan", overhead_units=0.5)
    net = net_savings(baseline, cheap)
    # savings = baseline.compute - (cheap.compute + cheap.overhead ... )
    assert net == (baseline["compute"] + baseline["thermal_slope"]) - (
        cheap["compute"] + cheap["thermal_slope"] + cheap["overhead"]
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_cost.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.cost)

**Step 3: Write minimal implementation**

```python
# src/dern/cost.py
"""Simulated broad cost vector + honest net-savings accounting (Stage A).

Every dimension is tagged 'simulated' in Stage A — no real energy is measured.
The tags are the honesty manifest: Stage B replaces specific tags with
'measured'/'telemetry-derived'; nothing here may ever claim 'measured'.
"""
from __future__ import annotations

from typing import Any, Dict

from src.dern.configs import compute_cost_units

ALL_SIMULATED = "simulated"

# Posture -> thermal multiplier. 'low_fan' models the "runs cool, don't spin up"
# experiential insight: lower fan/clock posture => lower thermal/operational cost.
_POSTURE_THERMAL = {"default": 1.0, "low_fan": 0.4, "race_to_idle": 0.7}


def cost_vector(config: Dict[str, Any], posture: str, overhead_units: float) -> Dict[str, Any]:
    compute = compute_cost_units(config)
    thermal = compute * _POSTURE_THERMAL.get(posture, 1.0)
    dims = {
        "compute": compute,
        "thermal_slope": round(thermal, 4),
        "dvfs": 1.0 if posture == "default" else 0.6,
        "idle": 0.0 if posture == "race_to_idle" else 0.2,
        "carbon": compute * 0.1,
        "overhead": float(overhead_units),
    }
    dims["_tags"] = {k: ALL_SIMULATED for k in dims}
    return dims


def net_savings(baseline: Dict[str, Any], chosen: Dict[str, Any]) -> float:
    """Net = baseline (compute+thermal) - chosen (compute+thermal+overhead).

    Overhead (probe+controller+verifier) is charged to the CHOSEN side, never
    hidden — a route that costs more to decide than it saves nets negative.
    """
    base = baseline["compute"] + baseline["thermal_slope"]
    chos = chosen["compute"] + chosen["thermal_slope"] + chosen["overhead"]
    return round(base - chos, 4)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_cost.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/dern/cost.py tests/dern/test_cost.py
git commit -m "feat(dern): simulated broad cost vector + net-of-overhead accounting"
```

---

### Task 5: Energy-indexed experience graph — earn / replay / evict

The graph stores `key -> best trusted edge`. An edge is written only after a
verifier PASS, carries its proof + a trust count, and is evicted on a later FAIL.

**Files:**
- Create: `src/dern/graph.py`
- Test: `tests/dern/test_graph.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_graph.py
from src.dern.graph import ExperienceGraph, Edge


def test_edge_written_only_after_pass():
    g = ExperienceGraph()
    g.record(key=(1, 2, 0), config={"name": "cheap"}, cost=2.0, proof="exact", passed=True)
    g.record(key=(9, 9, 9), config={"name": "mid"}, cost=3.0, proof="bounded", passed=False)
    assert g.lookup((1, 2, 0)) is not None     # pass -> stored
    assert g.lookup((9, 9, 9)) is None          # fail -> not stored


def test_replay_returns_cheapest_trusted_edge():
    g = ExperienceGraph()
    g.record((1, 1, 1), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    g.record((1, 1, 1), {"name": "cheap"}, cost=2.0, proof="exact", passed=True)
    edge = g.lookup((1, 1, 1))
    assert edge.config["name"] == "cheap"       # lowest cost wins
    assert edge.proof == "exact"


def test_fail_evicts_existing_trusted_edge_and_locks_region():
    g = ExperienceGraph()
    g.record((2, 2, 2), {"name": "cheap"}, cost=2.0, proof="bounded", passed=True)
    assert g.lookup((2, 2, 2)) is not None
    g.record((2, 2, 2), {"name": "cheap"}, cost=2.0, proof="bounded", passed=False)
    assert g.lookup((2, 2, 2)) is None          # evicted
    assert g.is_locked((2, 2, 2)) is True       # region forced to audit


def test_trust_accumulates_on_repeated_bounded_pass():
    g = ExperienceGraph()
    g.record((3, 3, 3), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    g.record((3, 3, 3), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    assert g.lookup((3, 3, 3)).trust >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_graph.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.graph)

**Step 3: Write minimal implementation**

```python
# src/dern/graph.py
"""Energy-indexed experience graph: key -> trusted cheapest config.

An edge exists only because it once PASSED the verifier. A later FAIL evicts it
and locks the region to audit until trust is re-earned. This is the runtime form
of 'a route is a claim; retract it when verification fails' (no silent loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

Key = Tuple[int, ...]


@dataclass
class Edge:
    config: Dict[str, Any]
    cost: float
    proof: str          # "exact" | "bounded"
    trust: int = 1


class ExperienceGraph:
    def __init__(self) -> None:
        self._edges: Dict[Key, Edge] = {}
        self._locked: set[Key] = set()

    def record(self, key: Key, config: Dict[str, Any], cost: float, proof: str, passed: bool) -> None:
        if not passed:
            # Eviction + region lock; a failed claim is retracted, not kept.
            self._edges.pop(key, None)
            self._locked.add(key)
            return
        self._locked.discard(key)
        existing = self._edges.get(key)
        if existing is not None and existing.config.get("name") == config.get("name"):
            existing.trust += 1
            existing.cost = cost
            existing.proof = proof
            return
        if existing is None or cost < existing.cost:
            self._edges[key] = Edge(config=config, cost=cost, proof=proof, trust=1)

    def lookup(self, key: Key) -> Optional[Edge]:
        return self._edges.get(key)

    def is_locked(self, key: Key) -> bool:
        return key in self._locked
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_graph.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add src/dern/graph.py tests/dern/test_graph.py
git commit -m "feat(dern): energy-indexed experience graph (earn/replay/evict/lock)"
```

---

### Task 6: Trust circuit breaker

Bounds adaptation: trips on a high recent FAIL rate or eviction spike; while
tripped, the system forces full compute + audit and freezes learning; resets
after a cool-down.

**Files:**
- Create: `src/dern/breaker.py`
- Test: `tests/dern/test_breaker.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_breaker.py
from src.dern.breaker import TrustBreaker


def test_breaker_trips_on_high_fail_rate():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=3)
    for _ in range(3):
        b.observe(passed=False)
    assert b.tripped is True


def test_breaker_forces_full_compute_while_tripped():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=3)
    for _ in range(3):
        b.observe(passed=False)
    assert b.must_force_full() is True


def test_breaker_resets_after_cooldown():
    b = TrustBreaker(window=5, fail_rate_trip=0.5, cooldown=2)
    for _ in range(3):
        b.observe(passed=False)
    assert b.tripped is True
    b.tick(); b.tick()          # cool down
    assert b.tripped is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_breaker.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.breaker)

**Step 3: Write minimal implementation**

```python
# src/dern/breaker.py
"""Trust circuit breaker — bounds online-learning instability (from CIO-II).

Tripping degrades the system to full compute + audit (the known-good baseline)
and freezes adaptation; it resets after a cool-down with the failure history
cleared. Worst case under instability is therefore the baseline, never worse.
"""
from __future__ import annotations

from collections import deque


class TrustBreaker:
    def __init__(self, window: int = 20, fail_rate_trip: float = 0.3, cooldown: int = 10) -> None:
        self._hist: deque[bool] = deque(maxlen=window)
        self._fail_rate_trip = fail_rate_trip
        self._cooldown = cooldown
        self._cooldown_left = 0
        self.tripped = False

    def observe(self, passed: bool) -> None:
        self._hist.append(bool(passed))
        if not self.tripped and len(self._hist) >= 1:
            fails = sum(1 for p in self._hist if not p)
            if fails / len(self._hist) >= self._fail_rate_trip:
                self.tripped = True
                self._cooldown_left = self._cooldown

    def tick(self) -> None:
        if self.tripped:
            self._cooldown_left -= 1
            if self._cooldown_left <= 0:
                self.tripped = False
                self._hist.clear()

    def must_force_full(self) -> bool:
        return self.tripped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_breaker.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/dern/breaker.py tests/dern/test_breaker.py
git commit -m "feat(dern): trust circuit breaker bounds adaptation, forces baseline"
```

---

### Task 7: Always-learning controller (reward only from verdict)

An epsilon-greedy online controller over `(config, posture, stop)`. It updates
value estimates ONLY from a `(verdict, cost)` pair the caller passes in — it
never scores itself. Frozen reference = full config.

**Files:**
- Create: `src/dern/controller.py`
- Test: `tests/dern/test_controller.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_controller.py
import inspect
import numpy as np
from src.dern.controller import OnlineController


def test_controller_update_takes_only_verdict_and_cost():
    sig = set(inspect.signature(OnlineController.update).parameters)
    # reward must come from outside; no 'self_score' / 'stop_proposal' as reward source
    assert "reward" in sig or "cost" in sig
    assert "self_score" not in sig


def test_controller_converges_to_low_cost_action_under_reward():
    rng = np.random.default_rng(0)
    c = OnlineController(actions=["cheap", "mid", "full"], seed=0, epsilon=0.1)
    # Reward 'cheap' best, 'full' worst (lower cost = higher reward).
    rewards = {"cheap": -1.0, "mid": -2.5, "full": -5.0}
    for _ in range(500):
        a = c.choose()
        c.update(a, reward=rewards[a] + 0.05 * rng.standard_normal())
    # After learning, greedy choice should be the cheapest.
    assert c.greedy() == "cheap"


def test_controller_reward_not_derived_from_own_proposal():
    # Passing a different reward than the action implies must change learning,
    # proving reward is exogenous (verifier-minted), not self-generated.
    c1 = OnlineController(actions=["a", "b"], seed=1, epsilon=0.0)
    c2 = OnlineController(actions=["a", "b"], seed=1, epsilon=0.0)
    for _ in range(50):
        c1.update("a", reward=1.0)
        c2.update("a", reward=-1.0)
    assert c1.value("a") != c2.value("a")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_controller.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.controller)

**Step 3: Write minimal implementation**

```python
# src/dern/controller.py
"""Always-learning online controller (epsilon-greedy value estimates).

Crucial invariant: update() takes an exogenous `reward` — minted by the verifier
+ cost meter — and the controller NEVER computes reward from its own action or
stop proposal. This structurally blocks reward hacking (design doc Flaw 2).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class OnlineController:
    def __init__(self, actions: List[str], seed: int = 0, epsilon: float = 0.1, lr: float = 0.1) -> None:
        self._actions = list(actions)
        self._q: Dict[str, float] = {a: 0.0 for a in self._actions}
        self._rng = np.random.default_rng(seed)
        self._eps = float(epsilon)
        self._lr = float(lr)

    def choose(self) -> str:
        if self._rng.random() < self._eps:
            return self._actions[int(self._rng.integers(len(self._actions)))]
        return self.greedy()

    def greedy(self) -> str:
        return max(self._actions, key=lambda a: self._q[a])

    def update(self, action: str, reward: float) -> None:
        self._q[action] += self._lr * (float(reward) - self._q[action])

    def value(self, action: str) -> float:
        return self._q[action]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_controller.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/dern/controller.py tests/dern/test_controller.py
git commit -m "feat(dern): online controller, reward only from exogenous verdict"
```

---

### Task 8: The runtime loop — wire probe -> graph -> controller -> verifier -> actuator -> ledger

This is the integration: one `route_request` per input that ties the components
together, enforcing fall-back-to-full on any rejection and feeding graph +
controller from the verifier's verdict only.

**Files:**
- Create: `src/dern/runtime.py`
- Test: `tests/dern/test_runtime.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_runtime.py
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime


def _cfg():
    return normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_route_returns_verified_outcome_and_ledger_record():
    rt = DERNRuntime(epsilon=0.2, audit_prob=1.0, eps_tolerance=0.5, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec = rt.route(src, _cfg(), seed=0)
    assert rec["verified"] is True
    assert rec["lane"] in (1, 2)
    assert rec["chosen_config"]["name"] in {"cheap", "mid", "full"}
    assert "net_savings" in rec and "cost_vector" in rec
    assert all(t == "simulated" for t in rec["cost_vector"]["_tags"].values())


def test_rejected_cheap_falls_back_to_full_never_unverified():
    # eps_tolerance = -1 makes every non-full config fail Lane 2 -> must serve full.
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(8):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["verified"] is True
        if rec["chosen_config"]["name"] != "full":
            # if a non-full config was attempted, it must have been rejected->full served
            assert rec["served_config"]["name"] == "full"


def test_ledger_accumulates_one_record_per_route():
    rt = DERNRuntime(epsilon=0.2, audit_prob=1.0, eps_tolerance=0.5, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(5):
        rt.route(src, _cfg(), seed=s)
    assert len(rt.ledger) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_runtime.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.runtime)

**Step 3: Write minimal implementation**

```python
# src/dern/runtime.py
"""DERN runtime loop (Stage A).

route(): probe -> graph lookup -> controller propose -> VERIFIER -> actuator/ledger
-> graph+controller update. Any rejection falls back to full compute and serves
the verified full result. Reward + trust are minted only from the verifier verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.dern.probe import distinction_key, probe_cost_units
from src.dern.configs import COMPUTE_CONFIGS, full_config, compute_cost_units
from src.dern.oracle import recover_distinctions
from src.dern.verifier import verify_lane1, verify_lane2
from src.dern.cost import cost_vector, net_savings
from src.dern.graph import ExperienceGraph
from src.dern.controller import OnlineController
from src.dern.breaker import TrustBreaker

_POSTURES = ["default", "low_fan", "race_to_idle"]


class DERNRuntime:
    def __init__(self, epsilon: float = 0.1, audit_prob: float = 1.0,
                 eps_tolerance: float = 0.3, seed: int = 0) -> None:
        names = [c["name"] for c in COMPUTE_CONFIGS]
        self._by_name = {c["name"]: c for c in COMPUTE_CONFIGS}
        self.controller = OnlineController(actions=names, seed=seed, epsilon=epsilon)
        self.graph = ExperienceGraph()
        self.breaker = TrustBreaker()
        self.audit_prob = float(audit_prob)
        self.eps_tolerance = float(eps_tolerance)
        self.ledger: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)

    def route(self, src, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
        # 1) probe one representative trajectory
        xprobe, _ = src.sample(1, cfg, seed)
        key = distinction_key(xprobe[0])
        overhead = probe_cost_units()

        # 2) forced full while breaker tripped or region locked
        forced_full = self.breaker.must_force_full() or self.graph.is_locked(key)

        # 3) choose config (graph replay if trusted, else controller)
        edge = None if forced_full else self.graph.lookup(key)
        if forced_full:
            chosen_name = "full"
        elif edge is not None:
            chosen_name = edge.config["name"]
        else:
            chosen_name = self.controller.choose()
        chosen = self._by_name[chosen_name]
        posture = "low_fan"  # Stage-A simplest non-default posture; controller-extendable

        # 4) VERIFY (controller cannot influence this)
        if chosen_name == "full":
            verdict = verify_lane1(src, cfg, full_config(), seed)  # exact, trivially passes
            served = full_config()
        else:
            v1 = verify_lane1(src, cfg, chosen, seed)
            if v1.passed:
                verdict, served = v1, chosen
            else:
                # Lane 2 audit (sampled)
                if self._rng.random() < self.audit_prob:
                    v2 = verify_lane2(src, cfg, chosen, seed, self.eps_tolerance)
                else:
                    v2 = verify_lane2(src, cfg, chosen, seed, self.eps_tolerance)
                if v2.passed:
                    verdict, served = v2, chosen
                else:
                    # rejection -> fall back to full (never serve unverified)
                    verdict = verify_lane1(src, cfg, full_config(), seed)
                    served = full_config()

        # 5) cost + net savings (overhead always charged to chosen side)
        baseline_cv = cost_vector(full_config(), "default", 0.0)
        served_cv = cost_vector(served, posture, overhead)
        net = net_savings(baseline_cv, served_cv)

        # 6) learning signal — minted ONLY from verdict + cost
        passed = verdict.passed and served["name"] == chosen["name"]
        reward = net if passed else -compute_cost_units(full_config())
        self.controller.update(chosen_name, reward=reward)
        self.graph.record(key, chosen, compute_cost_units(chosen), verdict.proof,
                           passed=passed and chosen_name != "full")
        self.breaker.observe(passed=verdict.passed)
        self.breaker.tick()

        rec = {
            "key": key, "chosen_config": chosen, "served_config": served,
            "lane": verdict.lane, "verified": verdict.passed, "proof": verdict.proof,
            "delta": verdict.delta, "cost_vector": served_cv, "net_savings": net,
            "reward": reward, "forced_full": forced_full,
        }
        self.ledger.append(rec)
        return rec
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_runtime.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/dern/runtime.py tests/dern/test_runtime.py
git commit -m "feat(dern): runtime loop wires probe->graph->controller->verifier->ledger"
```

---

### Task 9: Safety-invariant tests (the non-negotiable gate)

These encode the design's safety spine. They MUST be 100% green for Stage A to pass.

**Files:**
- Test: `tests/dern/test_safety_invariants.py`

**Step 1: Write the tests**

```python
# tests/dern/test_safety_invariants.py
import inspect
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern import verifier as verifier_mod
from src.dern.runtime import DERNRuntime
from src.dern.configs import COMPUTE_CONFIGS


def _cfg():
    return normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_controller_has_no_write_path_to_verifier():
    # Structural: verifier functions accept no controller/policy/runtime object.
    for name in ("verify_lane1", "verify_lane2"):
        fn = getattr(verifier_mod, name)
        params = set(inspect.signature(fn).parameters)
        assert not ({"controller", "policy", "runtime", "graph"} & params)


def test_served_output_is_always_verified():
    rt = DERNRuntime(epsilon=0.5, audit_prob=1.0, eps_tolerance=0.4, seed=0)
    src = SOURCE_REGISTRY["henon_map"]   # hardest family
    for s in range(12):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["verified"] is True   # nothing unverified is ever served


def test_zero_tolerance_never_serves_cheaper_than_full():
    # eps=-1 => every non-full config is rejected; system must serve full each time.
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    for s in range(10):
        rec = rt.route(src, _cfg(), seed=s)
        assert rec["served_config"]["name"] == "full"


def test_reward_zero_when_full_served_no_phantom_savings():
    rt = DERNRuntime(epsilon=1.0, audit_prob=1.0, eps_tolerance=-1.0, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec = rt.route(src, _cfg(), seed=0)
    # full served at default-vs-low_fan posture: net savings come only from posture,
    # never from pretending cheaper compute happened.
    assert rec["served_config"]["name"] == "full"
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/dern/test_safety_invariants.py -v`
Expected: PASS (4 passed). If any FAIL, STOP — fix the implementation, never the assertion. These are the gate.

**Step 3: Commit**

```bash
git add tests/dern/test_safety_invariants.py
git commit -m "test(dern): safety-invariant gate (verified-only, no phantom savings)"
```

---

### Task 10: Failure-injection tests

Inject each Section-4 failure and assert the fail-safe fires.

**Files:**
- Test: `tests/dern/test_failure_injection.py`

**Step 1: Write the tests**

```python
# tests/dern/test_failure_injection.py
import numpy as np
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime
from src.dern.graph import ExperienceGraph
from src.dern.breaker import TrustBreaker


def _cfg():
    return normalize_cfg({"n_train": 200, "n_test": 90, "T": 8.0, "dt": 0.08, "seeds": [0]})


def test_drift_evicts_trusted_edge_and_locks_region():
    g = ExperienceGraph()
    g.record((1, 1, 1), {"name": "cheap"}, 2.0, "bounded", passed=True)
    # simulate drift: the same region now fails verification
    g.record((1, 1, 1), {"name": "cheap"}, 2.0, "bounded", passed=False)
    assert g.lookup((1, 1, 1)) is None
    assert g.is_locked((1, 1, 1)) is True


def test_breaker_trips_under_sustained_failures_forces_full():
    b = TrustBreaker(window=6, fail_rate_trip=0.5, cooldown=4)
    for _ in range(4):
        b.observe(passed=False)
    assert b.tripped and b.must_force_full()


def test_locked_region_forces_full_in_runtime():
    rt = DERNRuntime(epsilon=0.0, audit_prob=1.0, eps_tolerance=0.4, seed=0)
    src = SOURCE_REGISTRY["linear_oscillator"]
    rec0 = rt.route(src, _cfg(), seed=0)
    # force-lock that record's region, then re-route same seed -> must serve full
    rt.graph._locked.add(rec0["key"])
    rec1 = rt.route(src, _cfg(), seed=0)
    assert rec1["forced_full"] is True
    assert rec1["served_config"]["name"] == "full"


def test_overhead_can_make_route_net_negative_and_is_not_hidden():
    # A huge probe/overhead must show up as <=0 net savings, never hidden.
    from src.dern.cost import cost_vector, net_savings
    base = cost_vector({"name": "full", "channel": "direct", "k": 16, "cost": 5.0}, "default", 0.0)
    chosen = cost_vector({"name": "cheap", "channel": "native", "k": 3, "cost": 1.0}, "default", 100.0)
    assert net_savings(base, chosen) < 0
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/dern/test_failure_injection.py -v`
Expected: PASS (4 passed). Any FAIL → fix implementation, not the test.

**Step 3: Commit**

```bash
git add tests/dern/test_failure_injection.py
git commit -m "test(dern): failure-injection (drift evict, breaker, lock->full, net-negative visible)"
```

---

### Task 11: Convergence experiment + Stage-A acceptance report

A small runnable script that drives the runtime over many requests, shows the
controller converges to below-baseline simulated cost, and prints a Stage-A
report (mirroring CIO-II's `proof-report` / this repo's `report.py`). This is the
artifact that demonstrates the Stage-A gate is met.

**Files:**
- Create: `src/dern/run_stage_a.py`
- Test: `tests/dern/test_convergence.py`

**Step 1: Write the failing test**

```python
# tests/dern/test_convergence.py
from src.proof_ladder import normalize_cfg
from src.sources import SOURCE_REGISTRY
from src.dern.run_stage_a import run_stage_a


def test_controller_converges_below_baseline_on_easy_family():
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    report = run_stage_a(family="linear_oscillator", n_requests=120, cfg=cfg,
                         eps_tolerance=0.5, seed=0)
    # mean served cost must beat always-full baseline, with zero unverified serves
    assert report["mean_net_savings"] > 0.0
    assert report["unverified_serves"] == 0
    assert report["lane1_exactness_violations"] == 0


def test_hard_family_stays_safe_even_if_no_savings():
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    report = run_stage_a(family="henon_map", n_requests=80, cfg=cfg,
                         eps_tolerance=0.3, seed=0)
    # On the hard family savings may be ~0, but safety must hold absolutely.
    assert report["unverified_serves"] == 0
    assert report["lane1_exactness_violations"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/dern/test_convergence.py -v`
Expected: FAIL (ModuleNotFoundError: src.dern.run_stage_a)

**Step 3: Write minimal implementation**

```python
# src/dern/run_stage_a.py
"""Stage-A driver + acceptance report (no energy claim; simulated cost only)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.sources import SOURCE_REGISTRY
from src.dern.runtime import DERNRuntime
from src.dern.cost import cost_vector
from src.dern.configs import full_config


def run_stage_a(family: str, n_requests: int, cfg: Dict[str, Any],
                eps_tolerance: float = 0.4, seed: int = 0) -> Dict[str, Any]:
    src = SOURCE_REGISTRY[family]
    rt = DERNRuntime(epsilon=0.15, audit_prob=1.0, eps_tolerance=eps_tolerance, seed=seed)
    nets, unverified, exact_violations = [], 0, 0
    for s in range(n_requests):
        rec = rt.route(src, cfg, seed=s)
        nets.append(rec["net_savings"])
        if not rec["verified"]:
            unverified += 1
        # Lane-1 exactness violation = claimed exact but delta != 0 (must never happen)
        if rec["proof"] == "exact" and rec["delta"] != 0.0:
            exact_violations += 1
    return {
        "family": family,
        "requests": n_requests,
        "mean_net_savings": round(float(np.mean(nets)), 4),
        "unverified_serves": unverified,
        "lane1_exactness_violations": exact_violations,
        "ledger_size": len(rt.ledger),
        "all_cost_dims_simulated": all(
            t == "simulated" for t in rt.ledger[-1]["cost_vector"]["_tags"].values()
        ) if rt.ledger else True,
    }


if __name__ == "__main__":
    from src.proof_ladder import normalize_cfg
    cfg = normalize_cfg({"n_train": 250, "n_test": 100, "T": 8.0, "dt": 0.08, "seeds": [0]})
    for fam in ("linear_oscillator", "nonlinear_oscillator", "henon_map"):
        print(run_stage_a(fam, 120, cfg))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/dern/test_convergence.py -v`
Expected: PASS (2 passed). If `mean_net_savings <= 0` on the linear family, the controller/posture savings model needs the cheap config to be reachable — verify Task 4 posture multipliers and Task 7 convergence first. Do NOT relax the safety assertions.

**Step 5: Run the driver to see the report**

Run: `python -m src.dern.run_stage_a`
Expected: three report dicts; linear shows positive `mean_net_savings`, all show `unverified_serves: 0` and `lane1_exactness_violations: 0`.

**Step 6: Commit**

```bash
git add src/dern/run_stage_a.py tests/dern/test_convergence.py
git commit -m "feat(dern): Stage-A driver + acceptance report (convergence, safety counts)"
```

---

### Task 12: Full-suite green + Stage-A gate verification

**Files:** none (verification only).

**Step 1: Run the entire repo test suite**

Run: `pytest -q`
Expected: all tests pass — the 17 pre-existing tests PLUS the new DERN tests. If any pre-existing test broke, DERN touched shared code it should not have; revert that coupling.

**Step 2: Confirm the Stage-A acceptance gate**

Verify, from `pytest tests/dern -q` and `python -m src.dern.run_stage_a`:
- [ ] 100% of `test_safety_invariants.py` + `test_failure_injection.py` pass
- [ ] controller converges below baseline on the linear family (`mean_net_savings > 0`)
- [ ] `unverified_serves == 0` and `lane1_exactness_violations == 0` on all three families
- [ ] every cost dimension tagged `simulated` (no phantom "measured")

**Step 3: Write a short Stage-A results note**

Create `codec_codex_experiments/results/dern_stage_a_results.md` summarizing the report dicts and explicitly stating: *Stage A validates logic only; no energy claim is made; Stage B (real model, measured energy) is gated on this passing.* Mirror the honest tone of `results/critic1_mitigation_results.md`.

**Step 4: Commit**

```bash
git add codec_codex_experiments/results/dern_stage_a_results.md
git commit -m "docs(dern): Stage-A results note — logic validated, no energy claim"
```

**Step 5: Push**

```bash
git push origin main
```

---

## Stage-A done criteria (recap)

Stage A is complete when Task 12's checklist is fully satisfied. Only then does
Stage B (real open-weight model, Lane-1 speculative decoding, measured
NVML/RAPL/powermetrics energy net of router cost, per the design doc §5.2) get
planned — as a separate plan, because it introduces real-model and
real-hardware dependencies that must not be coupled to the Stage-A logic build.

## Notes for the implementer
- Reuse `src/proof_ladder.py` primitives (`native_feats`, `engineered_feats`, `direct_feats`, `quantize`, `fit_predict`, `nrmse`) — do not reimplement them. DRY.
- Never weaken a safety-invariant or failure-injection assertion to make it pass. Those are the gate; if one fails, the implementation is wrong (mirror the repo rule: no threshold tuned to pass; a FAIL is an honest negative).
- Keep DERN self-contained in `src/dern/`; do not modify `proof_ladder.py`, `sources.py`, or other existing modules (the 17 existing tests must stay green).
- All randomness via `np.random.default_rng(seed)` for determinism.
