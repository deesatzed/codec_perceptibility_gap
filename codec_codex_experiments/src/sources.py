#!/usr/bin/env python3
"""Source-family registry for the F5 cross-family generalization mitigation.

A SourceFamily produces (trajectory, theta) datasets and declares its native /
expanded-physics channels. The proof-ladder and codec stages operate on the
generic trajectory array, so adding a family requires no stage-logic change.

Hard invariant (carried forward from the F1 leakage finding): no channel a
family exposes may contain the scored target. `expanded_physics_channel` is
built only from observables; any leaked channel must be named a
``*_leak_positive_control`` and reported separately.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

from .proof_ladder import (
    simulate,
    sample_theta,
    native_feats,
    coupling_feats,
    engineered_feats,
)


def simulate_nonlinear(theta: np.ndarray, n_osc: int, cfg: Dict[str, Any], seed: int) -> np.ndarray:
    """Duffing-type coupled oscillators: linear dynamics + cubic stiffness.

    theta: (B, n_osc+3) = [springs..., coupling g, damping c, nonlinearity beta].
    The cubic `-beta * x^3` term makes recovery genuinely harder as dimensions
    rise (the difficulty slope the linear family cannot produce; see S7 / F5).
    Acceleration and position are clipped exactly as the magnetic simulator does
    to keep the integrator stable on a local machine.
    """
    B = theta.shape[0]
    springs = theta[:, :n_osc]
    g = theta[:, n_osc]
    c = theta[:, n_osc + 1]
    beta = theta[:, n_osc + 2]
    steps = int(cfg["T"] / cfg["dt"])
    x = np.zeros((B, n_osc), dtype=float)
    v = np.zeros((B, n_osc), dtype=float)
    x[:, 0] = 1.0
    out = np.empty((B, steps, n_osc), dtype=float)
    for t in range(steps):
        lap = np.zeros_like(x)
        if n_osc > 1:
            lap[:, 1:] += x[:, :-1] - x[:, 1:]
            lap[:, :-1] += x[:, 1:] - x[:, :-1]
        drive = cfg["drive_amp"] * np.sin(cfg["drive_w"] * t * cfg["dt"])
        a = -springs * x - beta[:, None] * x ** 3 - c[:, None] * v + g[:, None] * lap + drive
        a = np.clip(a, -50.0, 50.0)
        v = v + cfg["dt"] * a
        x = x + cfg["dt"] * v
        x = np.clip(x, -10.0, 10.0)
        out[:, t, :] = x
    return out


def sample_theta_nonlinear(B: int, n_osc: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(7 + seed)
    springs = rng.uniform(0.5, 4.0, size=(B, n_osc))
    g = rng.uniform(0.05, 1.0, size=(B, 1))
    c = rng.uniform(0.10, 0.60, size=(B, 1))
    beta = rng.uniform(0.3, 2.5, size=(B, 1))
    return np.concatenate([springs, g, c, beta], axis=1)


# --------- discrete-time chaotic map family (critic1 criticism 3) ----------
# A coupled lattice of Henon maps. This is the "fundamentally distinct
# dynamical class" the critique asked for: state jumps discontinuously each
# step (no continuous momentum, no restoring force, no periodic/quasi-periodic
# attractor symmetry the oscillator families share). If the CodecGuard audit
# signal (S8) survives here, it is not merely tracking "a pendulum swings back".
#
# Per-site map: x_{t+1} = 1 - a_i * x_t^2 + y_t + g * (lattice diffusion of x);
#               y_{t+1} = b * x_t.
# The observable trajectory is the x-channel per site, shape (B, T, N) — the
# SAME shape the generic feature functions consume, so no stage logic changes.
def simulate_henon(theta: np.ndarray, n_sites: int, cfg: Dict[str, Any], seed: int) -> np.ndarray:
    """theta: (B, n_sites+2) = [a_1..a_N (chaos params), coupling g, b].

    Coupling note: a discrete Henon map only stays on its bounded strange
    attractor for a narrow parameter window. The *additive* Laplacian coupling
    used by the continuous oscillator simulators injects energy every step and
    blows the orbit to the clip wall (empirically >45% of orbits saturate at
    even small g). We therefore couple *parametrically* — each site's effective
    chaos parameter is nudged toward its neighbours' recent activity:
    ``a_eff_i = a_i * (1 - g * tanh(neighbour_mean))``. This keeps every orbit
    on a bounded attractor (no saturation) while still making the coupling
    leave a recoverable signature, which is what the audit needs to read.
    """
    B = theta.shape[0]
    a = theta[:, :n_sites]
    g = theta[:, n_sites]
    b = theta[:, n_sites + 1]
    steps = int(cfg["T"] / cfg["dt"])
    # Discrete iterations: one map step per output sample (no sub-stepping).
    rng = np.random.default_rng(2024 + seed)
    x = rng.uniform(-0.1, 0.1, size=(B, n_sites))
    y = rng.uniform(-0.1, 0.1, size=(B, n_sites))
    out = np.empty((B, steps, n_sites), dtype=float)
    for t in range(steps):
        nbr = np.zeros_like(x)
        if n_sites > 1:
            nbr[:, 1:] += x[:, :-1]
            nbr[:, :-1] += x[:, 1:]
            denom = np.ones(n_sites)
            denom[1:-1] = 2.0  # interior sites have two neighbours
            nbr = nbr / denom
        a_eff = a * (1.0 - g[:, None] * np.tanh(nbr))
        x_new = 1.0 - a_eff * x ** 2 + y
        y_new = b[:, None] * x
        # Bounded attractor stays within this range; the clip is only a finite
        # backstop for the rare divergent-parameter draw.
        x = np.clip(x_new, -10.0, 10.0)
        y = np.clip(y_new, -10.0, 10.0)
        out[:, t, :] = x
    return out


def sample_theta_henon(B: int, n_sites: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(11 + seed)
    # a spanning the period-doubling-to-chaos window of the Henon map; the lower
    # end is periodic/mixed, the upper end (~1.4) is the classic strange
    # attractor. This range keeps orbits bounded and makes a partially readable.
    a = rng.uniform(0.8, 1.4, size=(B, n_sites))
    g = rng.uniform(0.05, 0.60, size=(B, 1))
    b = rng.uniform(0.20, 0.35, size=(B, 1))
    return np.concatenate([a, g, b], axis=1)


class SourceFamily(Protocol):
    key: str
    target_labels: List[str]
    target_kind: str  # "continuous" | "categorical" | "mixed"

    def sample(self, n: int, cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (trajectory (B,T,N), theta (B,D)); never puts theta in trajectory."""
        ...

    def native_channel(self, x: np.ndarray) -> np.ndarray:
        ...

    def expanded_physics_channel(self, x: np.ndarray) -> np.ndarray:
        ...


class LinearOscillator:
    """Existing forced, coupled, damped linear oscillator family.

    Delegates to the original proof_ladder functions so the linear path is
    byte-identical to pre-refactor behavior (Phase-0 regression guard).
    """

    key = "linear_oscillator"
    target_kind = "continuous"

    def __init__(self, n_osc: int = 2) -> None:
        self.n_osc = n_osc
        self.target_labels = (
            [f"spring{i+1}" for i in range(n_osc)] + ["coupling_g", "damping_c"]
        )

    def sample(self, n: int, cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = sample_theta(int(n), self.n_osc, seed)
        traj = simulate(theta, self.n_osc, cfg, seed)
        return traj, theta

    def native_channel(self, x: np.ndarray) -> np.ndarray:
        return native_feats(x)

    def expanded_physics_channel(self, x: np.ndarray) -> np.ndarray:
        # native + cross-oscillator coupling structure; contains no target.
        return engineered_feats(x)


class NonlinearOscillator:
    """Duffing-type coupled oscillators: same observable shape as the linear
    family, but cubic stiffness makes added dimensions genuinely harder.

    Shares the linear family's generic feature channels (they operate on the
    raw trajectory and know nothing about the generator)."""

    key = "nonlinear_oscillator"
    target_kind = "continuous"

    def __init__(self, n_osc: int = 2) -> None:
        self.n_osc = n_osc
        self.target_labels = (
            [f"spring{i+1}" for i in range(n_osc)] + ["coupling_g", "damping_c", "nonlinearity_beta"]
        )

    def sample(self, n: int, cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = sample_theta_nonlinear(int(n), self.n_osc, seed)
        traj = simulate_nonlinear(theta, self.n_osc, cfg, seed)
        return traj, theta

    def native_channel(self, x: np.ndarray) -> np.ndarray:
        return native_feats(x)

    def expanded_physics_channel(self, x: np.ndarray) -> np.ndarray:
        return engineered_feats(x)


class HenonMap:
    """Coupled lattice of Henon maps — a discrete-time chaotic class with no
    continuous momentum and no periodic-attractor symmetry (critic1 #3).

    Same observable shape as the oscillator families (the generic feature
    channels operate on the raw (B,T,N) trajectory and are generator-agnostic),
    but the underlying dynamics is fundamentally non-oscillatory: this is the
    cross-family stress test that strips the 'physics of pendulums' crutch from
    the CodecGuard audit signal. The scored targets are the per-site chaos
    parameters plus coupling; none is ever placed in an observable channel.
    """

    key = "henon_map"
    target_kind = "continuous"

    def __init__(self, n_osc: int = 2) -> None:
        # n_osc names the lattice size to share the S7 dimensionality-sweep API.
        self.n_osc = n_osc
        self.target_labels = (
            [f"chaos_a{i+1}" for i in range(n_osc)] + ["coupling_g", "feedback_b"]
        )

    def sample(self, n: int, cfg: Dict[str, Any], seed: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = sample_theta_henon(int(n), self.n_osc, seed)
        traj = simulate_henon(theta, self.n_osc, cfg, seed)
        return traj, theta

    def native_channel(self, x: np.ndarray) -> np.ndarray:
        return native_feats(x)

    def expanded_physics_channel(self, x: np.ndarray) -> np.ndarray:
        return engineered_feats(x)


# Registry. New families register here; resolve_sources reads cfg.
SOURCE_REGISTRY: Dict[str, SourceFamily] = {
    LinearOscillator().key: LinearOscillator(),
    NonlinearOscillator().key: NonlinearOscillator(),
    HenonMap().key: HenonMap(),
}


def get_source(cfg: Dict[str, Any]) -> SourceFamily:
    """Resolve the single primary source family from cfg (default linear)."""
    key = cfg.get("source", "linear_oscillator")
    if isinstance(key, list):
        key = key[0] if key else "linear_oscillator"
    if key not in SOURCE_REGISTRY:
        raise KeyError(f"unknown source family '{key}'; known: {list(SOURCE_REGISTRY)}")
    return SOURCE_REGISTRY[key]


def resolve_sources(cfg: Dict[str, Any]) -> List[SourceFamily]:
    """Resolve the list of families to run (cfg['sources'] or cfg['source'])."""
    keys = cfg.get("sources")
    if keys is None:
        single = cfg.get("source", "linear_oscillator")
        keys = single if isinstance(single, list) else [single]
    out = []
    for k in keys:
        if k not in SOURCE_REGISTRY:
            raise KeyError(f"unknown source family '{k}'; known: {list(SOURCE_REGISTRY)}")
        out.append(SOURCE_REGISTRY[k])
    return out
