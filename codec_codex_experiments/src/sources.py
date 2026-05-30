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


# Registry. New families register here; resolve_sources reads cfg.
SOURCE_REGISTRY: Dict[str, SourceFamily] = {
    LinearOscillator().key: LinearOscillator(),
    NonlinearOscillator().key: NonlinearOscillator(),
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
