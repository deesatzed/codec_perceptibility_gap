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
