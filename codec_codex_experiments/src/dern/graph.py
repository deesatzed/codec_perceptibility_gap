"""Energy-indexed experience graph: key -> trusted cheapest config.

An edge exists only because it once PASSED the verifier. A later FAIL evicts it
and locks the region to audit until trust is re-earned. This is the runtime form
of 'a route is a claim; retract it when verification fails' (no silent loss).
"""
from __future__ import annotations

from dataclasses import dataclass
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
