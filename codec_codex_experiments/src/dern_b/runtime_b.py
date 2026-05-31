"""DERN Stage-B cascade runtime (real models, bounded-loss Lane 2).

Reuses the Stage-A control spine (experience graph, trust breaker, online
controller) unchanged. Per prompt:

  probe -> key -> graph lookup -> controller proposes {cheap, reference}
  -> if 'reference' (or forced): serve reference (authority), record.
  -> if 'cheap': generate cheap AND reference, AUDIT cheap vs reference at eps.
       pass -> serve cheap, record measured savings, mint trust.
       fail -> serve reference (never serve worse-than-reference), evict+lock.

Reward + trust are minted ONLY from the verifier verdict + measured cost. The
breaker forces reference on instability. Note: auditing requires running the
reference too, so a *single audited* request does not save compute — savings come
from REPLAY: once a region is trusted, the cheap route is reused WITHOUT re-running
the reference (audit is amortized, sampled by audit_prob on trusted regions).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.dern.graph import ExperienceGraph
from src.dern.breaker import TrustBreaker
from src.dern.controller import OnlineController

from src.dern_b.mlx_backend import MLXModel, CHEAP_PATH, REF_PATH
from src.dern_b.probe_text import distinction_key, probe_cost_units
from src.dern_b.verifier_b import verify_against_reference
from src.dern_b.cost_b import cost_record, net_savings

_CHEAP = {"name": "cheap"}
_REF = {"name": "reference"}


class DERNBRuntime:
    def __init__(self, epsilon: float = 0.4, audit_prob: float = 1.0,
                 max_tokens: int = 64, seed: int = 0,
                 cheap_path: str = CHEAP_PATH, ref_path: str = REF_PATH,
                 metric: str = "token_f1", audit_every_replay: bool = True) -> None:
        self.cheap = MLXModel(cheap_path)
        self.ref = MLXModel(ref_path)
        self.controller = OnlineController(actions=["cheap", "reference"], seed=seed, epsilon=0.15)
        self.graph = ExperienceGraph()
        self.breaker = TrustBreaker()
        self.epsilon = float(epsilon)
        self.audit_prob = float(audit_prob)
        self.max_tokens = int(max_tokens)
        self.metric = metric
        # ultrathink fix (cross-prompt replay attribution): when True, EVERY
        # served-cheap route is audited against the reference for ITS OWN prompt,
        # so a saving can never be booked on the strength of a different prompt's
        # audit (the region-key collision exploit). This makes *measured* replay
        # savings ~0 (audit runs both models); steady-state savings are recovered
        # as an explicitly-labeled projection elsewhere. Default True = honest.
        self.audit_every_replay = bool(audit_every_replay)
        self.ledger: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)

    def route(self, prompt: str) -> Dict[str, Any]:
        key = distinction_key(prompt)
        overhead = probe_cost_units()
        forced_ref = self.breaker.must_force_full() or self.graph.is_locked(key)

        edge = None if forced_ref else self.graph.lookup(key)
        if forced_ref:
            proposed = "reference"
        elif edge is not None:
            proposed = edge.config["name"]
        else:
            proposed = self.controller.choose()

        ref_gen = None
        verdict = None

        if proposed == "reference":
            served_gen = self.ref.generate(prompt, self.max_tokens)
            served = _REF
            passed = True  # reference is the authority; trivially "verified"
        else:
            cheap_gen = self.cheap.generate(prompt, self.max_tokens)
            # Audit policy: untrusted region OR sampled trusted region -> run reference.
            trusted = edge is not None and edge.proof == "bounded"
            do_audit = (not trusted) or self.audit_every_replay or (self._rng.random() < self.audit_prob)
            if do_audit:
                ref_gen = self.ref.generate(prompt, self.max_tokens)
                verdict = verify_against_reference(cheap_gen.text, ref_gen.text, self.epsilon, self.metric)
                if verdict.passed:
                    served_gen, served, passed = cheap_gen, _CHEAP, True
                else:
                    served_gen, served, passed = ref_gen, _REF, False  # fall back to authority
            else:
                # trusted replay, no audit this turn: serve cheap, amortized savings.
                # Baseline is the region's RECORDED reference cost (real, from the
                # audit that earned trust) — NOT the cheap cost (which would
                # falsely show ~0 savings). edge.cost stored the reference aps.
                served_gen, served, passed = cheap_gen, _CHEAP, True
                replay_ref_aps = float(edge.cost) if edge is not None else None

        # measured cost of what was SERVED + baseline (always-reference)
        served_cv = cost_record(served_gen, overhead_units=overhead)
        # baseline = reference cost for this prompt; reuse ref_gen if we ran it,
        # else estimate baseline from the served reference (when served==reference
        # the served IS the baseline). For a served cheap with audit, ref_gen exists.
        if served["name"] == "reference":
            baseline_cv = cost_record(served_gen, overhead_units=0.0)
        elif ref_gen is not None:
            baseline_cv = cost_record(ref_gen, overhead_units=0.0)
        else:
            # trusted replay: baseline is the region's RECORDED reference cost
            # (real, measured during the audit that earned trust). We synthesize a
            # baseline cost vector carrying that reference active-param-seconds and
            # the served reference token/latency proxy from the edge metadata.
            baseline_cv = dict(served_cv)
            baseline_cv["active_param_seconds"] = replay_ref_aps if replay_ref_aps is not None else served_cv["active_param_seconds"]
            baseline_cv["overhead"] = 0.0

        tok_save = net_savings(baseline_cv, served_cv, "total_tokens")
        aps_save = net_savings(baseline_cv, served_cv, "active_param_seconds")
        lat_save = net_savings(baseline_cv, served_cv, "wall_seconds")

        # learning signal: reward from verdict + measured savings only
        reward = aps_save if (passed and served["name"] == "cheap") else (
            0.0 if served["name"] == "reference" else -float(baseline_cv["active_param_seconds"]))
        self.controller.update(proposed, reward=float(reward))
        # graph trust only on a PASSED, AUDITED cheap serve. Store the REFERENCE
        # active-param-seconds as edge.cost so future un-audited replays measure
        # savings against the real reference cost, not the cheap cost.
        if proposed == "cheap" and verdict is not None:
            ref_aps = float(ref_gen.active_param_seconds) if ref_gen is not None else served_cv["active_param_seconds"]
            self.graph.record(key, _CHEAP, ref_aps, "bounded", passed=verdict.passed)
        self.breaker.observe(passed=passed)
        self.breaker.tick()

        rec = {
            "key": key, "proposed": proposed, "served": served["name"],
            "audited": verdict is not None,
            "verdict_passed": (verdict.passed if verdict is not None else None),
            "delta": (verdict.delta if verdict is not None else None),
            "served_text": served_gen.text,
            "cost": served_cv, "baseline_cost": baseline_cv,
            "token_savings": tok_save, "aps_savings": aps_save, "latency_savings": lat_save,
            "forced_ref": forced_ref,
            # per-prompt audit truth for honest comparison accounting (ultrathink):
            "replay_route": bool(edge is not None and not forced_ref),
            "cheap_agreed_this_prompt": (verdict.passed if verdict is not None else None),
            "this_prompt_ref_aps": (float(ref_gen.active_param_seconds) if ref_gen is not None else None),
            "this_prompt_cheap_aps": (float(cheap_gen.active_param_seconds) if proposed == "cheap" else None),
        }
        self.ledger.append(rec)
        return rec
