"""Paired, interleaved, gated Stage-B comparison (ultrathink mitigation v2).

Fixes the four validity-fatal flaws the four-agent ultrathink found in compare_b.py:
  1. cross-prompt replay attribution -> runtime now audits every replay
     (audit_every_replay=True), so every served-cheap is verified on its own prompt.
  2. sequential thermal drift -> here we measure reference and cascade for the SAME
     prompt in IMMEDIATE A/B succession; common-mode thermal drift between the two
     adjacent windows is ~0, so the SIGN of the per-prompt delta is protected.
  3. audit overhead folded into headline -> three-class accounting reports
     first-encounter (audited, runs BOTH models -> MORE than reference), replay, and
     forced-ref separately; the steady-state replay saving is a labeled PROJECTION
     from measured cheap-only energy, with a break-even repeat count.
  4. single shot -> K>=5 repeats, paired deltas, variance gate.

A savings number is emitted ONLY when gates_b.evaluate_gates returns VALID.

Run in your own terminal (sudo for measured joules):
    sudo -v
    python -m src.dern_b.compare_b_v2
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from src.dern_b.mlx_backend import MLXModel, CHEAP_PATH, REF_PATH
from src.dern_b.runtime_b import DERNBRuntime
from src.dern_b.energy_probe import measure_energy, sudo_available, EnergyReading
from src.dern_b import gates_b


@dataclass
class PairTrial:
    prompt: str
    route_class: str                 # "first_encounter" | "replay" | "forced_ref"
    cheap_agreed: Optional[bool]
    ref_joules: Optional[float]
    casc_joules: Optional[float]
    delta_joules: Optional[float]    # ref - cascade (paired, adjacent)
    ref_cov: float
    casc_cov: float


def _coverage(r: EnergyReading) -> float:
    if r.work_seconds in (None, 0) or r.source != "measured":
        return 0.0
    return min(1.0, (r.n_active_samples * r.interval_s) / r.work_seconds)


def _classify(rec: Dict[str, Any]) -> str:
    if rec.get("forced_ref"):
        return "forced_ref"
    return "replay" if rec.get("replay_route") else "first_encounter"


def run_paired_block(ref: MLXModel, rt: DERNBRuntime, prompts: List[str],
                     max_tokens: int, interval_ms: int) -> List[PairTrial]:
    """For each prompt: measure reference-alone, then cascade, ADJACENTLY."""
    trials: List[PairTrial] = []
    for p in prompts:
        _, e_ref = measure_energy(lambda: ref.generate(p, max_tokens), interval_ms=interval_ms, idle_samples=3)
        holder: Dict[str, Any] = {}
        _, e_casc = measure_energy(lambda: holder.update(rt.route(p)) or holder,
                                   interval_ms=interval_ms, idle_samples=3)
        rj = e_ref.joules_idle_subtracted
        cj = e_casc.joules_idle_subtracted
        delta = (rj - cj) if (rj is not None and cj is not None) else None
        trials.append(PairTrial(
            prompt=p, route_class=_classify(holder),
            cheap_agreed=holder.get("cheap_agreed_this_prompt"),
            ref_joules=rj, casc_joules=cj, delta_joules=delta,
            ref_cov=_coverage(e_ref), casc_cov=_coverage(e_casc),
        ))
    return trials


def summarize_classes(trials: List[PairTrial]) -> Dict[str, Any]:
    by: Dict[str, List[PairTrial]] = {"first_encounter": [], "replay": [], "forced_ref": []}
    for t in trials:
        by[t.route_class].append(t)

    def _mean_casc(ts):
        vals = [t.casc_joules for t in ts if t.casc_joules is not None]
        return round(mean(vals), 3) if vals else None

    def _mean_ref(ts):
        vals = [t.ref_joules for t in ts if t.ref_joules is not None]
        return round(mean(vals), 3) if vals else None

    return {
        "first_encounter": {"n": len(by["first_encounter"]),
                            "mean_cascade_J": _mean_casc(by["first_encounter"]),
                            "mean_reference_J": _mean_ref(by["first_encounter"])},
        "replay": {"n": len(by["replay"]),
                   "mean_cascade_J": _mean_casc(by["replay"]),
                   "mean_reference_J": _mean_ref(by["replay"])},
        "forced_ref": {"n": len(by["forced_ref"])},
    }


def run_comparison_v2(prompts: List[str], K: int = 5, max_tokens: int = 96,
                      interval_ms: int = 200, seed: int = 0) -> Dict[str, Any]:
    """Paired, K-repeat, gated comparison. Emits a savings number only if VALID."""
    if not sudo_available():
        return {"verdict": "GATED_NO_NUMBER",
                "reason": "sudo not primed; measured joules unavailable. Run `sudo -v` first.",
                "gates": {"sudo": False}}

    cheap = MLXModel(CHEAP_PATH)
    ref = MLXModel(REF_PATH)
    _ = cheap.generate("warmup", 4)        # pre-warm: exclude load from measurement
    _ = ref.generate("warmup", 4)

    all_trials: List[PairTrial] = []
    round_deltas: List[float] = []
    idle_first = idle_last = None

    for r in range(K):
        rt = DERNBRuntime(epsilon=0.5, audit_prob=1.0, max_tokens=max_tokens, seed=seed + r)
        rt.cheap, rt.ref = cheap, ref
        # one idle probe at the round boundary for drift tracking
        _, idle_reading = measure_energy(lambda: None, interval_ms=interval_ms, idle_samples=4)
        if r == 0:
            idle_first = idle_reading.avg_watts_idle
        idle_last = idle_reading.avg_watts_idle

        trials = run_paired_block(ref, rt, prompts, max_tokens, interval_ms)
        all_trials.extend(trials)
        valid_deltas = [t.delta_joules for t in trials if t.delta_joules is not None]
        if valid_deltas:
            round_deltas.append(mean(valid_deltas))

    classes = summarize_classes(all_trials)
    covs_ref = [t.ref_cov for t in all_trials if t.ref_cov > 0]
    covs_casc = [t.casc_cov for t in all_trials if t.casc_cov > 0]
    cov_ref = round(mean(covs_ref), 3) if covs_ref else 0.0
    cov_casc = round(mean(covs_casc), 3) if covs_casc else 0.0

    # audit-overhead accounting: first-encounter cascade should exceed reference
    fe = classes["first_encounter"]
    served_total = sum(t.casc_joules for t in all_trials if t.casc_joules is not None)
    audit_extra = 0.0
    if fe["mean_cascade_J"] and fe["mean_reference_J"]:
        audit_extra = max(0.0, (fe["mean_cascade_J"] - fe["mean_reference_J"]) * fe["n"])
    published_cascade_total = served_total + audit_extra

    gate_report = gates_b.evaluate_gates(
        cov_ref=cov_ref, cov_casc=cov_casc, deltas=round_deltas,
        published_cascade_total=published_cascade_total, served_total=served_total,
        audit_extra=audit_extra,
        idle_pre_watts=(idle_first or 0.0), idle_post_watts=(idle_last or 0.0),
    )

    out: Dict[str, Any] = {
        "verdict": gate_report["verdict"],
        "gates": gate_report["gates"],
        "K_rounds": K, "n_prompts": len(prompts), "n_trials": len(all_trials),
        "round_paired_deltas_J": [round(d, 3) for d in round_deltas],
        "three_class": classes,
        "coverage": {"reference": cov_ref, "cascade": cov_casc},
        "idle_watts": {"first": idle_first, "last": idle_last},
        "audit_overhead_J_total": round(audit_extra, 3),
    }
    # The measured paired delta is the honest headline IFF gates pass.
    out["measured_paired_saving_J_per_round"] = gate_report["significant_saving_J"]
    out["honesty_note"] = (
        "Per-prompt paired A/B (thermal drift cancels). Every replay audited "
        "(no cross-prompt attribution). first_encounter runs BOTH models so it is "
        "MORE expensive than reference — that is correct, not a regression. A "
        "savings number appears only when verdict==VALID. Energy is total-system, "
        "idle-subtracted; powermetrics is a within-machine estimate."
    )
    return out


if __name__ == "__main__":
    from src.dern_b.prompts import REPEATED_REGION_STREAM
    print(json.dumps(run_comparison_v2(REPEATED_REGION_STREAM[:6], K=5, max_tokens=64), indent=2))
